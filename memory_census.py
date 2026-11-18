"""
Memory census — WHO holds the daemon's Python objects (KNOWN BROKEN #55).

The 2026-11-18 16:12Z and 16:28Z readings (daemon up 1.3 to 1.6 h, 20 to 23
scans, deep scoring trimmed on all but the first two) said WHAT the memory
is: 3.8 to 3.9 M interpreter blocks against 0.29 M at start, the C heap only
150 to 170 MB of a 505 to 520 MB process. So live Python objects, and the
count flattens after the first hour, the shape of caches filling to a cap.
They could not say WHO holds them: `memory.largest_containers` lists
module-level containers by LENGTH in this repo's modules only, and the
longest was 1,193 entries. A cache of 50 keys whose values are each 40,000
objects, an instance attribute, a closure or a library's own cache is
invisible to it.

This module answers that, on demand and read-only:

- `types`: every object the garbage collector tracks, counted by type.
- `holders`: every module-level name in EVERY loaded module (repo and
  libraries) and every class attribute, with the number of objects reachable
  from it and their shallow bytes. An object reachable from two names is
  counted under the first.
- `holders` also lists function closures and the locals of every thread's
  running frames, which no module-level name reaches.
- `unreached`: tracked objects none of those reach, by type, and for the
  longest such containers the chain of referrers upward until a module, a
  class, a function or a running frame names the owner.
- `root="module.attr"` descends one level: the same count per key of a dict
  or per attribute of an instance.

It reports names, types, lengths and counts. It never emits a value, and a
dict key that looks like a URL is reported by length only.

COST: it holds the GIL while it walks (bounded by `budget_sec` and
`max_objects`), so the daemon runs it only when asked and not during a scan
(voltrade_daemon `_memory_census`). No trading module imports this one.
"""
import collections
import gc
import sys
import time
import types as _types

# Walked into. Everything else reachable is counted but not descended,
# unless it is a plain instance (see _descend).
_CONTAINERS = (dict, list, tuple, set, frozenset, collections.deque)
# Never counted and never walked into: these lead to the whole interpreter
# (a function's globals, a module's namespace, a frame's locals).
_SKIP = (_types.ModuleType, type, _types.FunctionType, _types.BuiltinFunctionType,
         _types.MethodType, _types.CodeType, _types.FrameType, _types.TracebackType,
         _types.GetSetDescriptorType, _types.MemberDescriptorType,
         _types.WrapperDescriptorType, _types.MethodDescriptorType,
         _types.MethodWrapperType, staticmethod, classmethod, property)
_ATOMS = (int, float, str, bytes, bool, type(None), complex)

DEFAULT_MAX_OBJECTS = 1_500_000   # ids remembered in all: ~90 MB while it runs
DEFAULT_PER_ROOT = 300_000        # one name may claim this many before it is cut
DEFAULT_BUDGET_SEC = 8.0
_KEY_CHARS = 48


def _type_name(obj) -> str:
    t = type(obj)
    mod = getattr(t, "__module__", "") or ""
    return t.__name__ if mod in ("builtins", "") else f"{mod}.{t.__name__}"


def _class_name(cls) -> str:
    return f"{type.__getattribute__(cls, '__module__')}.{type.__getattribute__(cls, '__qualname__')}"


def _key_label(key) -> str:
    """A dict key as a label: a short string as itself, a URL by its length
    only (a query string may carry a credential), anything else by type."""
    if type(key) is str:
        if "://" in key or "?" in key:
            return f"<url-like key, {len(key)} chars>"
        return key[:_KEY_CHARS]
    if type(key) in (int, float, bool) or key is None:
        return repr(key)[:_KEY_CHARS]
    if type(key) is tuple and len(key) <= 4 and all(type(k) in (str, int, float) for k in key):
        return "(" + ", ".join(_key_label(k) for k in key) + ")"
    return f"<{_type_name(key)}>"


# THE WALK NEVER CALLS INTO WHAT IT MEASURES. No len(), no sys.getsizeof()
# and no attribute read on an arbitrary object: each of those runs the
# object's own code (a mock answers every one of them, a lazy proxy imports
# on attribute access, pandas' __sizeof__ reads every cell). Lengths and
# sizes come from the builtin type's own slot; everything else is followed
# with gc.get_referents, the collector's own traversal.
_EXACT_SIZED = frozenset((dict, list, tuple, set, frozenset, collections.deque,
                          int, float, str, bytes, bytearray, bool, type(None), complex))
_LEN_OF = ((dict, dict.__len__), (list, list.__len__), (tuple, tuple.__len__),
           (set, set.__len__), (frozenset, frozenset.__len__),
           (collections.deque, collections.deque.__len__))


def _safe_len(obj):
    for base, fn in _LEN_OF:
        if isinstance(obj, base):
            return fn(obj)
    return None


def _safe_size(obj) -> int:
    t = type(obj)
    if t in _EXACT_SIZED or t.__module__ == "numpy":
        return sys.getsizeof(obj)
    return t.__basicsize__


def _is_instance_with_dict(obj) -> bool:
    return not isinstance(obj, _ATOMS) and getattr(type(obj), "__dictoffset__", 0) != 0


def _descend(obj) -> bool:
    # A dict or tuple of plain numbers and strings is not tracked by the
    # collector, and is still a container to walk.
    return isinstance(obj, _CONTAINERS) or (not isinstance(obj, _ATOMS) and gc.is_tracked(obj))


def _is_root(obj) -> bool:
    """Worth its own row: a container, or an instance that carries state."""
    return isinstance(obj, _CONTAINERS) or (
        not isinstance(obj, _SKIP) and not isinstance(obj, _ATOMS) and gc.is_tracked(obj))


class _Walk:
    """One census's shared state: the ids already counted and the clock."""

    def __init__(self, max_objects, per_root, budget_sec, clock=time.monotonic, barrier=()):
        self.seen = set()
        # Namespaces (module and class dicts): reaching one from an instance
        # or a compiled function would bill the whole program to that name.
        self.barrier = frozenset(barrier)
        self.max_objects = max_objects
        self.per_root = per_root
        self.deadline = clock() + budget_sec
        self.clock = clock
        self.stopped = None   # "max_objects" | "budget_sec" once the census is cut short

    def measure(self, obj):
        """(objects, shallow bytes, truncated) reachable from obj and not
        counted before. Iterative: a deep structure cannot overflow the stack."""
        seen = self.seen
        count = size = 0
        truncated = False
        stack = [obj]
        while stack:
            o = stack.pop()
            oid = id(o)
            if oid in seen or oid in self.barrier or isinstance(o, _SKIP):
                continue
            seen.add(oid)
            count += 1
            size += _safe_size(o)
            if count >= self.per_root:
                truncated = True
                break
            if not (count & 0xFFF):
                if len(seen) >= self.max_objects:
                    self.stopped = "max_objects"
                elif self.clock() > self.deadline:
                    self.stopped = "budget_sec"
                if self.stopped:
                    truncated = True
                    break
            if _descend(o):
                stack.extend([r for r in gc.get_referents(o) if id(r) not in seen])
                if len(stack) > 4 * self.per_root:
                    truncated = True   # a fan-out wider than the cap itself
                    break
        return count, size, truncated


def _row(name, obj, walk):
    n, size, cut = walk.measure(obj)
    row = {"name": name, "type": _type_name(obj), "objects": n, "mb": round(size / 1048576, 2)}
    if _safe_len(obj) is not None:
        row["len"] = _safe_len(obj)
    if cut:
        row["truncated"] = True
    return row


def _module_roots(modules):
    """(name, object) for every module-level name and every class attribute
    worth walking, in a stable order. Dunder names are skipped (`__builtins__`
    would bill the builtins to whichever module came first)."""
    for mod_name in sorted(modules):
        mod = modules[mod_name]
        if not isinstance(mod, _types.ModuleType):
            continue
        for attr, val in list(vars(mod).items()):
            if attr.startswith("__"):
                continue
            if isinstance(val, type):
                if type.__getattribute__(val, "__module__") != mod_name:
                    continue
                for cattr, cval in list(type.__getattribute__(val, "__dict__").items()):
                    if not cattr.startswith("__") and _is_root(cval):
                        yield f"{mod_name}.{attr}.{cattr}", cval
            elif _is_root(val):
                yield f"{mod_name}.{attr}", val


def _resolve(root, modules):
    """The object a dotted `module.attr[.attr]` names: the longest module
    prefix found in `modules`, then plain attribute reads."""
    parts = root.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        mod = modules.get(".".join(parts[:cut]))
        if mod is None:
            continue
        obj = mod
        for attr in parts[cut:]:
            try:
                obj = dict(vars(obj)).get(attr, _MISSING)
            except TypeError:
                obj = _MISSING
            if obj is _MISSING:
                break
        if obj is not _MISSING:
            return obj
    return _MISSING


_MISSING = object()


def _children(obj):
    """(label, child) pairs one level under obj: a dict's values by key, an
    instance's attributes, a sequence's elements grouped by type."""
    if isinstance(obj, dict):
        return [(f"[{_key_label(k)}]", v) for k, v in list(dict.items(obj))]
    if isinstance(obj, (list, tuple, set, frozenset, collections.deque)):
        groups = collections.OrderedDict()
        for el in [r for r in gc.get_referents(obj) if not isinstance(r, type)]:
            groups.setdefault(_type_name(el), []).append(el)
        return [(f"<{len(els)} x {tname}>", els) for tname, els in groups.items()]
    if _is_instance_with_dict(obj):
        for ref in gc.get_referents(obj):
            if type(ref) is dict:
                return [(f".{_key_label(k)}", v) for k, v in list(ref.items())]
    return []


def _owner_chain(obj, module_dicts, class_dicts, ignore, max_depth=8):
    """Referrers of obj, upward, as labels, until something with a name owns
    it. Each step takes the first referrer that is not this census's own
    bookkeeping."""
    chain = []
    cur = obj
    ignore_ids = {id(x) for x in ignore}
    for _ in range(max_depth):
        refs = gc.get_referrers(cur)
        ignore_ids.add(id(refs))
        nxt = None
        frame = None
        for r in refs:
            if id(r) in ignore_ids:
                continue
            if isinstance(r, _types.FrameType):
                if r.f_code.co_filename != __file__:
                    frame = frame or r
                continue
            nxt = r
            break
        if nxt is None:
            if frame is not None:
                chain.append(f"frame {frame.f_code.co_name} ({frame.f_code.co_filename.rsplit('/', 1)[-1]})")
            else:
                chain.append("no referrer found")
            break
        rid = id(nxt)
        if rid in module_dicts:
            chain.append(f"module {module_dicts[rid]}{_held_as(nxt, cur)}")
            break
        if rid in class_dicts:
            chain.append(f"class {class_dicts[rid]}{_held_as(nxt, cur)}")
            break
        if isinstance(nxt, _types.FunctionType):
            chain.append(f"function {nxt.__module__}.{nxt.__qualname__}")
            break
        if isinstance(nxt, dict):
            chain.append(f"dict(len={dict.__len__(nxt)}){_held_as(nxt, cur)}")
        else:
            label = _type_name(nxt)
            if _safe_len(nxt) is not None:
                label += f"(len={_safe_len(nxt)})"
            chain.append(label)
        ignore_ids.add(id(cur))
        cur = nxt
        del refs
    return chain


def _held_as(d, child, scan=20000):
    """' as [key]' when `child` is a value of dict `d` (first `scan` items)."""
    for i, (k, v) in enumerate(list(dict.items(d))):
        if v is child:
            return f" as [{_key_label(k)}]"
        if i >= scan:
            break
    return ""


def census(root=None, max_objects=DEFAULT_MAX_OBJECTS, per_root=DEFAULT_PER_ROOT,
           budget_sec=DEFAULT_BUDGET_SEC, top=25, samples=5, modules=None,
           clock=time.monotonic):
    """See the module docstring. `modules` defaults to sys.modules (tests
    pass their own). Returns a dict of plain JSON types; the daemon's
    dispatcher turns anything raised into an error reply."""
    started = clock()
    modules = dict(sys.modules if modules is None else modules)
    barrier = set()
    for mod in modules.values():
        if isinstance(mod, _types.ModuleType):
            barrier.add(id(vars(mod)))
    walk = _Walk(int(max_objects), int(per_root), float(budget_sec), clock, barrier)
    out = {"python_blocks": sys.getallocatedblocks()}

    if root:
        obj = _resolve(str(root), modules)
        if obj is _MISSING:
            return {"error": f"no such name: {str(root)[:120]}"}
        own = {"name": str(root), "type": _type_name(obj)}
        if _safe_len(obj) is not None:
            own["len"] = _safe_len(obj)
        rows = []
        for label, child in _children(obj):
            if walk.stopped:
                break
            rows.append(_row(label, child, walk))
        rows.sort(key=lambda r: (-r["objects"], r["name"]))
        own["objects"] = sum(r["objects"] for r in rows)
        own["mb"] = round(sum(r["mb"] for r in rows), 2)
        out.update({"root": own, "children": rows[:top], "children_total": len(rows)})
    else:
        rows = []
        for name, obj in _module_roots(modules):
            if walk.stopped:
                break
            r = _row(name, obj, walk)
            if r["objects"] >= 1000 or r.get("truncated"):
                rows.append(r)
        objs = gc.get_objects()

        # Holders that are not module-level names: a function's closure
        # variables and the locals of every thread's running frames.
        for fn in objs:
            if walk.stopped:
                break
            if not isinstance(fn, _types.FunctionType) or not fn.__closure__:
                continue
            for var, cell in zip(fn.__code__.co_freevars, fn.__closure__):
                held = gc.get_referents(cell)   # empty for a cell not yet filled
                if not held:
                    continue
                val = held[0]
                if id(val) in walk.seen or not _is_root(val):
                    continue
                r = _row(f"closure {fn.__module__}.{fn.__qualname__}:{var}", val, walk)
                if r["objects"] >= 1000 or r.get("truncated"):
                    rows.append(r)
        frames = list(sys._current_frames().values())
        for frame in frames:
            while frame is not None and not walk.stopped:
                if frame.f_code.co_filename != __file__:
                    for var, val in list(frame.f_locals.items()):
                        if val is objs or id(val) in walk.seen or not _is_root(val):
                            continue
                        r = _row(f"frame {frame.f_code.co_name}:{var}", val, walk)
                        if r["objects"] >= 1000 or r.get("truncated"):
                            rows.append(r)
                frame = frame.f_back
        del frames

        rows.sort(key=lambda r: (-r["objects"], r["name"]))
        out["holders"] = rows[:top]
        out["holders_over_1000"] = len(rows)

        # Tracked objects by type, and the ones nothing above reached. The
        # collector does not track a dict or tuple that holds only numbers
        # and strings, so these are counts of containers, not of blocks.
        by_type = collections.Counter()
        unreached = collections.Counter()
        longest = []   # (len, index) of the longest unreached containers
        seen = walk.seen
        for i, o in enumerate(objs):
            if isinstance(o, _SKIP) or o is seen or o is rows:
                continue
            tn = _type_name(o)
            by_type[tn] += 1
            if id(o) not in seen:
                unreached[tn] += 1
                if isinstance(o, _CONTAINERS):
                    n = _safe_len(o)
                    if n >= 1000:
                        longest.append((n, i))
        out["types"] = [{"type": t, "count": c} for t, c in by_type.most_common(20)]
        out["tracked_objects"] = sum(by_type.values())
        out["unreached"] = [{"type": t, "count": c} for t, c in unreached.most_common(12)]
        out["unreached_total"] = sum(unreached.values())

        # Who owns the longest containers nothing above reached: the chain
        # of referrers upward. gc.get_referrers reads the whole heap per
        # call, so this is the part the time budget cuts first.
        chains = []
        longest.sort(reverse=True)
        if samples > 0 and longest and not walk.stopped:
            module_dicts = {id(vars(m)): n for n, m in modules.items() if isinstance(m, _types.ModuleType)}
            class_dicts = {}
            for o in objs:
                if isinstance(o, type):
                    # vars(cls) is a proxy; the dict the collector reports
                    # as a referrer is the one behind it.
                    for ref in gc.get_referents(o):
                        if isinstance(ref, dict):
                            class_dicts[id(ref)] = _class_name(o)
            picked = [objs[i] for _, i in longest[:8 * samples]]
            ignore = [objs, picked, chains, longest]
            owners = set()
            for k in range(len(picked)):
                if len(chains) >= samples:
                    break
                if clock() > walk.deadline:
                    walk.stopped = "budget_sec"
                    break
                if id(picked[k]) in walk.seen:
                    continue   # inside a longer one already measured
                held_by = _owner_chain(picked[k], module_dicts, class_dicts, ignore)
                if held_by[-1] in owners:
                    continue   # one row per owner: a cut-short holder's tail repeats
                owners.add(held_by[-1])
                r = _row("unreached", picked[k], walk)
                r["held_by"] = held_by
                del r["name"]
                chains.append(r)
            del picked
        out["unreached_owners"] = chains
        del objs

    out["objects_walked"] = len(walk.seen)
    out["stopped_early"] = walk.stopped
    out["limits"] = {"max_objects": walk.max_objects, "per_root": walk.per_root, "budget_sec": float(budget_sec)}
    out["elapsed_sec"] = round(clock() - started, 2)
    return out
