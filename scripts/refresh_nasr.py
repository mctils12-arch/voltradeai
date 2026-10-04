#!/usr/bin/env python3
"""One-command FAA NASR 28-day cycle refresh (compiled from the manual 2026-10-04 refresh).

  python3 scripts/refresh_nasr.py --check     # offline; exit 0 current, 2 stale
  python3 scripts/refresh_nasr.py             # download current cycle, rebuild, print diff
  python3 scripts/refresh_nasr.py --zip F.zip # rebuild from a local bundle

Rebuilds datacore/aircraft/nasr_fixes.json and nasr_airways.json via the existing
builders (public-domain data). Staleness is judged on the FIX cycle only: the FAA
does not republish airways every cycle (AWY_BASE EFF_DATE can lag), so an unchanged
airways cycle is normal, not a failure. Commit the result as its own data PR.
"""
import datetime, importlib.util, io, json, os, sys, tempfile, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
FIXES = os.path.join(HERE, "..", "datacore", "aircraft", "nasr_fixes.json")
ANCHOR = datetime.date(2026, 10, 1)  # a known NASR effective date; cycles are 28 days
URL = "https://nfdc.faa.gov/webContent/28DaySub/extra/{d:%d}_{m}_{d:%Y}_CSV.zip"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def current_cycle(today):
    return ANCHOR + datetime.timedelta(days=((today - ANCHOR).days // 28) * 28)


def bundle_url(cycle):
    return URL.format(d=cycle, m=MONTHS[cycle.month - 1])


def committed_cycle(path=FIXES):
    with open(path) as f:
        return datetime.date.fromisoformat(json.load(f)["cycle"])


def diff_fixes(old, new, tol=1e-4):
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    moved = sorted(k for k in set(old) & set(new)
                   if abs(old[k][0] - new[k][0]) > tol or abs(old[k][1] - new[k][1]) > tol)
    return added, removed, moved


def _builder(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rebuild(zip_bytes):
    with open(FIXES) as f:
        old = json.load(f)["fixes"]
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(io.BytesIO(zip_bytes)).extractall(tmp)
        _builder("build_nasr_fixes").main(tmp)
        _builder("build_nasr_airways").main(tmp)
    with open(FIXES) as f:
        new = json.load(f)["fixes"]
    a, r, m = diff_fixes(old, new)
    print(f"fixes diff: +{len(a)} -{len(r)} moved {len(m)}; e.g. added {a[:3]} removed {r[:3]}")


def main(argv):
    cur, have = current_cycle(datetime.date.today()), committed_cycle()
    stale = have < cur
    print(f"committed fix cycle {have}; current FAA cycle {cur} -> {'STALE' if stale else 'current'}")
    if "--check" in argv:
        return 2 if stale else 0
    if "--zip" in argv:
        data = open(argv[argv.index("--zip") + 1], "rb").read()
    elif stale:
        data = urllib.request.urlopen(bundle_url(cur), timeout=120).read()
    else:
        return 0
    rebuild(data)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
