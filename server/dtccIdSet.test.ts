// dtccIdSet.test.ts — the EXACT compact set behind the DTCC dedup (2026-11-06
// dtcc-compact REPAIR). A row wrongly treated as seen is a permanent archive
// gap, so these pin exactness first: canonical decimals of every length and at
// every boundary round-trip exactly; everything else ("0123", 20 digits,
// signs, other scripts' digits, non-strings) stays a distinct member in the
// fallback; a randomized corpus and simulated polls get the same answers as a
// plain Set; growth (sync and sliced) never loses or invents a member. Then
// the cost: exact table bytes, ~11 B/ID at the target load, hash quality on
// the live ID shape.
import { test } from "node:test";
import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import v8 from "node:v8";
import vm from "node:vm";

// A real GC for the memory measurement below (2026-11-12): without one, garbage
// from the previous tests' large tables could be collected inside the
// measurement window and shrink the observed arrayBuffers growth (CI, PR #1723:
// 7.66 MB measured for an 11.43 MB table). --expose-gc set at runtime; the
// function comes from a fresh context.
v8.setFlagsFromString("--expose-gc");
const forceGc = vm.runInNewContext("gc") as () => void;
import { CompactIdSet, ID_SET_MAX_LOAD, ID_SET_TARGET_LOAD, idSetCapacityFor } from "./dtccIdSet";

function prng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** The live shape (2026-11-06 /api/data/dtcc-swaps): 19 digits, a rising
 *  10-digit prefix, a 7-digit sequence, "01". */
function liveIds(n: number, seed = 7): string[] {
  const rand = prng(seed);
  const out = new Array<string>(n);
  let prefix = 6100000000, mid = 0;
  for (let i = 0; i < n; i++) {
    prefix += 1 + Math.floor(rand() * 150);
    mid = (mid + 1 + Math.floor(rand() * 40)) % 10_000_000;
    out[i] = `${prefix}${String(mid).padStart(7, "0")}01`;
  }
  return out;
}
const sorted = (it: Iterable<string>) => Array.from(it).sort();

test("canonical decimals of every length 1-19, the boundaries and the live shape are stored exactly and iterate back as the same strings", () => {
  const ids = ["0", "1", "9", "10", "99", "123456789", "999999999", "1000000000", "1000000001", "4294967295", "4294967296",
    "9007199254740991", "9007199254740993", "18446744073709551", "9999999999999999999", "6192117454000002001", "6167534486000000101"];
  for (let len = 1; len <= 19; len++) ids.push("1" + "0".repeat(len - 1), "9".repeat(len));
  const set = new CompactIdSet();
  for (const id of ids) set.add(id);
  const unique = Array.from(new Set(ids));
  assert.equal(set.size, unique.length);
  assert.equal(set.numericSize, unique.length, "every one is a canonical decimal, so none falls back");
  assert.equal(set.fallbackSize, 0);
  for (const id of unique) assert.equal(set.has(id), true, id);
  assert.deepEqual(sorted(set), unique.slice().sort(), "decoding is the exact inverse: same strings out");
  // neighbors of the stored values are NOT members (a lossy key would collide here)
  for (const absent of ["2", "11", "1000000002", "9007199254740992", "9999999999999999998", "6192117454000002002", "6192117454000002000"]) {
    assert.equal(set.has(absent), false, absent);
  }
});

test("everything that is not a canonical decimal is its own member in the fallback and never collides with a numeric one ('0123' vs '123', '00' vs '0', 20 digits, 2^64, signs, spaces, other scripts' digits)", () => {
  const pairs: Array<[string, string]> = [
    ["0123", "123"], ["00", "0"], ["007", "7"], ["0000000000000000001", "1"],
    ["12345678901234567890", "1234567890123456789"], ["18446744073709551616", "1844674407370955161"],
    ["-1", "1"], ["+1", "1"], ["1e3", "1000"], [" 1", "1"], ["1 ", "1"], ["1.0", "1"], ["1\r", "1"],
    ["１２３", "123"], ["١٢٣", "123"], ["6192117454000002001x", "6192117454000002001"],
  ];
  for (const [weird, canonical] of pairs) {
    const set = new CompactIdSet();
    set.add(weird);
    assert.equal(set.has(weird), true, `${JSON.stringify(weird)} is a member`);
    assert.equal(set.has(canonical), false, `${JSON.stringify(weird)} must not make ${canonical} a member`);
    set.add(canonical);
    assert.equal(set.size, 2, `${JSON.stringify(weird)} and ${canonical} are two members`);
    assert.equal(set.fallbackSize, 1);
    assert.deepEqual(sorted(set), [weird, canonical].sort());
  }
  const misc = new CompactIdSet();
  for (const v of ["", "A1", "ünïcödé-字", "x"]) misc.add(v);
  assert.equal(misc.size, 4);
  assert.equal(misc.has(""), true);
  assert.equal(misc.has("a1"), false);
  // non-strings (a malformed archive row's JSON can hand one over) keep plain-Set semantics
  const loose = new CompactIdSet();
  const undef = (JSON.parse("{}") as { disseminationId: string }).disseminationId;
  loose.add(undef);
  loose.add(JSON.parse("123") as string);
  assert.equal(loose.size, 2);
  assert.equal(loose.has(undef), true);
  assert.equal(loose.has("123"), false, "the number 123 is not the string \"123\", as in a Set");
});

test("randomized corpus: exactly the answers of a plain Set<string> over 300k mixed adds (by string and by bytes) and lookups — numeric, leading-zero, over-long, alphanumeric, odd and repeated ids", () => {
  const rand = prng(20261106);
  const digits = (n: number) => { let s = ""; for (let i = 0; i < n; i++) s += String(Math.floor(rand() * 10)); return s; };
  const pool: string[] = [];
  const nextId = (): string => {
    const r = rand();
    if (r < 0.3) { const len = 1 + Math.floor(rand() * 19); return String(1 + Math.floor(rand() * 9)) + digits(len - 1); }
    if (r < 0.38) return "0" + digits(Math.floor(rand() * 19));
    if (r < 0.43) return String(1 + Math.floor(rand() * 9)) + digits(19 + Math.floor(rand() * 3));
    if (r < 0.5) return "A" + digits(1 + Math.floor(rand() * 8));
    if (r < 0.53) return ["", " 7", "-7", "+7", "7e1", "７", "07", "0"][Math.floor(rand() * 8)];
    return pool.length ? pool[Math.floor(rand() * pool.length)] : "0";
  };
  const set = new CompactIdSet();
  const ref = new Set<string>();
  for (let op = 0; op < 300_000; op++) {
    const id = nextId();
    if (rand() < 0.6) {
      if (rand() < 0.5) set.add(id); else { const b = Buffer.from(id, "utf8"); set.addBytes(b, 0, b.length); }
      ref.add(id);
      pool.push(id);
      assert.equal(set.size, ref.size);
    } else {
      assert.equal(set.has(id), ref.has(id), `has(${JSON.stringify(id)})`);
    }
  }
  assert.deepEqual(sorted(set), sorted(ref));
  assert.ok(set.numericSize > 30_000 && set.fallbackSize > 10_000, `both halves exercised (${set.numericSize} numeric, ${set.fallbackSize} fallback)`);
});

test("identical dedup decisions to a plain Set across simulated polls: re-served, fresh and in-poll duplicate ids", () => {
  const rand = prng(42);
  const archived = liveIds(50_000, 1);
  const set = new CompactIdSet(archived.length);
  const ref = new Set<string>();
  for (const id of archived) { set.add(id); ref.add(id); }
  const fresh = liveIds(40_000, 2).map((id) => "7" + id.slice(1)); // a different id range
  let next = 0;
  for (let poll = 0; poll < 8; poll++) {
    const rows: string[] = [];
    for (let k = 0; k < 6_000; k++) {
      const r = rand();
      rows.push(r < 0.6 ? archived[Math.floor(rand() * archived.length)] : r < 0.95 ? fresh[next++ % fresh.length] : (rows[Math.floor(rand() * rows.length)] ?? "0123"));
    }
    const ours = rows.filter((id) => !set.has(id));
    const theirs = rows.filter((id) => !ref.has(id));
    assert.deepEqual(ours, theirs, `poll ${poll}: the same rows judged fresh, in the same order (duplicates included)`);
    for (const id of ours) { set.add(id); ref.add(id); }
    assert.equal(set.size, ref.size);
  }
});

test("growth from the minimum capacity to 300k live-shaped ids: every member kept, absent ids absent, fill never above the max load, iteration exact", () => {
  const ids = liveIds(300_000, 3);
  const set = new CompactIdSet();
  const startCap = set.tableCapacity;
  for (const id of ids) {
    set.add(id);
    assert.ok(set.numericSize <= Math.floor(set.tableCapacity * ID_SET_MAX_LOAD));
  }
  assert.ok(set.tableCapacity > startCap, "it resized");
  assert.equal(set.size, ids.length);
  for (let i = 0; i < ids.length; i += 7) assert.equal(set.has(ids[i]), true);
  for (const id of liveIds(20_000, 4).map((s) => "5" + s.slice(1))) assert.equal(set.has(id), false);
  assert.deepEqual(sorted(set), ids.slice().sort());
});

test("a set sized for N never resizes while it fills to N, and sits at the target load", () => {
  const n = 100_000;
  const set = new CompactIdSet(n);
  const cap = set.tableCapacity;
  assert.equal(cap, idSetCapacityFor(n));
  for (const id of liveIds(n, 5)) set.add(id);
  assert.equal(set.tableCapacity, cap, "no resize");
  assert.ok(Math.abs(n / cap - ID_SET_TARGET_LOAD) < 0.001);
});

test("reserveAsync makes room in slices while the set keeps answering, and stays exact when members arrive mid-copy (the copy is discarded, the resize redone)", async () => {
  const base = liveIds(200_000, 6);
  for (const midCopyAdds of [false, true]) {
    const set = new CompactIdSet(base.length);
    for (const id of base) set.add(id);
    const capBefore = set.tableCapacity;
    let answeredDuringResize = 0;
    let addedDuring = 0;
    const extra = liveIds(150_000, 8).map((s) => "4" + s.slice(1));
    let resizing = true;
    const poke = () => {
      if (!resizing) return;
      if (set.has(base[answeredDuringResize % base.length])) answeredDuringResize++;
      if (midCopyAdds && addedDuring < 50) set.add(extra[addedDuring++]);
      setImmediate(poke);
    };
    setImmediate(poke);
    await set.reserveAsync(150_000, 1);
    resizing = false;
    assert.ok(set.tableCapacity > capBefore, "the table grew");
    assert.ok(answeredDuringResize > 0, "lookups were answered while the resize ran");
    for (const id of extra) set.add(id);
    assert.equal(set.size, base.length + extra.length);
    assert.deepEqual(sorted(set), [...base, ...extra].sort(), `midCopyAdds=${midCopyAdds}: nothing lost, nothing invented`);
  }
});

test("addBytes == add(buf.toString('utf8', ...)) for digits, multi-byte text and invalid UTF-8", () => {
  const lines: Buffer[] = [
    Buffer.from("6192117454000002001"), Buffer.from("0123"), Buffer.from("123"), Buffer.from("ünïcödé-字-🙂"),
    Buffer.from([0x31, 0x32, 0xc3]), Buffer.from([0x80, 0x31]), Buffer.from([0xff]), Buffer.from([0x31, 0x00, 0x32]),
  ];
  const viaBytes = new CompactIdSet();
  const viaString = new CompactIdSet();
  const ref = new Set<string>();
  for (const b of lines) {
    const padded = Buffer.concat([Buffer.from("xx"), b, Buffer.from("yy")]);
    viaBytes.addBytes(padded, 2, 2 + b.length);
    viaString.add(b.toString("utf8"));
    ref.add(b.toString("utf8"));
  }
  assert.deepEqual(sorted(viaBytes), sorted(ref));
  assert.deepEqual(sorted(viaString), sorted(ref));
});

test("hash quality on the live id shape: at the max load the longest probe stays in the range random keys give", () => {
  const n = 400_000;
  const longest = (ids: string[]) => {
    const set = new CompactIdSet(n * ID_SET_TARGET_LOAD / ID_SET_MAX_LOAD); // capacity ≈ n / max load
    for (const id of ids) set.add(id);
    assert.ok(set.numericSize / set.tableCapacity > ID_SET_MAX_LOAD - 0.01);
    return set._maxProbeForTests();
  };
  const rand = prng(99);
  const randomIds = Array.from({ length: n }, () => {
    let s = String(1 + Math.floor(rand() * 9));
    for (let i = 0; i < 18; i++) s += String(Math.floor(rand() * 10));
    return s;
  });
  const live = longest(liveIds(n, 9));
  const random = longest(Array.from(new Set(randomIds)));
  console.log(`[bench] dtccIdSet longest probe at load ${ID_SET_MAX_LOAD}: live-shaped ${live}, random ${random}`);
  assert.ok(live <= random * 3 + 64, `live-shaped ${live} vs random ${random}`);
});

test("memory: tableBytes is exact (capacity x 8) and, sized for 1M live-shaped ids, the set costs ~11.4 B/id in typed-array memory", () => {
  const n = 1_000_000;
  const ids = liveIds(n, 10);
  forceGc(); // earlier tests' tables are collected BEFORE the baseline, not inside the window
  const ab0 = process.memoryUsage().arrayBuffers;
  const t0 = performance.now();
  const set = new CompactIdSet(n);
  for (const id of ids) set.add(id);
  const buildMs = performance.now() - t0;
  forceGc(); // only what the set retains is counted
  const ab1 = process.memoryUsage().arrayBuffers;
  assert.equal(set.tableBytes(), set.tableCapacity * 8);
  const perId = set.tableBytes() / n;
  assert.ok(perId <= 8 / ID_SET_TARGET_LOAD + 0.01, `${perId.toFixed(2)} B/id`);
  // The set's typed-array memory, read from the set itself: every ArrayBuffer view it holds, summed, must equal
  // tableBytes exactly, with no build table left over. (Until 2026-11-12 this compared the process-wide
  // process.memoryUsage().arrayBuffers delta to tableBytes within 10%; on some CI runners that counter came out
  // 7,663,864 B for an 11,428,576 B table on three runs — PRs #1723, #1744 — even with a GC around the window,
  // while other runs and every local run matched. A process-wide counter is not the set's memory; the views are.
  // The delta stays in the bench line below for the record.)
  const own = Object.values(set as unknown as Record<string, unknown>).filter((v): v is ArrayBufferView => ArrayBuffer.isView(v));
  assert.equal(own.reduce((sum, v) => sum + v.byteLength, 0), set.tableBytes(), "the set's own typed arrays are exactly tableBytes");
  assert.equal(own.length, 1, "one table, no build table left behind");
  const t1 = performance.now();
  let hits = 0;
  for (let i = 0; i < 136_701; i++) if (set.has(ids[(i * 7919) % n])) hits++;
  const lookupMs = performance.now() - t1;
  assert.equal(hits, 136_701);
  console.log(`[bench] dtccIdSet 1M live-shaped ids: table ${(set.tableBytes() / 1e6).toFixed(1)} MB (${perId.toFixed(2)} B/id), process arrayBuffers +${ab1 - ab0} B, build ${buildMs.toFixed(0)} ms, 136,701 lookups ${lookupMs.toFixed(1)} ms`);
});
