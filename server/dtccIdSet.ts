/**
 * dtccIdSet.ts — an EXACT, compact set of DTCC Dissemination Identifiers
 * (2026-11-06 dtcc-compact REPAIR, T-DATACORE).
 *
 * WHY. dtccSwaps.ts dedups the vendor's cumulative file against every ID it
 * ever archived. With the dedup DB refused by the volume guard (the volume is
 * short of space), that set lives on the heap: live 2026-11-06 /api/health
 * reported dtcc.seenIds = 6,939,910 entries, ~0.5 GB (a JS Set<string> costs
 * 64-76 B per ID, measured on a synthetic 6.94M-ID index the same day).
 *
 * THE ID FORMAT, from evidence:
 *  - Live values (2026-11-06 /api/data/dtcc-swaps top_rows, 100 of 100): 19
 *    decimal digits, no leading zero, e.g. 6192117454000002001 — all above
 *    2^53, so a float64 cannot hold them exactly.
 *  - The governing public spec (CFTC Parts 43/45 Technical Specification
 *    v3.3, Appendix A, D1 "Dissemination identifier", the field DTCC's
 *    SBSDR file disseminates under the same name): "SDR generated unique and
 *    random identifier for each publicly disseminated message", Varchar(52),
 *    "Up to 52 alphanumeric characters" — so digits are NOT guaranteed.
 *
 * THE STRUCTURE. A canonical decimal — "0", or 1 to 19 digits with no leading
 * zero — is a number below 10^19 < 2^64, stored EXACTLY as two uint32 words
 * (no BigInt): v = a*10^9 + b with b < 10^9; hi = floor(a/4), lo = (a%4)*2^30
 * + b. That map is a bijection (b < 2^30), so two different canonical strings
 * can never share a key. The words sit in an open-addressing, linear-probing
 * table (one interleaved Uint32Array, 8 bytes a slot, any capacity) kept
 * between ID_SET_TARGET_LOAD and ID_SET_MAX_LOAD full — ~11.4 B/ID at 0.7.
 * EVERY other value (non-digits, "0123", over 19 digits, non-strings) goes to
 * a plain Set, so "0123" and "123" stay two members, exactly as in a
 * Set<string>. No hashing decides membership: a hash only picks where the
 * probe starts, and a hit compares the full 64-bit key — no false positives.
 * (A row wrongly treated as seen would be a permanent archive gap.)
 *
 * Iteration yields the fallback values (insertion order), then the numeric IDs
 * as their canonical strings, in table order: deterministic for the same
 * capacity and insertion sequence, but not insertion order.
 */

/** A hi word no key can have: the largest real one is floor((10^10 - 1) / 4) = 2,499,999,999. */
const EMPTY = 0xffffffff;
/** Grow once the numeric table would pass this fill. */
export const ID_SET_MAX_LOAD = 0.85;
/** New and resized tables are sized so the current count sits at this fill. */
export const ID_SET_TARGET_LOAD = 0.7;
const MIN_CAPACITY = 1024;
const TWO_POW_30 = 1073741824;
const TWO_POW_32 = 4294967296;

export function idSetCapacityFor(count: number): number {
  return Math.max(MIN_CAPACITY, Math.ceil(count / ID_SET_TARGET_LOAD));
}

// parse output (no allocation per call; read right after a successful parse)
let keyHi = 0;
let keyLo = 0;

/** true when `id` is a canonical decimal ("0", or 1-19 digits with no
 *  leading zero); its exact value is then in keyHi/keyLo. */
function parseCanonical(id: string): boolean {
  const n = id.length;
  if (n === 0 || n > 19) return false;
  const first = id.charCodeAt(0);
  if (first === 48) {
    if (n !== 1) return false;
    keyHi = 0; keyLo = 0;
    return true;
  }
  if (first < 49 || first > 57) return false;
  const split = n > 9 ? n - 9 : 0;
  let a = 0;
  for (let i = 0; i < split; i++) {
    const d = id.charCodeAt(i) - 48;
    if (d < 0 || d > 9) return false;
    a = a * 10 + d;
  }
  let b = 0;
  for (let i = split; i < n; i++) {
    const d = id.charCodeAt(i) - 48;
    if (d < 0 || d > 9) return false;
    b = b * 10 + d;
  }
  keyHi = Math.floor(a / 4);
  keyLo = (a % 4) * TWO_POW_30 + b;
  return true;
}

/** parseCanonical over the bytes [start, end) of a UTF-8 buffer: canonical
 *  decimals are pure ASCII, so the bytes are the characters. */
function parseCanonicalBytes(buf: Uint8Array, start: number, end: number): boolean {
  const n = end - start;
  if (n <= 0 || n > 19) return false;
  const first = buf[start];
  if (first === 48) {
    if (n !== 1) return false;
    keyHi = 0; keyLo = 0;
    return true;
  }
  if (first < 49 || first > 57) return false;
  const split = n > 9 ? start + n - 9 : start;
  let a = 0;
  for (let i = start; i < split; i++) {
    const d = buf[i] - 48;
    if (d < 0 || d > 9) return false;
    a = a * 10 + d;
  }
  let b = 0;
  for (let i = split; i < end; i++) {
    const d = buf[i] - 48;
    if (d < 0 || d > 9) return false;
    b = b * 10 + d;
  }
  keyHi = Math.floor(a / 4);
  keyLo = (a % 4) * TWO_POW_30 + b;
  return true;
}

/** The canonical decimal string of a stored key (exact inverse of the parse). */
function decodeKey(hi: number, lo: number): string {
  const a = hi * 4 + (lo >>> 30);
  const b = lo & 0x3fffffff;
  return a > 0 ? `${a}${String(b).padStart(9, "0")}` : String(b);
}

function fmix32(x: number): number {
  let h = x;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/** Where a key's probe starts: both words mixed (murmur3 finalizer), mapped
 *  onto [0, capacity) by multiply-shift, so the capacity need not be a power
 *  of two. Membership never depends on this — see the header. */
function homeSlot(hi: number, lo: number, capacity: number): number {
  const h = fmix32((lo ^ fmix32((hi ^ 0x9e3779b9) >>> 0)) >>> 0);
  return Math.floor((h * capacity) / TWO_POW_32);
}

function emptySlots(capacity: number): Uint32Array {
  return new Uint32Array(capacity * 2).fill(EMPTY);
}

/** Places a key known to be absent into `slots` (a table being built). */
function placeAbsent(slots: Uint32Array, capacity: number, hi: number, lo: number): void {
  let i = homeSlot(hi, lo, capacity);
  while (slots[2 * i] !== EMPTY) if (++i === capacity) i = 0;
  slots[2 * i] = hi;
  slots[2 * i + 1] = lo;
}

export class CompactIdSet implements Iterable<string> {
  private slots: Uint32Array;
  private capacity: number;
  private limit: number;
  private numeric = 0;
  private readonly fallback = new Set<string>();
  /** The table an in-flight reserveAsync is building (counted in tableBytes). */
  private building: Uint32Array | null = null;

  /** `expectedSize` sizes the table up front (e.g. a counted index). */
  constructor(expectedSize = 0) {
    this.capacity = idSetCapacityFor(expectedSize);
    this.slots = emptySlots(this.capacity);
    this.limit = Math.floor(this.capacity * ID_SET_MAX_LOAD);
  }

  get size(): number { return this.numeric + this.fallback.size; }
  /** Members stored in the typed-array table. */
  get numericSize(): number { return this.numeric; }
  /** Members held as values in the fallback Set (anything not a canonical decimal). */
  get fallbackSize(): number { return this.fallback.size; }
  get tableCapacity(): number { return this.capacity; }
  /** Exact bytes of the typed-array table(s) — the whole cost of the numeric members. */
  tableBytes(): number { return this.slots.byteLength + (this.building ? this.building.byteLength : 0); }

  has(id: string): boolean {
    if (typeof id === "string" && parseCanonical(id)) return this.slots[2 * this.find(keyHi, keyLo)] !== EMPTY;
    return this.fallback.has(id);
  }

  add(id: string): this {
    if (typeof id === "string" && parseCanonical(id)) this.addKey(keyHi, keyLo);
    else this.fallback.add(id);
    return this;
  }

  /** add(buf.toString("utf8", start, end)), without making a string for a
   *  canonical decimal (the boot load's 6.94M lines). */
  addBytes(buf: Buffer, start: number, end: number): this {
    if (parseCanonicalBytes(buf, start, end)) this.addKey(keyHi, keyLo);
    else this.fallback.add(buf.toString("utf8", start, end));
    return this;
  }

  values(): IterableIterator<string> { return this[Symbol.iterator](); }

  *[Symbol.iterator](): IterableIterator<string> {
    yield* this.fallback;
    const slots = this.slots;
    const cap = this.capacity;
    for (let i = 0; i < cap; i++) {
      const hi = slots[2 * i];
      if (hi !== EMPTY) yield decodeKey(hi, slots[2 * i + 1]);
    }
  }

  /** Makes room for `extra` more numeric members WITHOUT blocking: a needed
   *  resize builds the new table in slices (yielding to the loop whenever a
   *  slice has run `sliceMs`) while this one keeps answering, then swaps it in.
   *  If numeric members were added meanwhile, the copy is discarded and, if
   *  still needed, the resize is redone synchronously from the live table —
   *  correct before fast. */
  async reserveAsync(extra: number, sliceMs = 8): Promise<void> {
    if (this.numeric + extra <= this.limit) return;
    const capacity = idSetCapacityFor(this.numeric + extra);
    const old = this.slots;
    const oldCap = this.capacity;
    const startNumeric = this.numeric;
    const next = emptySlots(capacity);
    this.building = next;
    try {
      let sliceStart = performance.now();
      for (let i = 0; i < oldCap; i++) {
        const hi = old[2 * i];
        if (hi !== EMPTY) placeAbsent(next, capacity, hi, old[2 * i + 1]);
        if ((i & 4095) === 4095 && performance.now() - sliceStart >= sliceMs) {
          await new Promise<void>((resolve) => setImmediate(resolve));
          sliceStart = performance.now();
        }
      }
    } finally {
      this.building = null;
    }
    if (this.slots === old && this.numeric === startNumeric) {
      this.install(next, capacity);
    } else if (this.numeric + extra > this.limit) {
      this.rehash(idSetCapacityFor(this.numeric + extra));
    }
  }

  /** Longest probe distance in the table (O(capacity)): a hash-quality check for tests. */
  _maxProbeForTests(): number {
    let max = 0;
    for (let i = 0; i < this.capacity; i++) {
      const hi = this.slots[2 * i];
      if (hi === EMPTY) continue;
      const home = homeSlot(hi, this.slots[2 * i + 1], this.capacity);
      max = Math.max(max, i >= home ? i - home : i + this.capacity - home);
    }
    return max;
  }

  private find(hi: number, lo: number): number {
    const slots = this.slots;
    const cap = this.capacity;
    let i = homeSlot(hi, lo, cap);
    for (;;) {
      const h = slots[2 * i];
      if (h === EMPTY || (h === hi && slots[2 * i + 1] === lo)) return i;
      if (++i === cap) i = 0;
    }
  }

  private addKey(hi: number, lo: number): void {
    let i = this.find(hi, lo);
    if (this.slots[2 * i] !== EMPTY) return;
    if (this.numeric + 1 > this.limit) {
      this.rehash(idSetCapacityFor(this.numeric + 1));
      i = this.find(hi, lo);
    }
    this.slots[2 * i] = hi;
    this.slots[2 * i + 1] = lo;
    this.numeric++;
  }

  /** Synchronous resize (the fallback; routine growth goes through reserveAsync). */
  private rehash(capacity: number): void {
    const old = this.slots;
    const next = emptySlots(capacity);
    for (let i = 0; i < this.capacity; i++) {
      const hi = old[2 * i];
      if (hi !== EMPTY) placeAbsent(next, capacity, hi, old[2 * i + 1]);
    }
    this.install(next, capacity);
  }

  private install(slots: Uint32Array, capacity: number): void {
    this.slots = slots;
    this.capacity = capacity;
    this.limit = Math.floor(capacity * ID_SET_MAX_LOAD);
  }
}
