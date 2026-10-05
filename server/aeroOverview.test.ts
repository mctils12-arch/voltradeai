// aeroOverview.test.ts — the raster helpers behind zoomed-out FAA chart tiles
// (overview pyramid) and the parent fill for missing deeper tiles.

import { test } from "node:test";
import assert from "node:assert/strict";
import sharp from "sharp";

import { TILE_PX, childTiles, composeOverview, upscaleQuadrant } from "./aeroOverview";

async function solidJpeg(r: number, g: number, b: number): Promise<Buffer> {
  return sharp({ create: { width: TILE_PX, height: TILE_PX, channels: 3, background: { r, g, b } } }).jpeg({ quality: 95 }).toBuffer();
}
async function solidPng(r: number, g: number, b: number, alpha: number): Promise<Buffer> {
  return sharp({ create: { width: TILE_PX, height: TILE_PX, channels: 4, background: { r, g, b, alpha } } }).png().toBuffer();
}
async function pixel(img: Buffer, x: number, y: number): Promise<number[]> {
  const { data, info } = await sharp(img).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const i = (y * info.width + x) * 4;
  return Array.from(data.slice(i, i + 4));
}
const near = (a: number[], b: number[], tol = 4) => a.every((v, i) => Math.abs(v - b[i]) <= tol);

test("childTiles follows the quadtree NW, NE, SW, SE", () => {
  assert.deepEqual(childTiles(7, 30, 50), [
    { z: 8, x: 60, y: 100 }, { z: 8, x: 61, y: 100 }, { z: 8, x: 60, y: 101 }, { z: 8, x: 61, y: 101 },
  ]);
});

test("composeOverview puts each child, halved, in its own quadrant; all-opaque -> JPEG", async () => {
  const out = await composeOverview([await solidJpeg(255, 0, 0), await solidJpeg(0, 255, 0), await solidJpeg(0, 0, 255), await solidJpeg(10, 20, 30)]);
  assert.ok(out);
  assert.equal(out!.contentType, "image/jpeg");
  const meta = await sharp(out!.body).metadata();
  assert.equal(meta.width, TILE_PX); assert.equal(meta.height, TILE_PX);
  assert.ok(near(await pixel(out!.body, 40, 40), [255, 0, 0, 255], 8));
  assert.ok(near(await pixel(out!.body, 200, 40), [0, 255, 0, 255], 8));
  assert.ok(near(await pixel(out!.body, 40, 200), [0, 0, 255, 255], 8));
  assert.ok(near(await pixel(out!.body, 200, 200), [10, 20, 30, 255], 8));
});

test("a missing child leaves its quadrant transparent and the tile stays PNG", async () => {
  const out = await composeOverview([await solidJpeg(255, 0, 0), null, null, null]);
  assert.ok(out);
  assert.equal(out!.contentType, "image/png", "transparency must survive encoding");
  assert.equal((await pixel(out!.body, 200, 200))[3], 0);
  assert.ok(near(await pixel(out!.body, 40, 40), [255, 0, 0, 255]));
});

test("no visible child pixels -> null (cached as an empty marker, not an image)", async () => {
  assert.equal(await composeOverview([null, null, null, null]), null);
  assert.equal(await composeOverview([await solidPng(9, 9, 9, 0), null, Buffer.alloc(0), null]), null);
});

test("an undecodable child is treated as absent, never fatal", async () => {
  const out = await composeOverview([Buffer.from("not an image"), await solidJpeg(1, 200, 1), null, null]);
  assert.ok(out);
  assert.equal((await pixel(out!.body, 40, 40))[3], 0);
  assert.ok(near(await pixel(out!.body, 200, 40), [1, 200, 1, 255], 8));
});

test("transparent edge pixels never darken the chart (premultiplied resize)", async () => {
  // left half opaque orange, right half fully transparent black
  const raw = Buffer.alloc(TILE_PX * TILE_PX * 4);
  for (let y = 0; y < TILE_PX; y++) for (let x = 0; x < TILE_PX / 2; x++) {
    const i = (y * TILE_PX + x) * 4; raw[i] = 200; raw[i + 1] = 100; raw[i + 2] = 50; raw[i + 3] = 255;
  }
  const child = await sharp(raw, { raw: { width: TILE_PX, height: TILE_PX, channels: 4 } }).png().toBuffer();
  const out = await composeOverview([child, null, null, null]);
  const [r, g, b, a] = await pixel(out!.body, 63, 20); // the boundary column of the shrunk child
  assert.ok(a > 0);
  assert.ok(near([r, g, b], [200, 100, 50], 6), `colour went dark at the edge: ${r},${g},${b}`);
});

test("composeOverview rejects a wrong child count", async () => {
  await assert.rejects(composeOverview([null]), /exactly 4/);
});

test("upscaleQuadrant enlarges the right quarter of the parent", async () => {
  const raw = Buffer.alloc(TILE_PX * TILE_PX * 4);
  for (let y = 0; y < TILE_PX; y++) for (let x = 0; x < TILE_PX; x++) {
    const i = (y * TILE_PX + x) * 4; raw[i + 3] = 255;
    if (y < 128 && x >= 128) raw[i] = 255; // NE quadrant red
  }
  const parent = await sharp(raw, { raw: { width: TILE_PX, height: TILE_PX, channels: 4 } }).png().toBuffer();
  const ne = await upscaleQuadrant(parent, 1, 0);
  assert.ok(ne);
  assert.equal(ne!.contentType, "image/jpeg");
  assert.ok(near(await pixel(ne!.body, 128, 128), [255, 0, 0, 255], 8));
  const sw = await upscaleQuadrant(parent, 0, 1);
  assert.ok(near(await pixel(sw!.body, 128, 128), [0, 0, 0, 255], 8));
});

test("upscaleQuadrant: empty quarter -> null, bad bytes -> null", async () => {
  assert.equal(await upscaleQuadrant(await solidPng(5, 5, 5, 0), 0, 0), null);
  assert.equal(await upscaleQuadrant(Buffer.from("nope"), 0, 0), null);
});

test("overview work runs off the event loop: the loop keeps ticking during a burst", async () => {
  const kids = [await solidJpeg(1, 2, 3), await solidJpeg(4, 5, 6), await solidJpeg(7, 8, 9), await solidJpeg(10, 11, 12)];
  let maxGap = 0, last = performance.now();
  const iv = setInterval(() => { const t = performance.now(); maxGap = Math.max(maxGap, t - last); last = t; }, 5);
  const t0 = performance.now();
  await Promise.all(Array.from({ length: 8 }, () => composeOverview(kids)));
  const per = (performance.now() - t0) / 8;
  clearInterval(iv);
  assert.ok(per < 250, `overview tile took ${per.toFixed(1)} ms`);
  assert.ok(maxGap < 150, `event loop stalled ${maxGap.toFixed(0)} ms during overview work`);
});
