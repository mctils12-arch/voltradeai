// SWIM capture consumers (ops item 1, 2026-10-14): TFMS + NOTAM connect
// through swimConnector, every message is profiled by a content-free shape,
// raw samples are diag-token only, per-hour cadence for every product, and
// STDDS/ITWS stay configured-only.
import { test } from "node:test";
import assert from "node:assert/strict";
import express from "express";
import type { AddressInfo } from "node:net";
import {
  xmlShape, SwimCapture, startSwimCapture, registerSwimCaptureRoutes, swimCapture, _resetSwimCaptureForTests,
  CAPTURE_MAX_SHAPES, CAPTURE_SAMPLE_MAX_BYTES,
} from "./swimCapture";
import { bumpHourly, SWIM_HOURLY_KEEP, _resetSwimConnectorForTests } from "./swimConnector";

const TFMS_FI = `<?xml version="1.0"?><ns5:tfmDataService xmlns:ns5="urn:x"><fiOutput><fiMessage msgType="ADVISORY" sourceFacility="KDCC"><advisory><advisoryText>GROUND STOP KLGA SECRET-TEXT</advisoryText></advisory></fiMessage></fiOutput></ns5:tfmDataService>`;
const NOTAM = `<message:AIXMBasicMessage xmlns:message="m" gml:id="ID_123"><message:hasMember><event:Event gml:id="E1"><event:timeSlice><event:EventTimeSlice gml:id="T1"><gml:validTime/></event:EventTimeSlice></event:timeSlice></event:Event></message:hasMember></message:AIXMBasicMessage>`;

test("xmlShape: tag local names + type attributes only — never text, ids or other attribute values", () => {
  assert.equal(xmlShape(TFMS_FI), "tfmDataService>fiOutput>fiMessage[msgType=ADVISORY]>advisory>advisoryText");
  const n = xmlShape(NOTAM);
  assert.equal(n, "AIXMBasicMessage>hasMember>Event>timeSlice>EventTimeSlice>validTime");
  assert.ok(!xmlShape(TFMS_FI).includes("KDCC") && !n.includes("ID_123"));
  assert.equal(xmlShape(`<a xsi:type="fx:FlightType"><b/></a>`), "a[type=FlightType]>b");
  assert.equal(xmlShape("not xml at all"), "(not xml)");
  assert.equal(xmlShape(""), "(empty)");
});

test("SwimCapture: counts per shape, keeps the 2 latest samples, truncates big ones, bounds distinct shapes", () => {
  const c = new SwimCapture("TFMS");
  c.observe(TFMS_FI, 1); c.observe(TFMS_FI, 2); c.observe(TFMS_FI, 3); c.observe(NOTAM, 4);
  const v = c.shapesView();
  assert.equal(v.total, 4);
  assert.equal(v.shapes[0].count, 3);
  assert.equal(v.shapes[0].lastAt, 3);
  assert.ok(!JSON.stringify(v).includes("SECRET-TEXT"), "the public view is content-free");
  const s = c.samplesView(v.shapes[0].sig);
  assert.deepEqual(s.map((x) => x.at), [2, 3], "2 latest");
  const big = `<big>${"x".repeat(CAPTURE_SAMPLE_MAX_BYTES + 10)}</big>`;
  c.observe(big, 5);
  const b = c.samplesView("big")[0];
  assert.equal(b.truncated, true);
  assert.equal(b.payload.length, CAPTURE_SAMPLE_MAX_BYTES);
  const many = new SwimCapture("NOTAM");
  for (let i = 0; i < CAPTURE_MAX_SHAPES + 5; i++) many.observe(`<t${i}/>`, i);
  assert.equal(many.shapesView().distinct, CAPTURE_MAX_SHAPES);
  assert.equal(many.overflow, 5, "never silent past the cap");
});

test("bumpHourly: one bucket per UTC hour, last SWIM_HOURLY_KEEP kept", () => {
  const h: Array<{ hour: string; n: number }> = [];
  const t0 = Date.parse("2026-10-14T21:59:59Z");
  bumpHourly(h, t0); bumpHourly(h, t0 + 500); bumpHourly(h, t0 + 2000);
  assert.deepEqual(h, [{ hour: "2026-10-14T21", n: 2 }, { hour: "2026-10-14T22", n: 1 }]);
  for (let i = 0; i < 40; i++) bumpHourly(h, t0 + 3600_000 * (i + 2));
  assert.equal(h.length, SWIM_HOURLY_KEEP);
});

const ENV = {
  SWIM_USER: "someone@example.com", SWIM_PASSWORD: "hunter2-secret",
  SWIM_TFMS_URL: "tcps://ems1.swim.faa.gov:55443", SWIM_TFMS_VPN: "TFMS", SWIM_TFMS_QUEUE: "x.TFMS.OUT",
  SWIM_NOTAM_URL: "tcps://ems2.swim.faa.gov:55443", SWIM_NOTAM_VPN: "AIM_FNS", SWIM_NOTAM_QUEUE: "x.FNS.OUT",
  SWIM_STDDS_URL: "tcps://ems1", SWIM_STDDS_VPN: "STDDS", SWIM_STDDS_QUEUE: "x.STDDS.OUT",
  DIAG_TOKEN: "t".repeat(32),
} as unknown as NodeJS.ProcessEnv;

/** minimal solclientjs fake: every session connects, every consumer binds */
function fakeSolace() {
  const consumers: Record<string, (m?: unknown) => void> = {};
  const mod = {
    SolclientFactory: {
      init: () => {},
      createSession: (args: { vpnName: string }) => {
        const h: Record<string, (x?: unknown) => void> = {};
        return {
          on: (ev: string, fn: (x?: unknown) => void) => { h[ev] = fn; },
          connect: () => h.UP(),
          dispose: () => {},
          createMessageConsumer: () => {
            const ch: Record<string, (x?: unknown) => void> = {};
            return {
              on: (ev: string, fn: (x?: unknown) => void) => { ch[ev] = fn; if (ev === "MSG") consumers[args.vpnName] = fn; },
              connect: () => ch.CUP(),
            };
          },
        };
      },
    },
    SessionEventCode: { UP_NOTICE: "UP", CONNECT_FAILED_ERROR: "FAIL", DISCONNECTED: "DISC" },
    MessageConsumerEventName: { UP: "CUP", CONNECT_FAILED_ERROR: "CFAIL", DOWN: "CDOWN", MESSAGE: "MSG" },
    QueueType: { QUEUE: "QUEUE" }, MessageConsumerAcknowledgeMode: { CLIENT: "CLIENT" },
  };
  return { mod, consumers };
}
const msg = (xml: string) => ({ getXmlContentDecoded: () => xml, acknowledge: () => {} });

test("capture consumers: TFMS + NOTAM connect and profile; STDDS is never consumed; /products is public and secret-free; /samples is token-gated", async () => {
  _resetSwimConnectorForTests();
  _resetSwimCaptureForTests();
  const f = fakeSolace();
  const handles = await startSwimCapture({ env: ENV, importer: async () => f.mod, log: () => {} });
  assert.equal(handles.length, 2);
  assert.deepEqual(Object.keys(f.consumers).sort(), ["AIM_FNS", "TFMS"], "STDDS configured but never consumed");
  f.consumers.TFMS(msg(TFMS_FI));
  f.consumers.AIM_FNS(msg(NOTAM));
  await new Promise((r) => setImmediate(r)); // the connector drains on the next turn
  assert.equal(swimCapture("TFMS")!.total, 1);
  assert.equal(swimCapture("NOTAM")!.total, 1);

  const app = express();
  registerSwimCaptureRoutes(app, ENV);
  const srv = app.listen(0);
  try {
    const base = `http://127.0.0.1:${(srv.address() as AddressInfo).port}`;
    const pr = await (await fetch(`${base}/api/data/swim/products`)).json() as { products: Record<string, any> };
    assert.equal(pr.products.TFMS.connected, true);
    assert.equal(pr.products.TFMS.hourly.length, 1);
    assert.equal(pr.products.TFMS.shapes.shapes[0].sig, "tfmDataService>fiOutput>fiMessage[msgType=ADVISORY]>advisory>advisoryText");
    assert.equal(pr.products.STDDS.consumer, "configured-only");
    assert.equal(pr.products.STDDS.env.configured, true);
    assert.equal(pr.products.SFDPS.consumer, "flight plans (swimSfdps.ts)");
    const body = JSON.stringify(pr);
    assert.ok(!body.includes("hunter2") && !body.includes("someone@") && !body.includes("SECRET-TEXT"));
    assert.equal((await fetch(`${base}/api/data/swim/samples?product=TFMS`)).status, 404);
    assert.equal((await fetch(`${base}/api/data/swim/samples?product=TFMS&token=wrong`)).status, 404);
    const ok = await (await fetch(`${base}/api/data/swim/samples?product=TFMS&token=${"t".repeat(32)}`)).json() as { samples: Array<{ payload: string }> };
    assert.equal(ok.samples.length, 1);
    assert.ok(ok.samples[0].payload.includes("GROUND STOP"));
    assert.equal((await fetch(`${base}/api/data/swim/samples?product=STDDS&token=${"t".repeat(32)}`)).status, 400);
  } finally {
    srv.close();
    for (const h of handles) h.stop();
  }
});
