// SWIM TFMS parser — fixtures are trimmed copies of REAL messages captured
// from our queue on 2026-10-14 (capture PR #1342), element names and
// nesting unchanged: RSTR restriction, GDP cancel, GADV advisory, FXA flow
// area, APTC airport config, TMI_UPDATE reroute amendment, TMI_FLIGHT_LIST
// participation and an HCS flightPlanAmendmentInformation batch.
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseTfmsPayload, TfmsStore, TFMS_AMENDMENT_TTL_MS } from "./swimTfms";

const NOW = Date.parse("2026-10-14T23:30:00Z");
const fi = (msgType: string, body: string, src = "TFMS", ts = "2026-10-14T23:21:42Z") =>
  `<?xml version="1.0"?><ds:tfmDataService xmlns:ds="urn:a" xmlns:fi="urn:b" xmlns:fcm="urn:c" xmlns:fce="urn:d"><ds:fiOutput><fi:fiMessage sourceFacility="${src}" sourceTimeStamp="${ts}" msgType="${msgType}">${body}</fi:fiMessage></ds:fiOutput></ds:tfmDataService>`;

const RSTR = fi("RSTR", `<fi:restrictionMessage><fcm:eventTime>2026-10-14T19:20:00Z</fcm:eventTime><fcm:facility>N90</fcm:facility><fcm:action>4</fcm:action><fcm:restrictionId>17099454</fcm:restrictionId><fcm:restrictedNasElements>COATE</fcm:restrictedNasElements><fcm:startTime>2026-10-14T20:45:00Z</fcm:startTime><fcm:stopTime>2026-10-15T01:45:00Z</fcm:stopTime><fcm:airports>ORD</fcm:airports><fcm:restrictionType>DEPARTURE</fcm:restrictionType><fcm:restrictionCategory>APREQ</fcm:restrictionCategory><fcm:reasonText>VOL:Compacted Demand</fcm:reasonText></fi:restrictionMessage>`, "N90");
const GDP_CNX = fi("GDP", `<fi:gdpCancel><fcm:updateTime>2026-10-14T23:06:33Z</fcm:updateTime><fcm:tmiStatus>PURGED</fcm:tmiStatus><fcm:airportId>AUS</fcm:airportId><fcm:center>ZHU</fcm:center><fcm:pgmExpTime>2026-10-15T02:59:00Z</fcm:pgmExpTime><fcm:advisoryValidPeriod><fce:startTime>2026-10-14T18:12:00Z</fce:startTime><fce:endTime>2026-10-15T02:59:00Z</fce:endTime></fcm:advisoryValidPeriod></fi:gdpCancel>`);
const GADV = fi("GADV", `<fi:generalAdvisory><fcm:advisoryNumber>0092</fcm:advisoryNumber><fcm:origin>ATCSCC</fcm:origin><fcm:dateSent>2026-10-14T23:10:00Z</fcm:dateSent><fcm:effectivePeriod><fce:startTime>2026-10-14T23:10:00Z</fce:startTime><fce:endTime>2026-10-15T01:00:00Z</fce:endTime></fcm:effectivePeriod><fcm:advisoryTitle>ATCSCC ADVZY 092 LGA/ZNY 10/14/2026 CDM GROUND STOP</fcm:advisoryTitle><fcm:advisoryText>CTL ELEMENT: LGA
ELEMENT TYPE: APT
GROUND STOP PERIOD: 14/2310Z - 15/0100Z</fcm:advisoryText></fi:generalAdvisory>`, "DCCO");
const FXA = fi("FXA", `<fi:feaFca><fcm:fcaId>fca.dccops.lxstn28.20240721101238</fcm:fcaId><fcm:fcaName>JX4</fcm:fcaName><fcm:tmiStatus>UPDATED</fcm:tmiStatus><fcm:lastUpdate>2026-10-10T14:05:46Z</fcm:lastUpdate><fcm:fcaReason>NONE</fcm:fcaReason><fcm:startTime>2026-10-10T08:00:00Z</fcm:startTime><fcm:endTime>2026-10-17T23:59:00Z</fcm:endTime><fcm:line><fcm:ceiling>600</fcm:ceiling><fcm:floor>180</fcm:floor><fcm:points><fcm:point><fce:latitude>25.47</fce:latitude><fce:longitude>-87.27</fce:longitude></fcm:point><fcm:point><fce:latitude>28.52</fce:latitude><fce:longitude>-83.9</fce:longitude></fcm:point></fcm:points></fcm:line></fi:feaFca>`, "DCC");
const APTC = fi("APTC", `<fi:airportConfigMessage><fcm:eventTime>2026-10-14T16:50:00Z</fcm:eventTime><fcm:facility>M98</fcm:facility><fcm:airport>MSP</fcm:airport><fcm:arrRunwayConf>12L/12R</fcm:arrRunwayConf><fcm:depRunwayConf>12L/12R</fcm:depRunwayConf><fcm:arrRate>62</fcm:arrRate><fcm:depRate>66</fcm:depRate><fcm:updateTime>2026-10-14T16:50:00Z</fcm:updateTime><fcm:weather>LVMC</fcm:weather></fi:airportConfigMessage>`, "M98");
const TMI_UPDATE = fi("TMI_UPDATE", `<fi:eramAmendmentStatusUpdate><fcm:rrAmendment><fcm:flightIdfr>159831194</fcm:flightIdfr><fcm:acid>ENY3547</fcm:acid><fcm:gufi>KT780094Pv</fcm:gufi><fcm:controllingSector>ZID82</fcm:controllingSector><fcm:flightAmendmentStatus><fcm:tmiId>TRZID</fcm:tmiId><fcm:routeAmendment>KTYS./.HEVAN..BONNT.VEECK5.KORD</fcm:routeAmendment><fcm:amendmentStatus>SENT</fcm:amendmentStatus><fcm:amendmentTime>2026-10-14T23:21:42Z</fcm:amendmentTime></fcm:flightAmendmentStatus></fcm:rrAmendment></fi:eramAmendmentStatusUpdate>`);
const TMI_LIST = fi("TMI_FLIGHT_LIST", `<fi:tmiFlightDataList><fi:flightData><fcm:flight><fce:aircraftId>DAL1834</fce:aircraftId><fce:departurePoint><fce:airport>TPA</fce:airport></fce:departurePoint></fcm:flight><fcm:fxaFlightData><fcm:fxaFlight><fcm:fxaId><fce:fcaId>fca.zdc.lxstn21.20261003172620</fce:fcaId><fce:fcaName>-ZJXAREA3SPLIT</fce:fcaName></fcm:fxaId><fcm:entryTm>2026-10-15T00:10:01Z</fcm:entryTm></fcm:fxaFlight></fcm:fxaFlightData></fi:flightData></fi:tmiFlightDataList>`);
const FLTD = `<?xml version="1.0"?><ds:tfmDataService xmlns:ds="urn:a" xmlns:fdm="urn:e" xmlns:nxcm="urn:f" xmlns:nxce="urn:g"><fltdOutput><fdm:fltdMessage acid="SWA331" arrArpt="KSAN" depArpt="KSMF" fdTrigger="HCS_AMENDMENT_MSG" msgType="flightPlanAmendmentInformation" sourceFacility="KZLA" sourceTimeStamp="2026-10-14T23:21:44Z"><fdm:flightPlanAmendmentInformation><nxcm:qualifiedAircraftId><nxce:aircraftId>SWA331</nxce:aircraftId><nxce:gufi>KO736594Gk</nxce:gufi></nxcm:qualifiedAircraftId><nxcm:amendmentData><nxcm:newAltitude><nxce:assignedAltitude><nxce:simpleAltitude>240</nxce:simpleAltitude></nxce:assignedAltitude></nxcm:newAltitude><nxcm:newRouteOfFlight legacyFormat="KSMF./.MOD037038..HUULK.COMIX2.KSAN/2345"/></nxcm:amendmentData></fdm:flightPlanAmendmentInformation></fdm:fltdMessage><fdm:fltdMessage acid="N8325C" msgType="trackInformation" sourceTimeStamp="2026-10-14T23:21:44Z"><fdm:trackInformation/></fdm:fltdMessage></fltdOutput></ds:tfmDataService>`;
const TRACK_ONLY = `<ds:tfmDataService xmlns:ds="urn:a"><fltdOutput><fdm:fltdMessage xmlns:fdm="urn:e" acid="N1" msgType="trackInformation"/></fltdOutput></ds:tfmDataService>`;

test("RSTR / GADV / FXA / GDP: programs with windows, kinds and elements as broadcast", () => {
  const r = parseTfmsPayload(RSTR, NOW).programs[0];
  assert.equal(r.kind, "RESTRICTION");
  assert.equal(r.id, "RSTR:17099454");
  assert.equal(r.airport, "ORD");
  assert.equal(r.title, "APREQ DEPARTURE");
  assert.equal(r.reason, "VOL:Compacted Demand");
  assert.deepEqual(r.elements, ["COATE"]);
  assert.equal(r.end, Date.parse("2026-10-15T01:45:00Z"));
  const a = parseTfmsPayload(GADV, NOW).programs[0];
  assert.equal(a.kind, "ADVISORY");
  assert.equal(a.status, "GS", "the advisory title names a ground stop");
  assert.equal(a.airport, "LGA");
  const f = parseTfmsPayload(FXA, NOW).programs[0];
  assert.equal(f.kind, "FCA");
  assert.deepEqual(f.geometry, [[25.47, -87.27], [28.52, -83.9]]);
  assert.equal(f.floorFl, 180); assert.equal(f.ceilingFl, 600);
  const g = parseTfmsPayload(GDP_CNX, NOW).programs[0];
  assert.equal(g.id, "GDP:AUS");
  assert.equal(g.status, "PURGED");
});

test("APTC: live runway configuration + arrival/departure rates per airport", () => {
  const c = parseTfmsPayload(APTC, NOW).configs[0];
  assert.deepEqual({ ...c }, {
    airport: "MSP", arrRunways: "12L/12R", depRunways: "12L/12R", aar: 62, adr: 66, weather: "LVMC",
    facility: "M98", at: Date.parse("2026-10-14T16:50:00Z"),
  });
});

test("per flight: TMI reroute amendment (program id + route), HCS amendment (route + assigned FL), TMI participation", () => {
  const rr = parseTfmsPayload(TMI_UPDATE, NOW).amendments[0];
  assert.equal(rr.acid, "ENY3547");
  assert.equal(rr.kind, "TMI_REROUTE");
  assert.equal(rr.tmiId, "TRZID");
  assert.equal(rr.route, "KTYS./.HEVAN..BONNT.VEECK5.KORD");
  const p = parseTfmsPayload(FLTD, NOW);
  assert.equal(p.amendments.length, 1, "the batched trackInformation is counted, not parsed");
  assert.deepEqual(p.types, ["flightPlanAmendmentInformation", "trackInformation"]);
  const h = p.amendments[0];
  assert.equal(h.kind, "ERAM_AMENDMENT");
  assert.equal(h.route, "KSMF./.MOD037038..HUULK.COMIX2.KSAN/2345");
  assert.equal(h.assignedFl, 240);
  assert.equal(h.dep, "KSMF"); assert.equal(h.arr, "KSAN");
  const t = parseTfmsPayload(TMI_LIST, NOW).tmis[0];
  assert.equal(t.acid, "DAL1834");
  assert.equal(t.fcaName, "-ZJXAREA3SPLIT");
});

test("payloads with nothing worth parsing are counted only (no XML parse)", () => {
  const p = parseTfmsPayload(TRACK_ONLY, NOW);
  assert.deepEqual(p.types, ["trackInformation"]);
  assert.equal(p.amendments.length + p.programs.length + p.configs.length, 0);
  assert.deepEqual(parseTfmsPayload("", NOW).types, []);
});

test("store: active programs exclude purged/expired ones; per-flight lookup by callsign; airport config by FAA or ICAO id; TTL", () => {
  const s = new TfmsStore();
  for (const x of [RSTR, GDP_CNX, GADV, FXA, APTC, TMI_UPDATE, TMI_LIST, FLTD, TRACK_ONLY]) s.ingest(x, NOW);
  const act = s.active(NOW);
  const ids = act.programs.map((p) => p.id);
  assert.ok(ids.includes("RSTR:17099454") && ids.includes("FCA:fca.dccops.lxstn28.20240721101238"));
  assert.ok(!ids.includes("GDP:AUS"), "a purged GDP is not active");
  assert.ok(ids.some((i) => i.startsWith("ADV:ATCSCC:0092")));
  assert.equal(s.airportConfig("KMSP")!.aar, 62);
  assert.equal(s.airportConfig("MSP")!.depRunways, "12L/12R");
  assert.equal(s.forFlight("swa331", NOW).amendments[0].assignedFl, 240);
  assert.equal(s.forFlight("ENY3547", NOW).amendments[0].tmiId, "TRZID");
  assert.equal(s.forFlight("DAL1834", NOW).tmis.length, 1);
  assert.deepEqual(s.forFlight("SWA331", NOW + TFMS_AMENDMENT_TTL_MS + 1), { amendments: [], tmis: [] });
  assert.ok(!s.active(Date.parse("2026-10-15T02:00:00Z")).programs.some((p) => p.id === "RSTR:17099454"), "ended");
  const st = s.stats();
  assert.equal(st.typeCounts.trackInformation, 2);
  assert.equal(st.parseErrors, 0);
});
