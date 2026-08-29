// GATE 1 (DATA) reference-list parser test — research/open_questions.md
// "SHADOW-FLEET SIGNAL" validation plan. Fixture mirrors the real,
// live-fetched OFAC SDN.XML shape (probed 2026-08-29): a Vessel-type
// sdnEntry with idList entries carrying MMSI/IMO/Former Vessel Flag, plus
// a vesselInfo block for type/flag, alongside non-vessel entries (Entity/
// Individual) and vessel entries with no MMSI — both of which the plan's
// join key (MMSI, against our own AIS archive) makes unusable and this
// parser must exclude, not guess.
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseSdnVesselEntries, OFAC_SDN_XML_URL, fetchOfacSdnXml } from "./shadowFleetReference";

const XML = `<?xml version="1.0" standalone="yes"?>
<sdnList>
  <publshInformation><Publish_Date>08/28/2026</Publish_Date><Record_Count>3</Record_Count></publshInformation>
  <sdnEntry>
    <uid>15036</uid>
    <lastName>ARTAVIL</lastName>
    <sdnType>Vessel</sdnType>
    <remarks>(Linked To: NATIONAL IRANIAN TANKER COMPANY)</remarks>
    <programList><program>IRAN</program></programList>
    <idList>
      <id><uid>8045</uid><idType>Vessel Registration Identification</idType><idNumber>IMO 9187629</idNumber></id>
      <id><uid>8114</uid><idType>MMSI</idType><idNumber>572469210</idNumber></id>
      <id><uid>111445</uid><idType>Former Vessel Flag</idType><idNumber>Malta</idNumber></id>
    </idList>
    <vesselInfo>
      <callSign>T2EU4</callSign>
      <vesselType>Crude/Oil Products Tanker</vesselType>
      <vesselFlag>Iran</vesselFlag>
    </vesselInfo>
  </sdnEntry>
  <sdnEntry>
    <uid>36</uid>
    <lastName>AEROCARIBBEAN AIRLINES</lastName>
    <sdnType>Entity</sdnType>
    <programList><program>CUBA</program></programList>
  </sdnEntry>
  <sdnEntry>
    <uid>4238</uid>
    <lastName>MAR AZUL</lastName>
    <sdnType>Vessel</sdnType>
    <programList><program>CUBA</program></programList>
    <vesselInfo>
      <callSign>CL2192</callSign>
      <vesselType>Tug</vesselType>
      <vesselFlag>Cuba</vesselFlag>
    </vesselInfo>
  </sdnEntry>
</sdnList>`;

test("parses a Vessel entry with an MMSI id into a joinable reference row", () => {
  const rows = parseSdnVesselEntries(XML);
  assert.equal(rows.length, 1, "only the MMSI-bearing vessel entry is kept");
  const r = rows[0];
  assert.equal(r.uid, "15036");
  assert.equal(r.name, "ARTAVIL");
  assert.equal(r.mmsi, "572469210");
  assert.equal(r.imo, "9187629", "IMO digits extracted from the 'IMO 9187629' idNumber, prefix stripped");
  assert.equal(r.vesselType, "Crude/Oil Products Tanker");
  assert.equal(r.vesselFlag, "Iran");
  assert.deepEqual(r.programs, ["IRAN"]);
});

test("non-Vessel sdnType entries (Entity/Individual) are excluded", () => {
  const rows = parseSdnVesselEntries(XML);
  assert.ok(!rows.some((r) => r.name === "AEROCARIBBEAN AIRLINES"));
});

test("Vessel entries with no MMSI idList entry are excluded, not kept with a null field", () => {
  const rows = parseSdnVesselEntries(XML);
  assert.ok(!rows.some((r) => r.name === "MAR AZUL"), "MAR AZUL has vesselInfo but no MMSI id");
});

test("Former Vessel Flag id entries are never misread as MMSI", () => {
  const rows = parseSdnVesselEntries(XML);
  assert.equal(rows[0].mmsi, "572469210", "not 'Malta' or any Former Vessel Flag value");
});

test("empty document parses to an empty list, not an error", () => {
  assert.deepEqual(parseSdnVesselEntries("<sdnList></sdnList>"), []);
});

test("a malformed MMSI idNumber (non-digit or wrong length) is not accepted", () => {
  const bad = XML.replace("<idNumber>572469210</idNumber>", "<idNumber>N/A</idNumber>");
  const rows = parseSdnVesselEntries(bad);
  assert.ok(!rows.some((r) => r.uid === "15036"), "ARTAVIL has no other MMSI id, so it drops out entirely");
});

test("fetchOfacSdnXml sends the documented URL and User-Agent, and rejects on a non-2xx status", async () => {
  let calledUrl = "";
  let calledHeaders: Record<string, string> | undefined;
  const ok = async (url: string, init?: RequestInit) => {
    calledUrl = url;
    calledHeaders = init?.headers as Record<string, string> | undefined;
    return { ok: true, status: 200, text: async () => XML };
  };
  const xml = await fetchOfacSdnXml(ok);
  assert.equal(calledUrl, OFAC_SDN_XML_URL);
  assert.ok(calledHeaders?.["User-Agent"]?.includes("voltradeai-datacore"));
  assert.equal(xml, XML);

  const fail = async () => ({ ok: false, status: 503, text: async () => "" });
  await assert.rejects(() => fetchOfacSdnXml(fail), /http 503/);
});
