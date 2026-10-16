"""Tests for sec_filings_text.py (Lazy Prices PR 1 — EDGAR 10-K/10-Q archive
+ section extractor). All offline: HTTP, clock and sleep are injected."""
import gzip
import json
import os
import socket

import pytest

import sec_filings_text as m

REPO = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────

class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.sleeps.append(dt)
        self.t += dt


def free_limiter():
    clk = FakeClock()
    return m.RateLimiter(rate=10, clock=clk, sleep=clk.sleep)


class FakeHTTP:
    """url -> list of responses (consumed in order; last one repeats).
    A response is an HttpResponse, an Exception instance, or bytes/dict
    (shorthand for 200)."""

    def __init__(self, routes):
        self.routes = {k: (v if isinstance(v, list) else [v]) for k, v in routes.items()}
        self.calls = []

    def __call__(self, url, headers, timeout):
        self.calls.append((url, dict(headers)))
        if url not in self.routes:
            return m.HttpResponse(404, {}, b"not found")
        seq = self.routes[url]
        r = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(r, Exception):
            raise r
        if isinstance(r, dict) or isinstance(r, list):
            return m.HttpResponse(200, {}, json.dumps(r).encode())
        if isinstance(r, (bytes, str)):
            return m.HttpResponse(200, {}, r.encode() if isinstance(r, str) else r)
        return r


def client_for(routes, **kw):
    http = FakeHTTP(routes)
    sleeps = []
    c = m.EdgarClient(http_get=http, limiter=free_limiter(), sleep=sleeps.append, **kw)
    return c, http, sleeps


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic filings
# ─────────────────────────────────────────────────────────────────────────────

def footer(n):
    # running footer (B3), bare page number (B1), ToC back-link (B2)
    return (f"<p>Acme Corp | 2025 Form 10-K | {n}</p><p>{n}</p>"
            f"<p><a href='#toc'>Table of Contents</a></p><hr/>")


TEN_K = f"""<html><head><title>acme-20251231.htm</title>
<style>.c{{color:red}}</style><script>var leak = "Item 1A. Risk Factors SCRIPT_LEAK";</script></head>
<body>
<div style="display:none"><ix:header><ix:hidden>
<ix:nonNumeric name="dei:DocumentType">Item 7. Management's Discussion HIDDEN_FACT</ix:nonNumeric>
</ix:hidden><ix:resources><xbrli:context id="c1">CTX_LEAK</xbrli:context></ix:resources></ix:header></div>
<p>ACME CORP</p><p>ANNUAL REPORT ON FORM 10-K</p>
<p>TABLE OF CONTENTS</p>
<table>
<tr><td>PART I</td><td></td><td></td></tr>
<tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>5</td></tr>
<tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>9</td></tr>
<tr><td>Item 2.</td><td>Properties</td><td>9</td></tr>
<tr><td>Item 3.</td><td>Legal Proceedings</td><td>10</td></tr>
<tr><td>Item 4.</td><td>Mine Safety Disclosures</td><td>10</td></tr>
<tr><td>PART II</td><td></td><td></td></tr>
<tr><td>Item 7.</td><td>Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</td><td>12</td></tr>
<tr><td>Item 7A.</td><td>Quantitative and Qualitative Disclosures About Market Risk</td><td>20</td></tr>
<tr><td>Item 8.</td><td>Financial Statements and Supplementary Data</td><td>21</td></tr>
</table>
{footer(1)}
<p>PART I</p>
<p>Item 1. Business</p>
<p>Acme makes anvils and rocket skates for the desert market.</p>
{footer(3)}
<p><b>Item 1A.</b></p><p><b>Risk Factors</b></p>
<p>RF_START Our business depends on a small number of roadrunner-related customers, and the loss of any one could materially harm revenue.</p>
<p>Item 2 of this report describes our properties, which are concentrated in one desert.</p>
{footer(5)}
<p>Anvil demand is cyclical and we may be unable to forecast it; rocket skate warranty claims could exceed reserves.</p>
{footer(6)}
<p>RF_END Our insurance may not cover falling-anvil incidents.</p>
<p>Item 1B. Unresolved Staff Comments</p>
<p>None.</p>
<p>Item 2. Properties</p>
<p>We own one factory in the desert.</p>
<table><tr><td>ITEM 3.</td><td>LEGAL PROCEEDINGS</td></tr></table>
<p>LP_START We are a defendant in litigation brought by Coyote LLC alleging defective products; we believe the claims lack merit and intend to defend vigorously. LP_END</p>
{footer(10)}
<p>Item 4. Mine Safety Disclosures</p>
<p>Not applicable.</p>
<p>PART II</p>
<p>ITEM 7. MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</p>
<p>MDNA_START Revenue grew on strong anvil demand, as summarized below.</p>
<table>
<tr><td>(in millions)</td><td>2025</td><td>2024</td></tr>
<tr><td>Revenue</td><td>1,234.5</td><td>1,100.2</td></tr>
<tr><td>Cost of sales</td><td>567.8</td><td>500.1</td></tr>
</table>
{footer(12)}
<p>MDNA_END We expect continued growth in the desert segment.</p>
<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p>
<p>Interest rate risk is not material.</p>
<p>Item 8. Financial Statements and Supplementary Data</p>
<p>See the consolidated financial statements.</p>
</body></html>"""


def ten_q(with_part2_heading=True):
    part2 = "<p>PART II &#8212; OTHER INFORMATION</p>" if with_part2_heading else ""
    return f"""<html><body>
<ix:header><ix:hidden>HIDDEN_Q</ix:hidden></ix:header>
<p>ACME CORP FORM 10-Q</p>
<table>
<tr><td>PART I. FINANCIAL INFORMATION</td><td></td></tr>
<tr><td>Item 1.</td><td>Financial Statements</td><td>3</td></tr>
<tr><td>Item 2.</td><td>Management's Discussion and Analysis of Financial Condition and Results of Operations</td><td>15</td></tr>
<tr><td>Item 3.</td><td>Quantitative and Qualitative Disclosures About Market Risk</td><td>22</td></tr>
<tr><td>Item 4.</td><td>Controls and Procedures</td><td>22</td></tr>
<tr><td>PART II. OTHER INFORMATION</td><td></td></tr>
<tr><td>Item 1.</td><td>Legal Proceedings</td><td>23</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>23</td></tr>
<tr><td>Item 2.</td><td>Unregistered Sales of Equity Securities and Use of Proceeds</td><td>24</td></tr>
<tr><td>Item 6.</td><td>Exhibits</td><td>25</td></tr>
</table>
<p>PART I &#8212; FINANCIAL INFORMATION</p>
<p>Item 1. Financial Statements</p>
<p>FS_TEXT Condensed consolidated balance sheets follow.</p>
<table><tr><td>Cash</td><td>100</td><td>90</td></tr><tr><td>Debt</td><td>50</td><td>55</td></tr></table>
<p>Item 2. Management&#8217;s Discussion and Analysis of Financial Condition and Results of Operations</p>
<p>Q_MDNA_START Quarterly revenue rose on anvil demand.</p>
<p>Q_MDNA_END Liquidity remains adequate.</p>
<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>
<p>There have been no material changes to our market risk.</p>
<p>Item 4. Controls and Procedures</p>
<p>Disclosure controls were effective.</p>
{part2}
<p>Item 1. Legal Proceedings</p>
<p>Q_LP See Note 9, Commitments and Contingencies, for a description of legal proceedings.</p>
<p>Item 1A. Risk Factors</p>
<p>There have been no material changes to the risk factors disclosed in Part I, Item 1A of our Annual Report on Form 10-K for the year ended December 31, 2024.</p>
<p>Item 1A of our Annual Report on Form 10-K remains the place to read them.</p>
<p>Item 2. Unregistered Sales of Equity Securities and Use of Proceeds</p>
<p>None.</p>
<p>Item 6. Exhibits</p>
<p>Exhibit index follows.</p>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# HTML -> text + boilerplate
# ─────────────────────────────────────────────────────────────────────────────

def test_html_to_text_strips_script_style_ixheader_hidden_and_numeric_tables():
    lines, counts = m.html_to_text(TEN_K)
    blob = "\n".join(lines)
    for leak in ("SCRIPT_LEAK", "HIDDEN_FACT", "CTX_LEAK", "color:red", "acme-20251231.htm"):
        assert leak not in blob
    assert "1,234.5" not in blob and "567.8" not in blob  # numeric table dropped
    assert counts["tables_dropped"] == 1
    assert counts["tables_kept"] == 2  # ToC table + heading layout table
    assert "ITEM 3. LEGAL PROCEEDINGS" in blob  # heading inside a layout table survives
    assert "Management’s Discussion" in blob  # entity decoded


def test_source_newlines_inside_html_text_are_whitespace_not_line_breaks():
    # Regression (found in the 50-company gate run, InspireMD 10-Q): filer
    # software hard-wraps HTML source mid-heading.
    html = "<p><b>Item\n1A. Risk Factors</b></p><p>Except\nas set forth below, no changes.</p><p>Item 2. Other</p>"
    lines, _ = m.html_to_text(html)
    assert lines == ["Item 1A. Risk Factors", "Except as set forth below, no changes.", "Item 2. Other"]
    rf = m.extract_sections(html, "10-K")["sections"]["risk_factors"]
    assert rf["found"] and rf["start_heading"] == "Item 1A. Risk Factors"


def test_html_to_text_accepts_bytes_and_latin1():
    lines, _ = m.html_to_text("<p>Caf\xe9 Item 1A</p>".encode("latin-1"))
    assert lines == ["Café Item 1A"]


def test_sgml_text_keeps_only_first_document_drops_exhibits():
    raw = ("<SEC-DOCUMENT><DOCUMENT><TYPE>10-K<TEXT>Item 1A. Risk Factors\nReal risk text here.\n"
           "Item 1B. Unresolved Staff Comments\nNone.</TEXT></DOCUMENT>"
           "<DOCUMENT><TYPE>EX-21<TEXT>EXHIBIT_LEAK Item 7. subsidiaries</TEXT></DOCUMENT>")
    lines, counts = m.html_to_text(raw)
    assert counts["is_html"] is False
    assert "EXHIBIT_LEAK" not in "\n".join(lines)
    assert "Real risk text here." in lines


def test_boilerplate_rules_each_counted():
    lines = (["Acme Corp | 2025 Form 10-K | %d" % i for i in range(1, 6)]
             + ["12", "- 13 -", "Page 14", "F-3", "ii", "Table of Contents", "Back to Contents",
                "Index", "PART I Item 1A"] * 1 + ["PART I Item 1A"] * 4
             + ["Item 2. Properties", "Item 3. Legal Proceedings", "Item 4. Mine Safety",
                "Item 5. Market", "Item 6. Reserved", "Real sentence that stays."])
    out, c = m.strip_boilerplate(lines)
    assert c["repeated_header_lines"] == 10  # 5 footers + 5 running "PART I Item 1A"
    assert c["page_number_lines"] == 5
    assert c["toc_link_lines"] == 3
    kept, c2 = m.strip_boilerplate(["Civil", "xiv", "Mix", "iv", "Vi"])
    assert kept == ["Civil", "Mix"] and c2["page_number_lines"] == 3
    # Distinct Item headings must NOT be merged by digit-normalization.
    assert out == ["Item 2. Properties", "Item 3. Legal Proceedings", "Item 4. Mine Safety",
                   "Item 5. Market", "Item 6. Reserved", "Real sentence that stays."]


# ─────────────────────────────────────────────────────────────────────────────
# Section boundaries
# ─────────────────────────────────────────────────────────────────────────────

def test_10k_risk_factors_boundaries_and_toc_defeat():
    ext = m.extract_sections(TEN_K, "10-K")
    rf = ext["sections"]["risk_factors"]
    assert rf["found"] and rf["end_found"]
    assert rf["text"].startswith("RF_START")
    assert rf["text"].endswith("RF_END Our insurance may not cover falling-anvil incidents.")
    assert rf["start_heading"] == "Item 1A. Risk Factors"  # split heading joined
    assert rf["end_heading"].startswith("Item 1B.")
    assert rf["candidates"] == 2 and rf["toc_duplicate_defeated"] is True
    # prose cross-reference "Item 2 of this report" did NOT end the section
    assert "Item 2 of this report describes" in rf["text"]
    # boilerplate gone from the section
    assert "Acme Corp | 2025" not in rf["text"] and "Table of Contents" not in rf["text"]
    assert not any(ln.strip().isdigit() for ln in rf["text"].split("\n"))
    assert rf["char_count"] == len(rf["text"])
    assert rf["flags"] == {"no_material_changes": False, "not_applicable": False,
                           "incorporated_by_reference": False}


def test_10k_legal_proceedings_heading_in_table():
    lp = m.extract_sections(TEN_K, "10-K")["sections"]["legal_proceedings"]
    assert lp["found"]
    assert lp["text"].startswith("LP_START") and lp["text"].endswith("LP_END")
    assert lp["start_heading"] == "ITEM 3. LEGAL PROCEEDINGS"
    assert lp["end_heading"].startswith("Item 4.")


def test_10k_mdna_ends_at_7a_and_drops_tables():
    md = m.extract_sections(TEN_K, "10-K")["sections"]["mdna"]
    assert md["found"]
    assert md["text"].startswith("MDNA_START")
    assert md["text"].endswith("MDNA_END We expect continued growth in the desert segment.")
    assert md["end_heading"].startswith("Item 7A.")
    assert "1,234.5" not in md["text"] and "HIDDEN_FACT" not in md["text"]


def test_10k_mdna_ends_at_item_8_when_no_7a():
    html = TEN_K.replace("<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p>\n"
                         "<p>Interest rate risk is not material.</p>\n", "")
    md = m.extract_sections(html, "10-K")["sections"]["mdna"]
    assert md["end_heading"].startswith("Item 8.")
    assert md["text"].endswith("desert segment.")


def test_10k_extraction_metadata():
    ext = m.extract_sections(TEN_K, "10-K")
    e = ext["extraction"]
    assert e["version"] == m.EXTRACTION_VERSION
    assert "risk_factors_toc_duplicate_defeated" in e["flags"]
    assert not any(f.endswith("_not_found") for f in e["flags"])
    assert e["counts"]["page_number_lines"] >= 6 and e["counts"]["repeated_header_lines"] >= 5
    assert set(ext["sections"]) == {"risk_factors", "legal_proceedings", "mdna"}


def test_10q_sections_part_aware():
    ext = m.extract_sections(ten_q(), "10-Q")
    s = ext["sections"]
    assert s["mdna"]["text"] == "Q_MDNA_START Quarterly revenue rose on anvil demand.\nQ_MDNA_END Liquidity remains adequate."
    assert s["mdna"]["end_heading"].startswith("Item 3.")
    # Part II Item 1, never Part I Item 1 (Financial Statements)
    assert s["legal_proceedings"]["text"].startswith("Q_LP")
    assert "FS_TEXT" not in s["legal_proceedings"]["text"]
    assert s["legal_proceedings"]["end_heading"].startswith("Item 1A.")
    assert s["legal_proceedings"]["flags"]["incorporated_by_reference"] is True
    rf = s["risk_factors"]
    assert rf["found"] and rf["text"].startswith("There have been no material changes")
    assert rf["flags"]["no_material_changes"] is True
    assert "Item 1A of our Annual Report on Form 10-K remains" in rf["text"]  # xref kept, not a heading
    assert rf["end_heading"].startswith("Item 2. Unregistered")
    assert "HIDDEN_Q" not in rf["text"]


def test_10q_part_inferred_from_titles_when_part_ii_heading_missing():
    s = m.extract_sections(ten_q(with_part2_heading=False), "10-Q")["sections"]
    assert s["legal_proceedings"]["text"].startswith("Q_LP")
    assert s["risk_factors"]["flags"]["no_material_changes"] is True
    assert s["mdna"]["text"].endswith("Liquidity remains adequate.")


def test_10q_amendment_form_uses_base_form_specs():
    s = m.extract_sections(ten_q(), "10-Q/A")["sections"]
    assert s["mdna"]["found"]


def test_missing_sections_flagged_not_fabricated():
    ext = m.extract_sections("<html><body><p>Just a cover page.</p></body></html>", "10-K")
    assert ext["extraction"]["flags"][0] == "no_item_headings"
    assert all(not s["found"] and s["text"] == "" for s in ext["sections"].values())


def test_cross_reference_index_format_flagged():
    html = ("<html><body><p>FORM 10-K CROSS-REFERENCE INDEX</p>"
            "<p>Item 1A. Risk Factors 26-33</p><p>Item 1B. Unresolved Staff Comments Not applicable</p>"
            "<p>Item 3. Legal Proceedings 73-74</p><p>Item 4. Mine Safety Disclosures Not applicable</p>"
            "<p>Item 7. Management's Discussion and Analysis 8-24</p><p>Item 7A. Quantitative 25</p>"
            "</body></html>")
    ext = m.extract_sections(html, "10-K")
    assert "cross_reference_index_format" in ext["extraction"]["flags"]


def test_section_running_to_document_end_is_flagged():
    html = "<html><body><p>Item 7. Management's Discussion and Analysis</p><p>" + "x " * 200 + "</p></body></html>"
    md = m.extract_sections(html, "10-K")["sections"]["mdna"]
    assert md["found"] and md["end_found"] is False and md["end_heading"] is None


def test_disclosure_flags():
    f = m.disclosure_flags("Not applicable.")
    assert f["not_applicable"] and not f["no_material_changes"]
    assert m.disclosure_flags("None.")["not_applicable"]
    # both phrasings seen in the gate run (SCND 10-Q 0001654954-24-006395)
    assert m.disclosure_flags("Not required for smaller reporting companies.")["not_applicable"]
    assert m.disclosure_flags("As a smaller reporting company, we are not required to provide this information.")[
        "not_applicable"]
    assert not m.disclosure_flags("None of our customers is material.")["not_applicable"]
    assert m.disclosure_flags("The information required by this item is incorporated herein by reference.")[
        "incorporated_by_reference"]
    long_text = "no material changes " + "substantive risk text. " * 400
    assert not m.disclosure_flags(long_text)["no_material_changes"]  # long section: not a stub


def test_heading_regex_variants():
    heads = m.find_headings(["ITEM 1A — RISK FACTORS", "Item 1A:Risk Factors", "Items 1 and 2. Business and Properties",
                             "Item 1A of Part I describes risks", "Item 7A", "Quantitative disclosures"], "10-K")
    codes = [(h["code"], h["title"]) for h in heads]
    assert codes == [("1A", "RISK FACTORS"), ("1A", "Risk Factors"), ("1", "Business and Properties"),
                     ("7A", "Quantitative disclosures")]


# ─────────────────────────────────────────────────────────────────────────────
# Listing + paging
# ─────────────────────────────────────────────────────────────────────────────

def _cols(rows):
    keys = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "primaryDocument"]
    return {k: [r.get(k) for r in rows] for k in keys}


def _paged_routes():
    recent = _cols([
        {"accessionNumber": "0000000001-25-000003", "filingDate": "2025-02-20", "reportDate": "2024-12-31",
         "acceptanceDateTime": "2025-02-20T21:00:00.000Z", "form": "10-K", "primaryDocument": "a-2024.htm"},
        {"accessionNumber": "0000000001-25-000004", "filingDate": "2025-03-01", "reportDate": "2024-12-31",
         "acceptanceDateTime": "2025-03-01T21:00:00.000Z", "form": "10-K/A", "primaryDocument": "a-2024a.htm"},
        {"accessionNumber": "0000000001-25-000005", "filingDate": "2025-03-02", "reportDate": None,
         "acceptanceDateTime": "2025-03-02T21:00:00.000Z", "form": "8-K", "primaryDocument": "8k.htm"},
    ])
    page1 = _cols([
        {"accessionNumber": "0000000001-12-000001", "filingDate": "2012-05-01", "reportDate": "2012-03-31",
         "acceptanceDateTime": "2012-05-01T20:00:00.000Z", "form": "10-Q", "primaryDocument": "q1.htm"},
        {"accessionNumber": "0000000001-09-000001", "filingDate": "2009-05-01", "reportDate": "2009-03-31",
         "acceptanceDateTime": "2009-05-01T20:00:00.000Z", "form": "10-Q", "primaryDocument": "q0.htm"},
    ])
    page2 = _cols([
        {"accessionNumber": "0000000001-05-000001", "filingDate": "2005-03-01", "reportDate": "2004-12-31",
         "acceptanceDateTime": "2005-03-01T20:00:00.000Z", "form": "10-K", "primaryDocument": "k04.txt"},
    ])
    return {
        m.SUBMISSIONS_URL.format(cik10="0000000001"): {"filings": {"recent": recent, "files": [
            {"name": "CIK0000000001-submissions-001.json", "filingFrom": "2009-01-01", "filingTo": "2015-01-01"},
            {"name": "CIK0000000001-submissions-002.json", "filingFrom": "2001-01-01", "filingTo": "2008-12-31"},
        ]}},
        m.SUBMISSIONS_PAGE_URL.format(name="CIK0000000001-submissions-001.json"): page1,
        m.SUBMISSIONS_PAGE_URL.format(name="CIK0000000001-submissions-002.json"): page2,
    }


def test_list_filings_reads_files_pages_and_flags_amendments():
    c, http, _ = client_for(_paged_routes())
    rows = m.list_filings(1, since="2010-01-01", client=c)
    assert [r["accession"] for r in rows] == ["0000000001-12-000001", "0000000001-25-000003", "0000000001-25-000004"]
    assert [r["isAmendment"] for r in rows] == [False, False, True]
    assert rows[0]["acceptanceDateTime"] == "2012-05-01T20:00:00.000Z"
    assert rows[0]["reportDate"] == "2012-03-31" and rows[0]["primaryDocument"] == "q1.htm"
    assert set(rows[0]) >= {"accession", "form", "filingDate", "acceptanceDateTime", "reportDate",
                            "primaryDocument", "isAmendment", "cik"}
    urls = [u for u, _ in http.calls]
    assert any(u.endswith("submissions-001.json") for u in urls)
    # page whose filingTo predates `since` cannot hold a qualifying filing: skipped
    assert not any(u.endswith("submissions-002.json") for u in urls)


def test_list_filings_without_since_reads_every_page():
    c, http, _ = client_for(_paged_routes())
    rows = m.list_filings("0000000001", since="", client=c)
    assert [r["accession"] for r in rows][0] == "0000000001-05-000001"
    assert len(rows) == 5  # 3 10-K/10-Q(/A) from pages + recent, 8-K excluded
    assert sum(1 for u, _ in http.calls if "submissions-00" in u) == 2


def test_list_filings_page_skip_never_changes_output():
    c1, _, _ = client_for(_paged_routes())
    c2, _, _ = client_for(_paged_routes())
    full = [r for r in m.list_filings(1, since="", client=c1) if r["filingDate"] >= "2010-01-01"]
    assert full == m.list_filings(1, since="2010-01-01", client=c2)


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter
# ─────────────────────────────────────────────────────────────────────────────

def _max_in_window(times, window=1.0):
    worst = 0
    for i, t in enumerate(times):
        worst = max(worst, sum(1 for u in times[i:] if u < t + window))
    return worst


def test_limiter_never_exceeds_hard_cap_even_if_asked_for_more():
    clk = FakeClock()
    lim = m.RateLimiter(rate=50, hard_cap=25, clock=clk, sleep=clk.sleep)
    assert lim.rate == 10 and lim.hard_cap == 10
    times = [lim.acquire() for _ in range(300)]
    assert _max_in_window(times) <= m.HARD_CAP_PER_SEC
    assert times == sorted(times)


def test_limiter_default_rate_is_8_per_second():
    clk = FakeClock()
    lim = m.RateLimiter(clock=clk, sleep=clk.sleep)
    times = [lim.acquire() for _ in range(161)]
    assert _max_in_window(times) <= 9  # evenly spaced at 1/8 s (burst 1)
    assert times[-1] - times[0] == pytest.approx(160 / 8.0, rel=1e-6)


def test_limiter_set_rate_clamped():
    lim = free_limiter()
    lim.set_rate(1000)
    assert lim.rate == 10
    with pytest.raises(ValueError):
        lim.set_rate(0)


def test_default_clients_share_one_process_limiter():
    a, b = m.EdgarClient(), m.EdgarClient()
    assert a.limiter is b.limiter is m.process_limiter()


# ─────────────────────────────────────────────────────────────────────────────
# Retries / backoff / Retry-After
# ─────────────────────────────────────────────────────────────────────────────

URL = "https://data.sec.gov/x.json"


def test_user_agent_matches_typescript_pipelines():
    for ts in ("server/edgarForm4.ts", "server/sec8kEarnings.ts"):
        with open(os.path.join(REPO, ts)) as f:
            assert f'"User-Agent": "{m.SEC_UA}"' in f.read()
    c, http, _ = client_for({URL: b"ok"})
    c.get(URL)
    assert http.calls[0][1]["User-Agent"] == m.SEC_UA


def test_retry_honors_retry_after_seconds():
    c, http, sleeps = client_for({URL: [m.HttpResponse(429, {"Retry-After": "7"}), m.HttpResponse(200, {}, b"ok")]})
    assert c.get(URL) == b"ok"
    assert sleeps == [7.0]
    assert c.stats["retries"] == 1


def test_retry_after_shorter_than_backoff_uses_backoff():
    c, _, sleeps = client_for({URL: [m.HttpResponse(503, {"retry-after": "0"}),
                                     m.HttpResponse(503, {"retry-after": "1"}), b"ok"]})
    assert c.get(URL) == b"ok"
    assert sleeps == [1.0, 2.0]


def test_exponential_backoff_on_5xx_then_success():
    c, _, sleeps = client_for({URL: [m.HttpResponse(500), m.HttpResponse(502), m.HttpResponse(503), b"ok"]})
    assert c.get(URL) == b"ok"
    assert sleeps == [1.0, 2.0, 4.0]


def test_retry_on_timeouts():
    c, _, sleeps = client_for({URL: [socket.timeout("t"), TimeoutError("t2"), b"ok"]})
    assert c.get(URL) == b"ok"
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_retries():
    c, http, sleeps = client_for({URL: m.HttpResponse(500)})
    with pytest.raises(m.EdgarHTTPError):
        c.get(URL)
    assert len(http.calls) == m.MAX_RETRIES + 1 == 5
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


def test_404_not_retried():
    c, http, sleeps = client_for({})
    with pytest.raises(m.EdgarHTTPError) as ei:
        c.get(URL)
    assert ei.value.status == 404 and len(http.calls) == 1 and sleeps == []


def test_every_attempt_goes_through_the_limiter():
    clk = FakeClock()
    lim = m.RateLimiter(rate=8, clock=clk, sleep=clk.sleep)
    http = FakeHTTP({URL: [m.HttpResponse(503), m.HttpResponse(503), b"ok"]})
    c = m.EdgarClient(http_get=http, limiter=lim, sleep=lambda s: None)
    c.get(URL)
    assert len(lim._recent) == 3


def test_parse_retry_after_http_date():
    from datetime import datetime, timezone
    now = datetime(2026, 10, 16, 12, 0, 0, tzinfo=timezone.utc)
    assert m.parse_retry_after("Fri, 16 Oct 2026 12:00:30 GMT", now=now) == 30.0
    assert m.parse_retry_after("Fri, 16 Oct 2026 11:00:00 GMT", now=now) == 0.0
    assert m.parse_retry_after("garbage") is None and m.parse_retry_after(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance time
# ─────────────────────────────────────────────────────────────────────────────

def test_index_headers_acceptance_is_eastern_converted_to_utc():
    page = "<pre>&lt;ACCEPTANCE-DATETIME&gt;20260731060102\n&lt;TYPE&gt;10-Q</pre>"
    assert m.parse_index_headers_acceptance(page) == "2026-07-31T10:01:02Z"  # EDT
    assert m.parse_index_headers_et(page) == "2026-07-31T06:01:02"
    assert m.parse_index_headers_acceptance("<ACCEPTANCE-DATETIME>20260130060132") == "2026-01-30T11:01:32Z"  # EST
    assert m.parse_index_headers_acceptance("no header") is None


def test_eastern_offset_dst_boundaries():
    from datetime import datetime as dt
    assert m.eastern_utc_offset_hours(dt(2026, 3, 8, 1, 59)) == 5
    assert m.eastern_utc_offset_hours(dt(2026, 3, 8, 3, 0)) == 4
    assert m.eastern_utc_offset_hours(dt(2026, 11, 1, 1, 59)) == 4
    assert m.eastern_utc_offset_hours(dt(2026, 11, 1, 2, 0)) == 5
    # pre-2007 rule: first Sunday April .. last Sunday October
    assert m.eastern_utc_offset_hours(dt(2005, 4, 3, 2, 0)) == 4
    assert m.eastern_utc_offset_hours(dt(2005, 4, 2, 12, 0)) == 5
    assert m.eastern_utc_offset_hours(dt(2005, 10, 30, 2, 0)) == 5
    assert m.eastern_utc_offset_hours(dt(2005, 10, 29, 12, 0)) == 4


def test_classify_json_acceptance_modes_from_verified_examples():
    # AAPL 10-Q 0000320193-26-000006: header 06:01:32 EST, JSON 16:01:32Z
    assert m.classify_json_acceptance("2026-01-30T16:01:32.000Z", "2026-01-30T11:01:32Z") == "utc_plus_et_offset"
    # GE 10-Q 0000040545-24-000176: header 06:28:10 EDT, JSON 10:28:10Z
    assert m.classify_json_acceptance("2024-07-23T10:28:10.000Z", "2024-07-23T10:28:10Z") == "utc"
    assert m.classify_json_acceptance("2024-07-23T12:28:10.000Z", "2024-07-23T10:28:10Z") == "other"
    assert m.classify_json_acceptance(None, "2024-07-23T10:28:10Z") is None


def test_conservative_fallback_never_earlier_than_truth():
    for raw, truth in [("2026-01-30T16:01:32.000Z", "2026-01-30T11:01:32Z"),
                       ("2024-07-23T10:28:10.000Z", "2024-07-23T10:28:10Z")]:
        assert m.json_acceptance_conservative_utc(raw) >= truth


# ─────────────────────────────────────────────────────────────────────────────
# Archive
# ─────────────────────────────────────────────────────────────────────────────

def test_write_once_gz_never_overwrites(tmp_path):
    p = str(tmp_path / "raw" / "0000000001" / "a.htm.gz")
    assert m.write_once_gz(p, b"first") is True
    assert m.write_once_gz(p, b"second") is False
    assert m.read_gz(p) == b"first"
    assert [f for f in os.listdir(os.path.dirname(p)) if f.endswith(".tmp")] == []


def test_write_once_gz_atomic_on_failure(tmp_path, monkeypatch):
    p = str(tmp_path / "x" / "b.htm.gz")

    def boom(fd):
        raise OSError("disk full")
    monkeypatch.setattr(m.os, "fsync", boom)
    with pytest.raises(OSError):
        m.write_once_gz(p, b"data")
    assert not os.path.exists(p)
    assert os.listdir(os.path.dirname(p)) == []  # temp file cleaned up


def test_write_once_gz_lost_race_keeps_first(tmp_path, monkeypatch):
    p = str(tmp_path / "c.htm.gz")
    real_link = os.link

    def racing_link(src, dst):
        with open(dst, "wb") as f:  # another writer publishes first
            f.write(gzip.compress(b"winner"))
        return real_link(src, dst)
    monkeypatch.setattr(m.os, "link", racing_link)
    assert m.write_once_gz(p, b"loser") is False
    assert m.read_gz(p) == b"winner"
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []


def test_gzip_is_deterministic(tmp_path):
    a, b = str(tmp_path / "a.gz"), str(tmp_path / "b.gz")
    m.write_once_gz(a, b"same")
    m.write_once_gz(b, b"same")
    assert open(a, "rb").read() == open(b, "rb").read()


def test_write_sections_rewrites_only_on_version_change(tmp_path):
    p = str(tmp_path / "s.json.gz")
    doc = {"extraction": {"version": "v1"}, "x": 1}
    assert m.write_sections(p, doc) is True
    assert m.write_sections(p, dict(doc, x=2)) is False
    assert json.loads(m.read_gz(p))["x"] == 1
    assert m.write_sections(p, {"extraction": {"version": "v2"}, "x": 3}) is True
    assert json.loads(m.read_gz(p))["x"] == 3


def test_archive_paths_and_base_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert m.archive_base_dir() == os.path.join(str(tmp_path), "datacore_archive")
    assert m.raw_path("/a", 320193, "0000320193-26-000020", "aapl-20260627.htm") == \
        "/a/filings_10kq/raw/0000320193/0000320193-26-000020.htm.gz"
    assert m.sections_path("/a", "320193", "0000320193-26-000020") == \
        "/a/filings_10kq/sections/0000320193/0000320193-26-000020.json.gz"
    assert m.doc_ext(None) == "bin" and m.doc_ext("x.TXT") == "txt"


def _filing(acc="0000000001-25-000003", form="10-K", doc="a-2024.htm"):
    return {"cik": "0000000001", "accession": acc, "form": form, "filingDate": "2025-02-20",
            "acceptanceDateTime": "2025-02-21T02:00:00.000Z", "reportDate": "2024-12-31",
            "primaryDocument": doc, "isAmendment": m.is_amendment(form)}


def _doc_routes(f, headers=True):
    nod = f["accession"].replace("-", "")
    r = {m.ARCHIVES_DOC_URL.format(cik=1, acc_nodash=nod, doc=f["primaryDocument"]): TEN_K.encode()}
    if headers:
        r[m.INDEX_HEADERS_URL.format(cik=1, acc_nodash=nod, acc=f["accession"])] = \
            b"&lt;ACCEPTANCE-DATETIME&gt;20250220160000"
    return r


def test_process_filing_end_to_end(tmp_path):
    f = _filing()
    c, http, _ = client_for(_doc_routes(f))
    out = m.process_filing(c, f, str(tmp_path))
    assert out["accepted_at"] == "2025-02-20T21:00:00Z" and out["accepted_at_source"] == "index_headers"
    assert out["acceptance_datetime_raw"] == "2025-02-21T02:00:00.000Z"
    assert out["_io"]["json_mode"] == "utc_plus_et_offset"
    assert out["period"] == "2024-12-31" and out["form"] == "10-K"
    raw = m.raw_path(str(tmp_path), f["cik"], f["accession"], f["primaryDocument"])
    assert m.read_gz(raw) == TEN_K.encode()
    stored = json.loads(m.read_gz(m.sections_path(str(tmp_path), f["cik"], f["accession"])))
    assert set(stored) >= {"cik", "accession", "form", "filing_date", "accepted_at", "period",
                           "sections", "extraction"}
    assert "_io" not in stored
    assert stored["sections"]["risk_factors"]["text"].startswith("RF_START")
    n = len(http.calls)
    m.process_filing(c, f, str(tmp_path))  # rerun: archived bytes win, nothing refetched
    assert len(http.calls) == n


def test_process_filing_without_index_headers_uses_conservative_json(tmp_path):
    f = _filing()
    c, _, _ = client_for(_doc_routes(f, headers=False))
    out = m.process_filing(c, f, str(tmp_path))
    assert out["accepted_at_source"] == "submissions_json_unverified"
    assert out["accepted_at"] == "2025-02-21T02:00:00Z"


# ─────────────────────────────────────────────────────────────────────────────
# Pairing
# ─────────────────────────────────────────────────────────────────────────────

def F(acc, form, period, filed=None, cik="0000000001"):
    return {"cik": cik, "accession": acc, "form": form, "reportDate": period,
            "filingDate": filed or period, "isAmendment": m.is_amendment(form)}


def test_pairing_10k_previous_fiscal_year():
    fs = [F("k23", "10-K", "2023-12-31"), F("k24", "10-K", "2024-12-31"), F("q24", "10-Q", "2024-12-31")]
    p = m.pair_prior_filings(fs)
    assert p["k24"] == {"prior": "k23", "gap_days": 366}
    assert p["k23"]["prior"] is None
    assert p["q24"]["prior"] is None  # 10-K is never a 10-Q's prior


def test_pairing_10q_same_fiscal_quarter_one_year_back():
    fs = [F("q1_24", "10-Q", "2024-03-31"), F("q2_24", "10-Q", "2024-06-30"), F("q3_24", "10-Q", "2024-09-30"),
          F("q1_25", "10-Q", "2025-03-31")]
    p = m.pair_prior_filings(fs)
    assert p["q1_25"]["prior"] == "q1_24"
    assert p["q3_24"]["prior"] is None


def test_pairing_52_53_week_fiscal_year():
    p = m.pair_prior_filings([F("a", "10-Q", "2023-07-01"), F("b", "10-Q", "2024-06-29")])
    assert p["b"] == {"prior": "a", "gap_days": 364}


def test_pairing_tolerance_edges():
    base = "2025-01-01"
    from datetime import date, timedelta
    ok = (date.fromisoformat(base) - timedelta(days=365 + 45)).isoformat()
    too_far = (date.fromisoformat(base) - timedelta(days=365 + 46)).isoformat()
    assert m.pair_prior_filings([F("a", "10-K", ok), F("b", "10-K", base)])["b"]["prior"] == "a"
    assert m.pair_prior_filings([F("a", "10-K", too_far), F("b", "10-K", base)])["b"]["prior"] is None


def test_pairing_excludes_amendments_by_default():
    fs = [F("k23", "10-K", "2023-12-31", "2024-02-20"), F("k23a", "10-K/A", "2023-12-31", "2024-04-01"),
          F("k24", "10-K", "2024-12-31")]
    p = m.pair_prior_filings(fs)
    assert "k23a" not in p and p["k24"]["prior"] == "k23"
    fs2 = [F("k23a", "10-K/A", "2023-12-31"), F("k24", "10-K", "2024-12-31")]
    assert m.pair_prior_filings(fs2)["k24"]["prior"] is None
    assert m.pair_prior_filings(fs2, include_amendments=True)["k24"]["prior"] == "k23a"


def test_pairing_tie_break_earlier_filed_and_other_cik_ignored():
    fs = [F("x2", "10-K", "2023-12-31", "2024-03-01"), F("x1", "10-K", "2023-12-31", "2024-02-01"),
          F("other", "10-K", "2023-12-31", "2024-01-15", cik="0000000002"), F("k24", "10-K", "2024-12-31")]
    assert m.pair_prior_filings(fs)["k24"]["prior"] == "x1"


def test_pairing_falls_back_to_filing_date():
    fs = [{"cik": "1", "accession": "a", "form": "10-K", "reportDate": None, "filingDate": "2024-02-20"},
          {"cik": "1", "accession": "b", "form": "10-K", "reportDate": None, "filingDate": "2025-02-18"}]
    assert m.pair_prior_filings(fs)["b"]["prior"] == "a"


# ─────────────────────────────────────────────────────────────────────────────
# Sampling + report (offline end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def test_stratified_sample_is_deterministic_and_spans_the_list():
    tick = {str(i): {"cik_str": 1000 + i // 2, "ticker": f"T{i}", "title": f"Co {i}"} for i in range(200)}
    entries = m.unique_ciks_in_rank_order(tick)
    assert len(entries) == 100 and entries[0]["ticker"] == "T0"  # first ticker per CIK kept
    s1 = m.stratified_strata(entries, 10, seed=7)
    s2 = m.stratified_strata(entries, 10, seed=7)
    assert s1 == s2
    assert m.stratified_strata(entries, 10, seed=8) != s1
    firsts = [s[0]["rank"] for s in s1]
    assert all(10 * i <= r < 10 * (i + 1) for i, r in enumerate(firsts))


def test_run_sample_offline_report(tmp_path):
    f = _filing()
    routes = _paged_routes()
    routes.update(_doc_routes(f))
    nod = "000000000125000004"
    routes[m.ARCHIVES_DOC_URL.format(cik=1, acc_nodash=nod, doc="a-2024a.htm")] = TEN_K.encode()
    routes[m.SUBMISSIONS_PAGE_URL.format(name="CIK0000000001-submissions-001.json")] = _cols([])
    routes[m.COMPANY_TICKERS_URL] = {"0": {"cik_str": 1, "ticker": "ACME", "title": "Acme"},
                                     "1": {"cik_str": 2, "ticker": "FUND", "title": "No 10-K filer"}}
    routes[m.SUBMISSIONS_URL.format(cik10="0000000002")] = {"filings": {"recent": _cols([]), "files": []}}
    c, _, _ = client_for(routes)
    rep = m.run_sample(1, "2025-01-01", str(tmp_path), seed=1, workers=2, client=c)
    assert "DATA-GATE REPORT" in rep
    assert "filings archived (originals)       1 10-K=1" in rep
    assert "amendments archived (excluded)     1" in rep
    assert "10-K  risk_factors          1/1" in rep
    assert "JSON acceptanceDateTime mode" in rep


def test_10k_code_only_fallback_for_typod_title():
    html = TEN_K.replace("ITEM 7. MANAGEMENT&#8217;S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION",
                         "ITEM 7. MANAGEMENT&#8217;S DISCUSSN AND ANYALYSI OF FINANCIAL CONDITION")
    html = html.replace("<p>MDNA_START Revenue grew", "<p>MDNA_START " + "Detailed discussion. " * 12 + "Revenue grew")
    md = m.extract_sections(html, "10-K")["sections"]["mdna"]
    assert md["found"] and md["title_mismatch_fallback"] is True
    assert md["text"].startswith("MDNA_START") and md["end_heading"].startswith("Item 7A.")
    # the matched ToC row stays a candidate but cannot win: it is a stub
    assert m.extract_sections(TEN_K, "10-K")["sections"]["mdna"]["title_mismatch_fallback"] is False


def test_10q_never_uses_code_only_fallback():
    html = ten_q().replace("<p>Item 2. Management&#8217;s Discussion", "<p>Item 2. Managemnt Discusion")
    s = m.extract_sections(html, "10-Q")["sections"]["mdna"]
    assert s["title_mismatch_fallback"] is False
    assert "Q_MDNA_START" not in s["text"]  # the ToC row is the only title-matched candidate


def test_cross_reference_index_with_page_numbers_on_own_lines():
    # Synchrony-style: Item rows split across lines, page numbers separate,
    # body organized under the company's own headings.
    html = ("<html><body><p>Item 1A.</p><p>Risk Factors</p><p>Item 1B.</p><p>Unresolved Staff Comments</p>"
            "<p>Item 3.</p><p>Legal Proceedings</p><p>Item 4.</p><p>Mine Safety</p>"
            "<p>Item 7.</p><p>Management's Discussion and Analysis</p><p>Item 7A.</p><p>Market Risk</p>"
            "<p>RISK FACTORS RELATING TO OUR BUSINESS</p><p>" + "Body text. " * 100 + "</p></body></html>")
    flags = m.extract_sections(html, "10-K")["extraction"]["flags"]
    assert "no_substantive_sections" in flags and "cross_reference_index_format" in flags


def test_no_substantive_sections_without_xref_signature():
    html = "<html><body><p>Item 9. Changes in Accountants</p><p>None.</p><p>Item 10. Directors</p></body></html>"
    flags = m.extract_sections(html, "10-K")["extraction"]["flags"]
    assert "no_substantive_sections" in flags and "cross_reference_index_format" not in flags


def test_write_sections_upgrades_fallback_timestamp_once(tmp_path):
    # A filing first saved with the unverified JSON time must be rewritten
    # when the verified index-headers time arrives — same extraction version.
    p = str(tmp_path / "s.json.gz")
    base = {"extraction": {"version": "v1"}}
    assert m.write_sections(p, dict(base, accepted_at="2025-01-01T15:00:00Z",
                                     accepted_at_source="submissions_json_unverified")) is True
    assert m.write_sections(p, dict(base, accepted_at="2025-01-01T11:00:00Z",
                                     accepted_at_source="index_headers")) is True
    saved = json.loads(m.read_gz(p))
    assert saved["accepted_at_source"] == "index_headers"
    assert saved["accepted_at"] == "2025-01-01T11:00:00Z"
    # Verified copy is never downgraded or rewritten again at the same version.
    assert m.write_sections(p, dict(base, accepted_at="2025-01-01T15:00:00Z",
                                     accepted_at_source="submissions_json_unverified")) is False
    assert m.write_sections(p, dict(base, accepted_at="2025-01-01T11:00:00Z",
                                     accepted_at_source="index_headers")) is False
