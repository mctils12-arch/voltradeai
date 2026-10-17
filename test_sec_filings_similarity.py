"""Offline tests for sec_filings_similarity.py (Lazy Prices PR 2c)."""
import math

import sec_filings_similarity as s

WORDS = ("revenue customers competition regulation supply inflation litigation cyber "
         "currency tariff interest liquidity covenant pension warranty recall").split()
TEXT = " ".join(WORDS * 8)


def sec(text, **kw):
    d = {"found": True, "short": False, "text": text,
         "flags": {"no_material_changes": False, "not_applicable": False, "incorporated_by_reference": False}}
    d.update(kw)
    return d


def test_identical_text_scores_one():
    r = s.section_similarity(sec(TEXT), sec(TEXT))
    assert r["excluded"] is None and math.isclose(r["cosine"], 1.0) and r["jaccard"] == 1.0


def test_disjoint_vocab_scores_zero_and_numbers_ignored():
    other = " ".join(w + "x" for w in WORDS * 8)
    r = s.section_similarity(sec(TEXT), sec(other))
    assert r["cosine"] == 0.0 and r["jaccard"] == 0.0
    assert s.tokenize("Revenue rose 12.5% in 2024") == ["revenue", "rose", "in"]


def test_partial_change_between_zero_and_one():
    changed = TEXT.replace("tariff", "pandemic").replace("recall", "drought")
    r = s.section_similarity(sec(TEXT), sec(changed))
    assert 0.5 < r["cosine"] < 1.0 and r["jaccard"] < 1.0


def test_exclusions_never_scored():
    assert s.section_similarity(sec(TEXT, found=False), sec(TEXT))["excluded"] == "not_found"
    assert s.section_similarity(sec(TEXT), sec(TEXT, short=True))["excluded"] == "short"
    nmc = sec(TEXT, flags={"no_material_changes": True})
    r = s.section_similarity(nmc, sec(TEXT))
    assert r["excluded"] == "no_material_changes" and r["cosine"] is None
    assert s.section_similarity(sec("a few words only"), sec(TEXT))["excluded"] == "too_few_tokens"
    assert s.section_similarity(None, sec(TEXT))["excluded"] == "not_found"


def doc(acc, period, text):
    return {"cik": "0000000005", "accession": acc, "form": "10-K", "period": period,
            "accepted_at": "2024-02-20T21:00:00Z", "accepted_at_source": "index_headers",
            "sections": {"risk_factors": sec(text)}}


def test_build_rows_pairs_prior_year_and_carries_current_accepted_at():
    filings = [{"cik": "5", "accession": "A-2", "form": "10-K", "reportDate": "2023-12-31", "filingDate": "2024-02-20"},
               {"cik": "5", "accession": "A-1", "form": "10-K", "reportDate": "2022-12-31", "filingDate": "2023-02-20"},
               {"cik": "5", "accession": "A-0", "form": "10-K", "reportDate": "2020-12-31", "filingDate": "2021-02-20"}]
    docs = {"A-2": doc("A-2", "2023-12-31", TEXT), "A-1": doc("A-1", "2022-12-31", TEXT)}
    rows = s.build_rows(filings, lambda cik, acc: docs.get(acc))
    assert len(rows) == 1                      # A-1 has no prior with a doc; A-0 unpaired
    r = rows[0]
    assert r["accession"] == "A-2" and r["prior_accession"] == "A-1"
    assert r["accepted_at"] == "2024-02-20T21:00:00Z" and r["gap_days"] == 365
    assert math.isclose(r["sections"]["risk_factors"]["cosine"], 1.0)


def test_archive_roundtrip_and_distribution_report(tmp_path):
    """PR 2d: rows are rebuilt from the sections archive alone (no network)."""
    import sec_filings_text as sft
    arch = str(tmp_path)
    for acc, per, fd, text in (("A-2", "2023-12-31", "2024-02-20", TEXT),
                               ("A-1", "2022-12-31", "2023-02-20", TEXT.replace("tariff", "pandemic"))):
        d = doc(acc, per, text)
        d["filing_date"] = fd
        p = sft.sections_path(arch, "5", acc)
        import os
        os.makedirs(os.path.dirname(p), exist_ok=True)
        sft.write_once_gz(p, __import__("json").dumps(d).encode())
    rows = s.rows_from_archive(arch)
    assert len(rows) == 1 and rows[0]["accession"] == "A-2"
    rep = s.distribution_report(rows)
    assert "pairs: 1" in rep and "risk_factors" in rep and "10-K" in rep


def test_distribution_report_counts_exclusions():
    rows = [{"form": "10-Q", "sections": {"mdna": {"excluded": "short", "cosine": None, "jaccard": None}}}]
    assert "'short': 1" in s.distribution_report(rows)
