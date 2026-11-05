"""Offline tests for sec_issuer_pairs.py: all-form issuer pairs, counted drops, cache roundtrip."""
import io
import zipfile

import sec_cik_symbol_map as m
import sec_issuer_pairs as ip

TSV = (b"ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\tISSUERTRADINGSYMBOL\n"
       b"a1\t05-JAN-2016\t3\t0000001234\tOLDCO\n"
       b"a2\t06-JAN-2016\t5\t0000001234\toldco\n"
       b"a3\t06-JAN-2016\t4\t0000001234\tOLDCO\n"        # duplicate (ticker,cik,day) collapses
       b"a4\t07-JAN-2016\t4\t55\tGEF, GEF-B\n"            # dual class: rejected
       b"a5\t07-JAN-2016\t4\t\tXYZ\n"                     # no cik
       b"a6\t99-ZZZ-2016\t4\t56\tABC\n"                   # bad date
       b"a7\t08-JAN-2016\t4\t57\tN/A\n")                  # placeholder


def test_pairs_dedupe_and_every_drop_is_counted():
    sk = {}
    got = ip.pairs_from_submission_tsv(TSV, sk)
    assert got == [("OLDCO", "0000001234", "2016-01-05"), ("OLDCO", "0000001234", "2016-01-06")]
    assert sk == {"bad_ticker": 2, "no_cik": 1, "bad_date": 1}


def test_any_document_type_counts_unlike_the_p2_cache():
    # Form 3 (initial statement, no transaction) still identifies the issuer.
    assert ip.pairs_from_submission_tsv(TSV)[0][2] == "2016-01-05"


def test_zip_cache_roundtrip_feeds_build_map(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SUBMISSION.tsv", TSV)
    ip.write_pairs(str(tmp_path), "2016q1", ip.pairs_from_zip_bytes(buf.getvalue()))
    sk = {}
    rows = list(ip.rows_from_pairs_cache(str(tmp_path), sk))
    assert sk == {}
    mp = m.build_map(rows)
    assert m.resolve(mp, "1234", "2016-01-05") == ("OLDCO", "exact")
    assert m.resolve(mp, "1234", "2016-03-01") == ("OLDCO", "padded")
