#!/usr/bin/env python3
"""FAA chart bake — the whole cycle, unattended.

    python3 scripts/faa_charts/run.py                 # newest edition, all families, publish
    python3 scripts/faa_charts/run.py --families sectional --no-upload --work /tmp/x

For each chart family (sectional, tac, ifrlow, ifrhigh):
  1. find the newest FAA edition whose files are posted (cycle.py — from the
     FAA's own directory listings; nothing hand-maintained);
  2. skip it if the published manifest already carries that edition;
  3. download + unzip the GeoTIFFs (one zip at a time, zip deleted after);
  4. reuse the stored map edges for every chart whose georeferencing
     fingerprint is unchanged, otherwise re-measure the family's edges
     against the FAA mosaic (edges.py);
  5. bake one PMTiles (bake.py);
  6. GATE: compare sampled tiles with the FAA mosaic (verify.py) — a failed
     gate publishes nothing and the site keeps serving the FAA service;
  7. upload the PMTiles + updated edges to R2, then the manifest LAST (the
     server switches only on a manifest that names a fully uploaded file),
     and delete editions older than the one being replaced.

Publishing needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and
R2_TILES_BUCKET (the public tiles bucket behind R2_PUBLIC_URL). Exit code 0
= every family is current (baked now or already published); 1 = a family
failed (its reason is in the JSON summary).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cycle  # noqa: E402
from common import ChartSource  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EDGES_DIR = os.path.join(REPO, "datacore", "faa_charts", "edges")
R2_PREFIX = "tiles/faa"
MANIFEST_KEY = f"{R2_PREFIX}/manifest.json"
VERIFY_ZOOM = {"sectional": 10, "tac": 11, "ifrlow": 10, "ifrhigh": 8}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ── manifest + R2 ───────────────────────────────────────────────────────────

def public_get_json(key: str):
    base = (os.environ.get("R2_PUBLIC_URL") or "").rstrip("/")
    if not base:
        return None
    try:
        # Cloudflare answers 403 to Python's default user agent
        req = urllib.request.Request(f"{base}/{key}?t={int(time.time())}", headers={"user-agent": "VolTradeAI-chartbake/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def r2_client():
    missing = [k for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_TILES_BUCKET")
               if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"cannot publish: missing environment variable(s) {', '.join(missing)}")
    try:
        import boto3  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "boto3"])
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(retries={"max_attempts": 6, "mode": "standard"}),
    ), os.environ["R2_TILES_BUCKET"]


def r2_put_file(path: str, key: str, content_type: str):
    from boto3.s3.transfer import TransferConfig

    s3, bucket = r2_client()
    s3.upload_file(path, bucket, key, ExtraArgs={"ContentType": content_type},
                   Config=TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=8))
    head = s3.head_object(Bucket=bucket, Key=key)
    if head["ContentLength"] != os.path.getsize(path):
        raise RuntimeError(f"upload size mismatch for {key}")


def r2_put_json(obj, key: str):
    s3, bucket = r2_client()
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(obj, indent=1).encode(),
                  ContentType="application/json", CacheControl="no-cache")


def r2_delete(key: str):
    s3, bucket = r2_client()
    s3.delete_object(Bucket=bucket, Key=key)


# ── download ────────────────────────────────────────────────────────────────

def download_family(fam, edition: dt.date, work: str) -> list:
    """[(chart name, tif path)] — chart name = the GeoTIFF's file stem, the
    FAA's stable name for a chart across editions."""
    names = cycle.list_family(fam, edition)
    if not names:
        raise RuntimeError(f"no files listed for {fam.id} {edition}")
    os.makedirs(work, exist_ok=True)
    specs = []
    for n in names:
        url = cycle.family_url(fam, edition, n)
        zp = os.path.join(work, n)
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"user-agent": "VolTradeAI-chartbake/1.0"})
                with urllib.request.urlopen(req, timeout=120) as r, open(zp, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                with zipfile.ZipFile(zp) as z:
                    for m in z.namelist():
                        if m.lower().endswith(".tif"):
                            z.extract(m, work)
                            specs.append((os.path.splitext(os.path.basename(m))[0], os.path.join(work, m)))
                break
            except (urllib.error.URLError, zipfile.BadZipFile, ConnectionError, TimeoutError) as e:
                if attempt == 3:
                    raise RuntimeError(f"download failed: {url}: {e}")
                time.sleep(5 * (attempt + 1))
            finally:
                if os.path.exists(zp):
                    os.remove(zp)
    log(f"[{fam.id}] {len(names)} zips -> {len(specs)} GeoTIFFs")
    return specs


# ── edges ───────────────────────────────────────────────────────────────────

def load_edges(fam_id: str):
    """Published edges (R2) win over the repo seed: a re-measurement made by
    an unattended run is stored there without a git commit."""
    pub = public_get_json(f"{R2_PREFIX}/edges/{fam_id}.json")
    if pub:
        return pub, "r2"
    p = os.path.join(EDGES_DIR, f"{fam_id}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f), "repo"
    return None, None


def edges_reusable(edges_json, charts) -> tuple:
    """(reusable, reasons). Every chart the stored edges know must still be
    present with an unchanged fingerprint; every new chart forces a
    re-measurement too (it may own part of the mosaic)."""
    from edges import fingerprint, fingerprint_matches

    if not edges_json:
        return False, ["no stored edges"]
    stored = edges_json.get("charts", {})
    reasons = []
    by_name = {c.name: c for c in charts}
    for name, e in stored.items():
        c = by_name.get(name)
        if c is None:
            reasons.append(f"{name}: no longer published")
        elif not fingerprint_matches(e.get("fingerprint"), fingerprint(c)):
            reasons.append(f"{name}: georeferencing changed")
    known = set(stored) | set(edges_json.get("excluded", []))
    for name in by_name:
        if name not in known:
            reasons.append(f"{name}: new chart")
    return (not reasons), reasons


# ── one family ──────────────────────────────────────────────────────────────

def run_family(fam, today: dt.date, args, manifest: dict) -> dict:
    import bake
    import edges
    import verify

    out = {"family": fam.id}
    edition = dt.date.fromisoformat(args.edition) if args.edition else cycle.newest_published(fam, today)
    if not edition:
        return {**out, "status": "error", "reason": "no published edition found on aeronav"}
    ed = edition.isoformat()
    out["edition"] = ed
    cur = (manifest.get("families") or {}).get(fam.id) or {}
    if cur.get("edition") == ed and not args.force:
        return {**out, "status": "current"}

    work = os.path.join(args.work, fam.id)
    src_dir = os.path.join(work, "src")
    marker = os.path.join(src_dir, f"_complete_{ed}")
    if args.keep_src and os.path.exists(marker):
        with open(marker) as f:
            specs = [tuple(x) for x in json.load(f)]
    else:
        shutil.rmtree(work, ignore_errors=True)
        specs = download_family(fam, edition, src_dir)
        with open(marker, "w") as f:
            json.dump(specs, f)
    charts = [ChartSource.open(n, p) for n, p in specs]

    if args.edges:
        with open(args.edges) as f:
            edges_json, origin = json.load(f), "file"
    else:
        edges_json, origin = load_edges(fam.id)
    ok, reasons = edges_reusable(edges_json, charts)
    if not ok or args.remeasure:
        log(f"[{fam.id}] measuring chart edges ({'; '.join(reasons[:6]) or 'forced'})")
        res = edges.measure_family(fam, charts, os.path.join(args.work, "faacache"), log,
                                   coarse_cache=os.path.join(args.work, f"{fam.id}-coarse.pkl"))
        edges_json = edges.to_json(fam.id, ed, res, {c.name: c for c in charts})
        edges_json["excluded"] = sorted(c.name for c in charts if c.name not in edges_json["charts"])
        origin = "measured"
    out["edges"] = origin
    edges_path = os.path.join(args.work, f"{fam.id}-edges.json")
    with open(edges_path, "w") as f:
        json.dump(edges_json, f)  # kept locally too: the seed for datacore/faa_charts/edges/
    if args.measure_only:
        return {**out, "status": "measured", "edges_path": edges_path}
    pm = os.path.join(args.work, f"{fam.id}-{ed}.pmtiles")
    rep = bake.bake_family(fam, specs, edges_json, pm, os.path.join(work, "tiles"), ed, log=log)
    for c in charts:
        c.ds.close()
    if not args.keep_src:
        shutil.rmtree(src_dir, ignore_errors=True)
    out["bake"] = rep
    gate = verify.verify_pmtiles(fam, pm, os.path.join(args.work, "faacache"), VERIFY_ZOOM[fam.id])
    out["verify"] = gate
    log(f"[{fam.id}] gate: {'PASS' if gate['ok'] else 'FAIL'} coverage={gate['coverage']} "
        f"spill={gate['spill']} diff={gate['median_structural_diff']}")
    if not gate["ok"]:
        return {**out, "status": "gate-failed"}
    if args.no_upload:
        return {**out, "status": "baked", "pmtiles": pm}

    key = f"{R2_PREFIX}/{fam.id}/{ed}.pmtiles"
    log(f"[{fam.id}] uploading {os.path.getsize(pm) / 1e9:.2f} GB -> {key}")
    r2_put_file(pm, key, "application/vnd.pmtiles")
    if origin == "measured":
        r2_put_file(edges_path, f"{R2_PREFIX}/edges/{fam.id}.json", "application/json")
    entry = {
        "edition": ed, "key": key, "bytes": os.path.getsize(pm),
        "min_zoom": rep["min_zoom"], "max_zoom": rep["max_zoom"], "faa_min_zoom": fam.min_zoom,
        "tile_type": "webp", "baked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "charts": rep["charts"], "verify": {k: gate[k] for k in ("coverage", "spill", "median_structural_diff", "sampled_tiles")},
        # the edition being replaced stays servable: a just-posted cycle is
        # not in force until its date, and until then this one is current
        "previous": {k: v for k, v in cur.items() if k != "previous"} or None,
    }
    out["entry"] = entry
    out["delete_key"] = (cur.get("previous") or {}).get("key")
    out["status"] = "published"
    os.remove(pm)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=",".join(cycle.FAMILIES))
    ap.add_argument("--edition", help="YYYY-MM-DD (default: newest published)")
    ap.add_argument("--work", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "faa_chart_bake"))
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-bake even if the manifest is current")
    ap.add_argument("--remeasure", action="store_true", help="re-measure edges even if fingerprints match")
    ap.add_argument("--measure-only", action="store_true", help="stop after the edges (written to --work)")
    ap.add_argument("--edges", help="use this edges JSON instead of the published/repo one (one family)")
    ap.add_argument("--keep-src", action="store_true", help="keep the downloaded GeoTIFFs (re-runs skip the download)")
    args = ap.parse_args(argv)
    today = dt.datetime.now(dt.timezone.utc).date()
    manifest = public_get_json(MANIFEST_KEY) or {"version": 1, "families": {}}
    results = []
    for fid in [f.strip() for f in args.families.split(",") if f.strip()]:
        fam = cycle.FAMILIES[fid]
        try:
            r = run_family(fam, today, args, manifest)
        except Exception as e:  # one family failing never blocks the others
            r = {"family": fid, "status": "error", "reason": f"{type(e).__name__}: {e}"}
        results.append(r)
        log(f"[{fid}] {r['status']}" + (f" — {r.get('reason')}" if r.get("reason") else ""))
        if r["status"] == "published":
            manifest.setdefault("families", {})[fid] = r["entry"]
            manifest["updated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            r2_put_json(manifest, MANIFEST_KEY)  # LAST: the server switches on this
            # keep exactly one older edition (the one just replaced) for a
            # rollback; the edition before THAT is deleted
            old = r.get("delete_key")
            if old and old != r["entry"]["key"]:
                try:
                    r2_delete(old)
                except Exception as e:
                    log(f"[{fid}] could not delete {old}: {e}")
    print(json.dumps({"results": results}, indent=1, default=str))
    return 0 if all(r["status"] in ("current", "published", "baked", "measured") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
