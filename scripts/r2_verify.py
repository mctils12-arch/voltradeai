#!/usr/bin/env python3
"""r2_verify.py — prove the Cloudflare R2 write path works, end to end.

WHY THIS EXISTS (EDGE DOCTRINE #3: "never analyze the same thing twice with
reasoning — the second occurrence becomes a script"). On 2026-08-12 a
session spent real effort re-deriving how R2 is wired, because the original
upload procedure for the 135 pmtiles objects in the bucket was never
recorded — no script, no runbook, no logged command. That is a live instance
of the doctrine failure the constitution names. This script is the fix: the
next session runs it instead of reasoning.

WHAT IT CHECKS, in order, stopping at the first real failure:
  1. every required env var is present (names only, never values)
  2. the PUBLIC read path serves (the pub-*.r2.dev URL the site uses)
  3. the S3 WRITE endpoint is reachable and the credentials authenticate
  4. a round trip: put a small object, read it back, delete it

Usage:
    python3 scripts/r2_verify.py                 # check + round trip
    python3 scripts/r2_verify.py --no-write      # read-only checks
    R2_ENDPOINT=https://<acct>.r2.cloudflarestorage.com python3 scripts/r2_verify.py

ENV IT READS:
    R2_ACCESS_KEY_ID / R2_ACCESS_KEY       access key
    R2_SECRET_ACCESS_KEY / R2_SECRET       secret
    R2_PUBLIC_URL                          public read base (pub-*.r2.dev)
    R2_ENDPOINT                            S3 API base — the one usually missing.
                                           Accepts a full URL or a bare 32-hex
                                           account id.
    R2_BUCKET                              bucket name (default: voltradeai-tiles)

FINDING THE ACCOUNT ID (the value people get stuck on):
    Cloudflare dashboard -> R2 -> your bucket -> Settings -> "S3 API" shows
    the endpoint already assembled. Or read it out of the dashboard URL:
    dash.cloudflare.com/<ACCOUNT_ID>/r2/overview
    It is NOT the access key id, and NOT the `pub-<hash>` in the public URL —
    both are 32 hex chars and both are wrong. Verified 2026-08-12.

NOTE ON TLS: this container proxies outbound HTTPS. If boto3 raises
SSLError, pass the proxy CA via AWS_CA_BUNDLE=/root/.ccr/ca-bundle.crt
(this script does that automatically when the file exists).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

CA_BUNDLE = "/root/.ccr/ca-bundle.crt"
DEFAULT_BUCKET = "voltradeai-tiles"


def env_any(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def normalize_endpoint(raw: str | None) -> str | None:
    """Accept a full URL or a bare account id; return a full S3 endpoint."""
    if not raw:
        return None
    raw = raw.strip().rstrip("/")
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    # bare account id
    return f"https://{raw}.r2.cloudflarestorage.com"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the R2 read + write path")
    ap.add_argument("--no-write", action="store_true", help="skip the write round trip")
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", DEFAULT_BUCKET))
    args = ap.parse_args()

    ok = True

    # ── 1. credentials present ────────────────────────────────────────────
    ak = env_any("R2_ACCESS_KEY_ID", "R2_ACCESS_KEY")
    sk = env_any("R2_SECRET_ACCESS_KEY", "R2_SECRET")
    pub = os.environ.get("R2_PUBLIC_URL")
    endpoint = normalize_endpoint(os.environ.get("R2_ENDPOINT"))

    print("── env ──")
    print(f"  access key      {'present' if ak else 'MISSING'}")
    print(f"  secret          {'present' if sk else 'MISSING'}")
    print(f"  public url      {pub or 'MISSING'}")
    print(f"  s3 endpoint     {endpoint or 'MISSING'}")
    print(f"  bucket          {args.bucket}")

    if not (ak and sk):
        print("\nFAIL: credentials missing. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY.")
        return 1

    # ── 2. public read path ───────────────────────────────────────────────
    if pub:
        try:
            import urllib.request

            req = urllib.request.Request(pub, method="GET")
            try:
                urllib.request.urlopen(req, timeout=15)
                print("\n── public read ──\n  reachable (200)")
            except Exception as e:  # a 404 at the bucket root is EXPECTED and fine
                code = getattr(e, "code", None)
                if code in (403, 404):
                    print(f"\n── public read ──\n  reachable (HTTP {code} at bucket root — expected)")
                else:
                    print(f"\n── public read ──\n  WARN: {type(e).__name__}: {str(e)[:80]}")
                    ok = False
        except Exception as e:
            print(f"\n── public read ──\n  WARN: {type(e).__name__}")

    # ── 3. S3 endpoint + auth ─────────────────────────────────────────────
    if not endpoint:
        print(
            "\nSTOP: R2_ENDPOINT is not set — this is the S3 API base used for WRITES.\n"
            "  Public reads (pub-*.r2.dev) work without it; uploads cannot.\n"
            "  Find it: Cloudflare -> R2 -> bucket -> Settings -> 'S3 API',\n"
            "  or read the account id out of the dashboard URL.\n"
            "  It is NOT the access key id and NOT the pub-<hash>."
        )
        return 2

    if os.path.exists(CA_BUNDLE):
        os.environ.setdefault("AWS_CA_BUNDLE", CA_BUNDLE)

    try:
        import boto3
        import botocore
        from botocore.config import Config
    except ImportError:
        print("\nFAIL: boto3 not installed (pip install boto3)")
        return 1

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="auto",
        config=Config(connect_timeout=10, read_timeout=20, retries={"max_attempts": 1}),
    )

    print("\n── s3 auth ──")
    try:
        resp = s3.list_buckets()
        names = [b["Name"] for b in resp.get("Buckets", [])]
        print(f"  authenticated. buckets visible: {names or '(none)'}")
        if args.bucket not in names and names:
            print(f"  WARN: '{args.bucket}' not among them — check --bucket / R2_BUCKET")
            ok = False
    except botocore.exceptions.ClientError as e:
        err = e.response.get("Error", {})
        print(f"  FAIL: {err.get('Code')} — {str(err.get('Message'))[:100]}")
        print("  (InvalidAccessKeyId/SignatureDoesNotMatch = wrong creds;")
        print("   a TLS/connection error usually means a WRONG ACCOUNT ID in R2_ENDPOINT)")
        return 1
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:140]}")
        print("  A TLS handshake failure here almost always means the account id")
        print("  in R2_ENDPOINT is wrong — Cloudflare wildcards DNS, so a bad id")
        print("  still resolves but will not complete TLS.")
        return 1

    # ── 4. write round trip ───────────────────────────────────────────────
    if args.no_write:
        print("\n(--no-write: skipping the round trip)")
        return 0 if ok else 1

    key = f"_verify/r2_verify_{int(time.time())}.txt"
    body = b"voltradeai r2_verify round trip\n"
    print("\n── write round trip ──")
    try:
        s3.put_object(Bucket=args.bucket, Key=key, Body=body, ContentType="text/plain")
        print(f"  put    {key}")
        got = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read()
        assert got == body, "round-trip body mismatch"
        print("  get    matches")
        s3.delete_object(Bucket=args.bucket, Key=key)
        print("  delete cleaned up")
        print("\nPASS: R2 write path is fully working. A bake can upload directly.")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:140]}")
        print("  Credentials authenticated but writing failed — check the token's")
        print("  permissions (needs Object Read & Write, not just Read).")
        return 1

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
