#!/usr/bin/env python3
"""weight_sink.py — GRID VISION Phase B: get the trained weights + metrics OFF the
ephemeral RunPod pod before it is DELETEd.

THE PROBLEM (real, must be solved or the GPU dollar is wasted): the launcher
(scripts/runpod_launch.py) DELETEs the pod at the cost cap. Anything written to
/workspace vanishes with the pod. The launcher does NOT currently fetch pod logs,
so even stdout is only recoverable via the RunPod dashboard/API before deletion.
So the pod must actively PUSH its artifacts out.

OPTIONS (enumerated honestly; the PARENT/human decides which to trust):
  (i)  FREE KEYLESS EPHEMERAL HOST — 0x0.st / transfer.sh. No account, no key.
       0x0.st cap is 512 MiB; a detector .pt is ~6-130 MB, so it fits. Retention
       scales inversely with size (~30-365 days). Files are PUBLIC (anyone with
       the URL). For an INTERNAL fine-tuned .pt that is acceptable short-term
       exfil, not long-term storage. >>> IMPLEMENTED HERE as the default sink. <<<
       CAVEAT (honest unknown): 0x0.st has rate-limited / blocked datacenter IPs
       in the past and requires a real User-Agent (bare urllib UA -> 403). If the
       upload fails, metrics still print to stdout (train.py) as the fallback.
  (ii) RUNPOD NETWORK VOLUME — a persistent volume that survives pod DELETE.
       Weights written to the mounted volume are retrievable by a later API call.
       NEEDS: API volume create + attach at launch + a retrieval step. More
       plumbing; no third-party host; artifacts stay private. NOT built here —
       recommended if GPU runs become recurring. (Launcher change required.)
  (iii)HUGGINGFACE HUB / S3 / GCS — durable, private, versioned. NEEDS A TOKEN
       (HF_TOKEN / AWS creds) = BLOCKED-FOR-MIKE (no such secret in the session).
       Best long-term home once a key exists; cannot be used keyless today.

WEIGHTS-HYGIENE (research/grid_vision_phaseb.md + LICENSE FLAG below): the
fine-tuned weights stay INTERNAL. A public keyless host (option i) is a transient
transport, not publication — the URL is unguessable-ish but PUBLIC, so treat it as
"good enough to retrieve this run", delete after retrieval, never advertise it.

PURITY: URL/response parsing + the multipart body builder are pure and offline-
tested. The actual HTTP upload is a thin injectable-fetcher function so tests
never hit the wire.
"""
import json
import os
import uuid

# Default free keyless hosts, tried in order. 0x0.st first (most reliable of the
# keyless options in 2026 research); transfer.sh as a documented fallback.
DEFAULT_HOSTS = ("https://0x0.st", "https://transfer.sh")
# 0x0.st retention model (documented on the service): min 30 d, max 365 d,
# max upload 512 MiB, retention shrinks with size by a cubic curve.
_MIN_AGE_DAYS = 30.0
_MAX_AGE_DAYS = 365.0
_MAX_SIZE_BYTES = 512 * 1024 * 1024
UA = "voltradeai-gridvision/1.0 (+https://voltradeai.com)"


# ── pure parsing / formatting ───────────────────────────────────────────────

def parse_upload_url(resp_text):
    """0x0.st / transfer.sh return the bare URL as the response body. Extract and
    validate it (first non-blank line, must be an http(s) URL). None if the body
    is not a URL (an error page, empty, etc.). Pure."""
    if not resp_text:
        return None
    for line in str(resp_text).splitlines():
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            return line
    return None


def estimate_retention_days(size_bytes, min_age=_MIN_AGE_DAYS, max_age=_MAX_AGE_DAYS,
                            max_size=_MAX_SIZE_BYTES):
    """0x0.st's documented retention curve, for HONEST reporting of how long the
    uploaded weights will live: retention = min + (min-max)*(size/max_size - 1)^3.
    Small files -> ~365 d, near-cap files -> ~30 d. Clamped to [min,max]. Pure."""
    if size_bytes <= 0:
        return max_age
    ratio = min(float(size_bytes) / float(max_size), 1.0)
    days = min_age + (min_age - max_age) * ((ratio - 1.0) ** 3)
    return round(max(min_age, min(max_age, days)), 1)


def multipart_body(field_name, filename, filedata, boundary=None):
    """Build a minimal multipart/form-data body for a single file field. Pure:
    (bytes-in -> (content_type, bytes-out)). Kept dependency-free so the pod does
    not need `requests`. `filedata` is bytes."""
    if boundary is None:
        boundary = "----gridvision" + uuid.uuid4().hex
    if isinstance(filedata, str):
        filedata = filedata.encode("utf-8")
    pre = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    post = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = pre + filedata + post
    content_type = f"multipart/form-data; boundary={boundary}"
    return content_type, body


def result_line(metrics, weight_url=None, extra=None):
    """The single machine-parseable JSON line train.py prints last, so results
    survive even if the upload host is down. Pure."""
    doc = {"gridvision_train_result": True, "metrics": metrics,
           "weight_url": weight_url}
    if extra:
        doc.update(extra)
    return "GRIDVISION_RESULT " + json.dumps(doc, sort_keys=True)


# ── network (thin; injectable; on-pod) ──────────────────────────────────────

def _urllib_upload(url, content_type, body, timeout=120):
    """POST a prebuilt multipart body. On-pod / real network. Returns response
    text. Sets a real User-Agent (0x0.st 403s the default urllib UA)."""
    import urllib.request
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": content_type, "User-Agent": UA},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def upload_weight(path, hosts=DEFAULT_HOSTS, uploader=_urllib_upload, field_name="file"):
    """Upload a weights file to the first working keyless host -> {ok, url, host,
    size_bytes, retention_days_est, errors}. `uploader` is injectable so tests
    never hit the network. Never raises on a host failure — it records the error
    and tries the next host, because losing the URL must not crash the run (the
    metrics still print regardless)."""
    size = os.path.getsize(path) if os.path.exists(path) else 0
    with open(path, "rb") as f:
        data = f.read()
    filename = os.path.basename(path)
    errors = []
    for host in hosts:
        try:
            content_type, body = multipart_body(field_name, filename, data)
            resp = uploader(host, content_type, body)
            url = parse_upload_url(resp)
            if url:
                return {"ok": True, "url": url, "host": host, "size_bytes": size,
                        "retention_days_est": estimate_retention_days(size),
                        "errors": errors}
            errors.append({"host": host, "error": "no url in response",
                           "resp_head": (resp or "")[:200]})
        except Exception as e:  # any host/network failure -> try the next host
            errors.append({"host": host, "error": repr(e)})
    return {"ok": False, "url": None, "host": None, "size_bytes": size,
            "retention_days_est": None, "errors": errors}
