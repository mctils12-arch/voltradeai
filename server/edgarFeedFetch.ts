/**
 * The one request for an EDGAR "getcurrent" Atom feed page, shared by the
 * Form 4, 13F and 8-K pollers. It retries a slow or dropped request.
 *
 * WHY ([REPAIR] 2026-11-19, research/experiments.md): each poller asked for
 * its feed page once with a 12 s limit, and a timeout lost the whole poll for
 * 15 minutes. Production 2026-11-19: about 20 failed polls of 78 on each of
 * the three streams; 2026-11-20 02:34Z: Form 4 at 2 failed of 6, last error
 * "The operation was aborted due to timeout". Measured from a session the
 * same hour (61 requests to the three feed URLs, sequential): 46 answered in
 * under 3.4 s, 15 took 6.5 to 21.2 s, and 2 were over 12 s. In the longest
 * run (45 requests) 9 of the 10 slow ones were followed by a fast one. The page size
 * made no difference (10 entries took 8.1 s, 40 took 0.3 s). So the slowness
 * belongs to the single request, and asking again is the fix.
 *
 * What is retried: a request that threw (timeout, reset connection) and an
 * HTTP 5xx. What is not: any other status. A 400 is EDGAR refusing the URL
 * (2026-11-16, an unsupported page size) and a 403 or 429 is EDGAR's
 * fair-access limit; asking again changes neither and the second would be
 * rude. Worst case is three requests in about a minute, once per 15-minute
 * poll, against a fair-access ceiling of 10 requests a second.
 *
 * Later attempts wait longer than the first: production's tail is not
 * measured (it loses more polls than the session's 2 in 61 predicts), so an
 * attempt that follows a timeout must not give up at the same point.
 */
import { noteFeedAttemptFailed, noteFeedRecovered, type EdgarPollStream } from "./edgarPollHealth";

export interface EdgarFeedRequestInit { headers: Record<string, string>; signal: AbortSignal }
export interface EdgarFeedResponse { ok: boolean; status: number; text(): Promise<string> }
export type EdgarFeedFetchFn = (url: string, init: EdgarFeedRequestInit) => Promise<EdgarFeedResponse>;

/** Time limit of each attempt, in order. Its length is the number of attempts. */
export const EDGAR_FEED_ATTEMPT_TIMEOUTS_MS: readonly number[] = [12_000, 20_000, 30_000];
/** Wait between a failed attempt and the next. */
export const EDGAR_FEED_RETRY_PAUSE_MS = 500;

export interface EdgarFeedFetchOptions {
  timeoutsMs?: readonly number[];
  pauseMs?: number;
}

/** True for a failure worth a second request: no status at all (the request
 *  threw) or a server-side 5xx. */
export function edgarFeedFailureIsRetryable(status: number | null): boolean {
  return status === null || (status >= 500 && status <= 599);
}

/** The feed page's text. Throws the last attempt's error when every attempt
 *  failed, or at once on a status that a retry cannot change. */
export async function fetchEdgarFeed(
  fetchImpl: EdgarFeedFetchFn,
  url: string,
  headers: Record<string, string>,
  stream: EdgarPollStream,
  opts: EdgarFeedFetchOptions = {},
): Promise<string> {
  const timeouts = opts.timeoutsMs && opts.timeoutsMs.length ? opts.timeoutsMs : EDGAR_FEED_ATTEMPT_TIMEOUTS_MS;
  const pauseMs = opts.pauseMs ?? EDGAR_FEED_RETRY_PAUSE_MS;
  let lastError: unknown = new Error(`${url} -> no attempt made`);
  for (let attempt = 0; attempt < timeouts.length; attempt++) {
    let status: number | null = null;
    try {
      const r = await fetchImpl(url, { headers, signal: AbortSignal.timeout(timeouts[attempt]) });
      status = r.status;
      if (r.ok) {
        const text = await r.text();
        if (attempt > 0) noteFeedRecovered(stream);
        return text;
      }
      lastError = new Error(`${url} -> ${r.status}`);
    } catch (e) {
      status = null; // a body that failed mid-read is a dropped request, whatever its status line said
      lastError = e;
    }
    if (!edgarFeedFailureIsRetryable(status)) throw lastError;
    noteFeedAttemptFailed(stream, lastError);
    if (attempt < timeouts.length - 1 && pauseMs > 0) await new Promise((res) => setTimeout(res, pauseMs));
  }
  throw lastError;
}
