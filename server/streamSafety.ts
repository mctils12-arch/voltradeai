/**
 * streamSafety.ts — feed a fetch() response body into a Node writable without
 * letting a network failure become an uncaught exception (2026-11-05 REPAIR).
 *
 * undici errors a response body stream with TypeError("terminated") when the
 * connection dies mid-body (socket reset, its 300 s body timeout). A body fed
 * through `.pipe()` has no 'error' listener on its SOURCE, so that error was
 * an uncaught exception and the FATAL handler restarted the whole container
 * (live: 2026-11-04 05:03Z and 2026-11-05 01:17Z). stream.pipeline listens on
 * every stage and tears all of them down when any one fails or the
 * destination closes early, so a visitor who leaves mid-tile also cancels the
 * upstream body at once instead of leaving it to stall into that timeout.
 */
import { Readable, pipeline } from "node:stream";

/** Streams `body` into `dest` (an HTTP response, a file). Never throws and never
 *  leaves an error unlistened: `onDone` hears success (null) or the failure. */
export function pipeWebBody(
  body: ReadableStream<Uint8Array>,
  dest: NodeJS.WritableStream,
  onDone: (err: Error | null) => void = () => {},
): void {
  let src: Readable;
  try {
    src = Readable.fromWeb(body as any);
  } catch (e) {
    (dest as any).destroy?.();
    onDone(e as Error);
    return;
  }
  pipeline(src, dest, (err) => onDone(err ?? null));
}
