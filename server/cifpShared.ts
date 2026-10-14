// ONE FAA CIFP store per process (2026-10-14). The procedures API and the
// flight-plan route parser (SID/STAR expansion) read the same 28-day cycle;
// two stores would download and index the 9 MB file twice. The NASR
// gazetteer is attached as the independent fallback/cross-check, exactly as
// procedures.ts always configured it.
import { CifpStore } from "./cifp";
import { lookupFix } from "./navFixes";

let shared: CifpStore | null = null;

export function sharedCifpStore(): CifpStore {
  if (!shared) shared = new CifpStore({ external: (id) => lookupFix(id) });
  return shared;
}
