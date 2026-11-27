/**
 * xmlText.ts — the text value of an XML element, as the document meant it.
 *
 * WHY (2026-11-27, open_questions.md (ph7)). The EDGAR parsers in this
 * directory read XML with regular expressions, which return an element's
 * content exactly as it sits in the file: "DEERE &amp; CO", "President &amp;
 * CEO". Unescaping is the last step of reading an XML text node, and the 8-K
 * and Form 4 parsers skipped it, so the archive, the API, the CSV and the page
 * all carried the escaped form (production, 30 days to 2026-11-27: 12 earnings
 * rows and 265 strings on 30 days of insider filings).
 *
 * WHAT IT DOES. One pass over the string, replacing the five entities XML
 * predefines (amp, lt, gt, quot, apos) and numeric character references
 * (decimal and hex). One pass matters: "&amp;lt;" is the text "&lt;", not "<".
 * Anything else that looks like an entity ("&T;", "&nbsp;") is not XML's to
 * decode and is left as written.
 *
 * READING OLD ROWS. Rows archived before this module hold the escaped form.
 * Archives are append-only and are not rewritten; the readers apply this
 * function to the name fields of each row they return. On a row written after
 * the fix that is a second pass over clean text, which changes nothing unless
 * the filer's own text contains a literal escaped entity (they wrote
 * "&amp;amp;"); in that one case the reader shows "&" where the filer's text,
 * read strictly, says "&amp;".
 */

const PREDEFINED: Record<string, string> = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };

const ENTITY = /&(#[xX][0-9a-fA-F]{1,6}|#[0-9]{1,7}|amp|lt|gt|quot|apos);/g;

/** A code point XML 1.0 allows in character data. */
function allowedCodePoint(cp: number): boolean {
  if (cp === 0x9 || cp === 0xa || cp === 0xd) return true;
  if (cp >= 0x20 && cp <= 0xd7ff) return true;
  if (cp >= 0xe000 && cp <= 0xfffd) return true;
  return cp >= 0x10000 && cp <= 0x10ffff;
}

/** Unescapes XML character data once. A reference XML does not define, or one
 *  to a code point XML forbids, is returned as written. */
export function unescapeXmlText(s: string): string {
  if (s.indexOf("&") < 0) return s;
  return s.replace(ENTITY, (whole, body: string) => {
    if (body[0] !== "#") return PREDEFINED[body];
    const hex = body[1] === "x" || body[1] === "X";
    const cp = parseInt(body.slice(hex ? 2 : 1), hex ? 16 : 10);
    return Number.isFinite(cp) && allowedCodePoint(cp) ? String.fromCodePoint(cp) : whole;
  });
}

/** The same for a field that may be absent; null and undefined pass through. */
export function unescapeXmlTextOrNull<T extends string | null | undefined>(s: T): T {
  return (typeof s === "string" ? unescapeXmlText(s) : s) as T;
}
