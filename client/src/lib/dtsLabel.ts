// dtsLabel — how the Treasury Daily Statement page words a line's category.
//
// FiscalData publishes the literal STRING "null" as the category of its
// roll-up rows ("Treasury General Account Total Deposits" and the like). The
// archive keeps the field exactly as published (server/treasuryDts.ts), so the
// page was printing the word "null" in its Category column (found on
// production by the 2026-11-19 site sweep, at all three widths). The account
// column already says what a roll-up row is; its category is simply absent.

/** A category as the page shows it: the published text, or a dash when the
 *  statement published none (null, empty, or FiscalData's literal "null"). */
export function dtsCategoryLabel(category: string | null | undefined): string {
  if (category == null) return "—";
  const c = String(category).trim();
  return c === "" || c.toLowerCase() === "null" ? "—" : c;
}
