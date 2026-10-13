/**
 * The From address on every email the site sends through Resend (welcome
 * and password-reset mail in auth.ts, owner alerts in bot.ts).
 *
 * voltradeai.com was verified in Resend on 2026-10-13 (human-directed). The
 * previous sender, onboarding@resend.dev, is Resend's TEST address: it only
 * delivers to the Resend account owner, so users never received welcome or
 * password-reset mail. noreply@ is send-only; the site's public contact
 * address is hello@voltradeai.com.
 */
export const EMAIL_FROM = "VolTradeAI <noreply@voltradeai.com>";
