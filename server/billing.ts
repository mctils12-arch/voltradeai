import type { Express, Request, Response } from "express";
import express from "express";
import Stripe from "stripe";
import Database from "better-sqlite3";
import path from "path";
import { requireAuth } from "./auth";

/**
 * VolTradeAI Billing — Stripe subscription integration.
 *
 * Tier model:
 *   - free  → default for all users on signup
 *   - pro   → unlocked by an active Stripe subscription
 *
 * The owner (OWNER_EMAIL env) is implicitly Pro and bypasses Stripe entirely.
 *
 * Required environment variables (set on Railway):
 *   STRIPE_SECRET_KEY        sk_live_... or sk_test_...
 *   STRIPE_WEBHOOK_SECRET    whsec_... (set after webhook endpoint is created)
 *   STRIPE_PRICE_ID_MONTHLY  price_... (the $29/mo recurring price ID)
 *   APP_URL                  https://voltradeai.com (for checkout redirects)
 *
 * If STRIPE_SECRET_KEY is missing the module logs a warning at startup and
 * the /api/billing/* routes return 503 — useful during local dev.
 */

const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY || "";
const STRIPE_WEBHOOK_SECRET = process.env.STRIPE_WEBHOOK_SECRET || "";
const STRIPE_PRICE_ID_MONTHLY = process.env.STRIPE_PRICE_ID_MONTHLY || "";
const APP_URL = process.env.APP_URL || "https://voltradeai.com";

const dbPath = path.resolve(process.cwd(), "voltradeai.db");
const db = new Database(dbPath);

let stripe: Stripe | null = null;
if (STRIPE_SECRET_KEY) {
  stripe = new Stripe(STRIPE_SECRET_KEY, { apiVersion: "2024-12-18.acacia" as any });
  console.log("[BILLING] Stripe initialized");
} else {
  console.warn("[BILLING] STRIPE_SECRET_KEY not set — /api/billing/* will return 503");
}

/** Middleware: 503 if Stripe isn't configured. */
function requireStripe(_req: Request, res: Response, next: () => void) {
  if (!stripe) {
    return res.status(503).json({
      error: "Billing not configured",
      message: "Stripe is not yet configured on this server.",
    });
  }
  next();
}

/** Resolve current authenticated user record from session cookie. */
function getCurrentUser(req: Request): { id: number; email: string; tier: string; stripe_customer_id: string | null } | null {
  const token = (req as any).cookies?.session;
  if (!token) return null;
  const session = db.prepare("SELECT user_id FROM sessions WHERE token = ?").get(token) as any;
  if (!session) return null;
  const user = db.prepare(
    "SELECT id, email, tier, stripe_customer_id FROM users WHERE id = ?"
  ).get(session.user_id) as any;
  return user || null;
}

/**
 * Find or create the Stripe customer for a given user.
 * Caches customer ID in users.stripe_customer_id to avoid duplicate creation.
 */
async function getOrCreateStripeCustomer(userId: number, email: string): Promise<string> {
  if (!stripe) throw new Error("Stripe not initialized");

  const existing = db.prepare("SELECT stripe_customer_id FROM users WHERE id = ?").get(userId) as any;
  if (existing?.stripe_customer_id) {
    return existing.stripe_customer_id;
  }

  // Search Stripe in case a customer with this email already exists
  // (handles the case where webhook updates were missed)
  const search = await stripe.customers.search({
    query: `email:'${email.replace(/'/g, "\\'")}'`,
    limit: 1,
  });

  let customerId: string;
  if (search.data.length > 0) {
    customerId = search.data[0].id;
  } else {
    const customer = await stripe.customers.create({
      email,
      metadata: { user_id: String(userId), source: "voltradeai" },
    });
    customerId = customer.id;
  }

  db.prepare("UPDATE users SET stripe_customer_id = ? WHERE id = ?").run(customerId, userId);
  return customerId;
}

/**
 * Update a user's tier + subscription state from a Stripe Subscription object.
 * Idempotent — safe to call multiple times for the same subscription.
 */
function syncUserFromSubscription(sub: Stripe.Subscription) {
  const customerId = typeof sub.customer === "string" ? sub.customer : sub.customer.id;
  const user = db.prepare("SELECT id, email FROM users WHERE stripe_customer_id = ?").get(customerId) as any;
  if (!user) {
    console.warn(`[BILLING] Webhook: no user matched stripe_customer_id=${customerId}`);
    return;
  }

  // Active or trialing → user is Pro. Anything else (canceled, past_due, etc.) → free.
  const isActive = sub.status === "active" || sub.status === "trialing";
  const newTier = isActive ? "pro" : "free";
  const periodEnd = sub.current_period_end ? sub.current_period_end : null;

  db.prepare(`
    UPDATE users SET
      tier = ?,
      stripe_subscription_id = ?,
      subscription_status = ?,
      subscription_period_end = ?
    WHERE id = ?
  `).run(newTier, sub.id, sub.status, periodEnd, user.id);

  console.log(`[BILLING] User ${user.email}: tier=${newTier}, status=${sub.status}, sub=${sub.id}`);
}

// ─── Public API ─────────────────────────────────────────────────────────────

export function registerBillingRoutes(app: Express) {
  /**
   * Stripe webhook endpoint.
   * MUST be registered BEFORE the global JSON body parser, because Stripe
   * requires the raw request body for signature verification.
   *
   * In server/index.ts, this function is called BEFORE app.use(express.json())
   * to ensure the raw-body middleware below takes effect.
   */
  app.post(
    "/api/billing/webhook",
    express.raw({ type: "application/json" }),
    async (req: Request, res: Response) => {
      if (!stripe || !STRIPE_WEBHOOK_SECRET) {
        return res.status(503).json({ error: "Webhook not configured" });
      }

      const sig = req.headers["stripe-signature"];
      if (!sig) return res.status(400).json({ error: "Missing stripe-signature" });

      let event: Stripe.Event;
      try {
        event = stripe.webhooks.constructEvent(req.body, sig as string, STRIPE_WEBHOOK_SECRET);
      } catch (err: any) {
        console.error("[BILLING] Webhook signature verification failed:", err.message);
        return res.status(400).json({ error: `Webhook Error: ${err.message}` });
      }

      try {
        switch (event.type) {
          case "checkout.session.completed": {
            const session = event.data.object as Stripe.Checkout.Session;
            const subscriptionId = session.subscription as string | null;
            const customerId = session.customer as string | null;
            if (subscriptionId && customerId && stripe) {
              const sub = await stripe.subscriptions.retrieve(subscriptionId);
              syncUserFromSubscription(sub);
            }
            break;
          }

          case "customer.subscription.created":
          case "customer.subscription.updated":
          case "customer.subscription.deleted":
          case "customer.subscription.trial_will_end": {
            const sub = event.data.object as Stripe.Subscription;
            syncUserFromSubscription(sub);
            break;
          }

          case "invoice.payment_failed": {
            const invoice = event.data.object as Stripe.Invoice;
            // Stripe will retry; subscription will move to past_due via subscription.updated.
            console.log(`[BILLING] Payment failed for invoice ${invoice.id}`);
            break;
          }

          default:
            // Other event types we don't care about — just ack
            break;
        }
        res.json({ received: true });
      } catch (err: any) {
        console.error("[BILLING] Webhook handler error:", err);
        res.status(500).json({ error: "Webhook handler error", message: err.message });
      }
    }
  );

  // From here on, JSON body is parsed normally (parser is global, registered in server/index.ts).

  /**
   * POST /api/billing/checkout — create a Stripe Checkout session for the
   * authenticated user and return its URL. Frontend redirects to it.
   */
  app.post("/api/billing/checkout", requireAuth, requireStripe, async (req: Request, res: Response) => {
    try {
      if (!STRIPE_PRICE_ID_MONTHLY) {
        return res.status(500).json({ error: "STRIPE_PRICE_ID_MONTHLY not configured" });
      }
      const user = getCurrentUser(req);
      if (!user) return res.status(401).json({ error: "Not authenticated" });

      // Already Pro? Send them to the customer portal instead.
      if (user.tier === "pro") {
        return res.status(400).json({
          error: "Already Pro",
          message: "You already have an active Pro subscription.",
        });
      }

      const customerId = await getOrCreateStripeCustomer(user.id, user.email);

      const session = await stripe!.checkout.sessions.create({
        mode: "subscription",
        customer: customerId,
        line_items: [{ price: STRIPE_PRICE_ID_MONTHLY, quantity: 1 }],
        success_url: `${APP_URL}/app?checkout=success&session_id={CHECKOUT_SESSION_ID}`,
        cancel_url: `${APP_URL}/pricing?checkout=cancel`,
        allow_promotion_codes: true,
        billing_address_collection: "auto",
        // No trial — per product decision (charge on day one).
        metadata: { user_id: String(user.id), email: user.email },
      });

      res.json({ url: session.url });
    } catch (err: any) {
      console.error("[BILLING] Checkout error:", err);
      res.status(500).json({ error: "Checkout failed", message: err.message });
    }
  });

  /**
   * POST /api/billing/portal — open the Stripe Customer Portal for the
   * authenticated user (manage card, cancel, view invoices).
   */
  app.post("/api/billing/portal", requireAuth, requireStripe, async (req: Request, res: Response) => {
    try {
      const user = getCurrentUser(req);
      if (!user) return res.status(401).json({ error: "Not authenticated" });
      if (!user.stripe_customer_id) {
        return res.status(400).json({
          error: "No subscription",
          message: "No Stripe customer record yet — sign up for Pro first.",
        });
      }

      const portal = await stripe!.billingPortal.sessions.create({
        customer: user.stripe_customer_id,
        return_url: `${APP_URL}/app`,
      });

      res.json({ url: portal.url });
    } catch (err: any) {
      console.error("[BILLING] Portal error:", err);
      res.status(500).json({ error: "Portal session failed", message: err.message });
    }
  });

  /**
   * GET /api/billing/status — return the user's current tier + subscription state.
   * Frontend uses this to decide whether to show "Upgrade" or "Manage subscription".
   */
  app.get("/api/billing/status", requireAuth, (req: Request, res: Response) => {
    const user = getCurrentUser(req);
    if (!user) return res.status(401).json({ error: "Not authenticated" });

    const fullUser = db.prepare(
      "SELECT email, tier, subscription_status, subscription_period_end, stripe_subscription_id FROM users WHERE id = ?"
    ).get(user.id) as any;

    res.json({
      email: fullUser.email,
      tier: fullUser.tier || "free",
      subscriptionStatus: fullUser.subscription_status || null,
      subscriptionPeriodEnd: fullUser.subscription_period_end || null,
      hasStripeCustomer: !!user.stripe_customer_id,
      stripeConfigured: !!stripe,
    });
  });

  /**
   * POST /api/admin/users/:email/tier  — manual tier override (owner only).
   * Use to grant Pro to specific users without Stripe (e.g. early access friends).
   * Body: { tier: "free" | "pro" }
   */
  app.post("/api/admin/users/:email/tier", requireAuth, async (req: Request, res: Response) => {
    const requester = getCurrentUser(req);
    const OWNER_EMAIL = process.env.OWNER_EMAIL || "mctils12@gmail.com";
    if (!requester || requester.email !== OWNER_EMAIL) {
      return res.status(403).json({ error: "Owner only" });
    }
    const { tier } = req.body as { tier?: string };
    if (tier !== "free" && tier !== "pro") {
      return res.status(400).json({ error: "tier must be 'free' or 'pro'" });
    }
    const targetEmail = req.params.email.toLowerCase();
    const target = db.prepare("SELECT id FROM users WHERE LOWER(email) = ?").get(targetEmail) as any;
    if (!target) return res.status(404).json({ error: "User not found" });

    db.prepare("UPDATE users SET tier = ?, subscription_status = ? WHERE id = ?").run(
      tier,
      tier === "pro" ? "manual_grant" : null,
      target.id
    );
    console.log(`[BILLING] Owner manually set ${targetEmail} → tier=${tier}`);
    res.json({ ok: true, email: targetEmail, tier });
  });
}
