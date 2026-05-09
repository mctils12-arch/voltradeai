# Stripe Billing Setup

This walkthrough gets your $30/mo Pro tier live on production.

## 0. What's already wired

- ✅ Database has `tier`, `stripe_customer_id`, `stripe_subscription_id`,
      `subscription_status`, `subscription_period_end` columns on `users`
- ✅ Server module `server/billing.ts` exposes:
  - `POST /api/billing/checkout` — start Stripe Checkout
  - `POST /api/billing/portal` — open customer self-service portal
  - `GET /api/billing/status` — current tier/subscription
  - `POST /api/billing/webhook` — receives Stripe events
  - `POST /api/admin/users/:email/tier` — manual override (owner only)
- ✅ React `/pricing` page with working Free + Pro tier cards
- ✅ Sign Up button on the landing routes to `/login` then to `/pricing`
- ✅ Pro middleware (`requirePro`) ready for gating future Pro features
- ✅ Bot routes locked to owner only via `requireOwner`
- ✅ `/api/auth/me` returns `tier` for frontend gating
- ✅ Webhook registered before `express.json()` so signature verification works
- ✅ Stripe SDK added to `package.json` (run `npm install` after pulling)

## 1. Create the Stripe product

1. Sign in at <https://dashboard.stripe.com>
2. **Stay in TEST MODE** (toggle in top-right) until everything works end-to-end
3. Left sidebar → **Product catalog** → **Add product**
4. Name: `VolTradeAI Pro`
5. Description: `Full decision intelligence — every variable on every trade`
6. **Pricing**:
   - Type: **Recurring**
   - Amount: `$30.00`
   - Billing period: **Monthly**
   - Tax behavior: leave default
7. Click **Add product**
8. After creation, click into the product → copy the **Price ID** (looks like `price_1AbCdEfGh...`)

## 2. Get your API keys

1. Left sidebar → **Developers → API keys**
2. Copy the **Secret key** (test mode: starts `sk_test_...`)
3. Save it — you'll paste it into Railway in step 4

## 3. Create the webhook endpoint (do this AFTER deploy)

You need a deployed URL before this step works.

1. Deploy code to Railway (steps below in section 5)
2. Once deployed, your URL is something like `https://voltradeai-production.up.railway.app` (or your custom domain)
3. Stripe dashboard → **Developers → Webhooks → Add endpoint**
4. Endpoint URL: `https://YOUR_DEPLOYED_URL/api/billing/webhook`
5. **Events to send**: select these (and only these):
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `customer.subscription.trial_will_end`
   - `invoice.payment_failed`
6. Click **Add endpoint**
7. After creation, click into the endpoint → reveal the **Signing secret** (starts `whsec_...`)
8. Save it — paste into Railway

## 4. Set Railway environment variables

Open Railway → your project → **Variables** tab → add:

| Variable                   | Value                                |
|----------------------------|--------------------------------------|
| `STRIPE_SECRET_KEY`        | `sk_test_...` (or `sk_live_...`)     |
| `STRIPE_PRICE_ID_MONTHLY`  | `price_1AbCdEf...` from step 1       |
| `STRIPE_PRICE_ID_ANNUAL`   | `price_1AbCdEf...` (annual price ID) |
| `STRIPE_WEBHOOK_SECRET`    | `whsec_...` from step 3              |
| `APP_URL`                  | `https://voltradeai.com`             |
| `OWNER_EMAIL`              | (already set: `mctils12@gmail.com`)  |

Railway redeploys automatically when env vars change. Wait for the redeploy to finish (~2 min) before testing.

## 5. Deploy

```bash
cd voltradeai-main
npm install     # picks up the new `stripe` dependency
npm run build
git add .
git commit -m "Wire Stripe billing"
git push
```

Railway picks up the push and deploys.

## 6. Test in Stripe TEST MODE

Stripe provides test card numbers that simulate real cards. Use any of these:

| Card                | Behavior                       |
|---------------------|--------------------------------|
| `4242 4242 4242 4242` | Successful payment              |
| `4000 0000 0000 0002` | Declined card                  |
| `4000 0027 6000 3184` | 3D Secure auth required        |

Any future expiry, any CVC, any ZIP.

Walk through the flow:

1. Open `https://voltradeai.com` (your deployed URL) in a fresh incognito window
2. Click **Sign up** on the landing → register a new account
3. Visit `/pricing` → click **Upgrade to Pro**
4. Stripe checkout opens → use card `4242 4242 4242 4242`
5. After payment, you'll be redirected back to `/app?checkout=success`
6. The webhook fires (visible in Stripe Dashboard → Developers → Webhooks → your endpoint → Recent events) — should show `checkout.session.completed` with status 200
7. Refresh `/api/auth/me` (or just reload the app) — `tier` should now be `"pro"`
8. Visit `/pricing` — the Pro card should show "Manage subscription" instead of "Upgrade"

## 7. Switch to LIVE MODE

Once everything works in test mode:

1. Stripe dashboard → top-right toggle → **switch to Live mode**
2. **Repeat steps 1, 2, 3 in live mode** to create a live product, live API key, and live webhook endpoint
3. Update Railway env vars: replace test keys with live keys (`sk_live_...`, `price_...` from live mode product, `whsec_...` from live webhook)
4. Wait for Railway redeploy
5. Test once with a real card (your own) — you can refund it from the Stripe dashboard immediately after

## Troubleshooting

**"Webhook signature verification failed"**
- Webhook secret in Railway env doesn't match the one in Stripe dashboard
- Solution: re-copy the secret from Stripe → Webhooks → your endpoint → Signing secret

**"STRIPE_PRICE_ID_MONTHLY not configured"**
- Env var missing or has wrong name
- Solution: check Railway → Variables, restart deployment

**Webhook fires but tier doesn't update**
- The webhook arrived before the user's `stripe_customer_id` was saved
- Solution: this should self-heal on next subscription update event. If not, manually grant Pro:
  ```
  curl -X POST https://voltradeai.com/api/admin/users/USER_EMAIL/tier \
       -H "Content-Type: application/json" \
       -H "Cookie: session=YOUR_OWNER_SESSION_TOKEN" \
       -d '{"tier":"pro"}'
  ```

**Customer can't access Pro features after paying**
- Frontend may be caching `/api/auth/me` — invalidate the React Query cache or force a logout/login
- Verify in DB: `SELECT tier, subscription_status FROM users WHERE email = ?`

## What's locked to the owner

The bot/trading engine is **owner-only** for now (you = `OWNER_EMAIL`).
This is enforced by `requireOwner` middleware on every `/api/bot/*` and
`/api/system/*` route. The "AI Engine" tab is hidden in the UI for
non-owners. This is intentional — auto-trading customer money requires RIA
registration which we have not completed.

When you eventually register as an RIA and want to open the bot to Pro
customers, swap `requireOwner` → `requirePro` on the relevant routes and
remove `requiresOwner: true` from the AI Engine tab in `home.tsx`.
