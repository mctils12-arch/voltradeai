# Beehiiv Newsletter — Current Wiring

## What's connected right now

The newsletter signup section on the landing page is **live** as of the latest
build. It works like this:

1. User types their email in the form on `/` (landing page)
2. Clicks **Subscribe**
3. The form's `onsubmit` redirects them to:
   ```
   https://voltradeai.beehiiv.com/subscribe?email=<their-email>
   ```
4. Beehiiv's hosted subscribe page shows up with the email pre-filled
5. They confirm — Beehiiv adds them to your list and sends the welcome email
6. They can then come back to your site

This works on the **free Beehiiv tier** with **no API key required** and **no
Stripe Identity Verification needed**.

The "Read past issues →" link below the form goes to your publication
homepage at `https://voltradeai.beehiiv.com`.

### Configuration baked in

- **Publication ID**: `pub_b6b623b8-cff5-4fbd-86b0-aa7c9ab5e3e1` (saved for future API integration)
- **Subdomain**: `voltradeai.beehiiv.com`
- **Schedule shown to users**: "Sunday · 7PM ET"

If you ever change the subdomain, update these two places:
- `client/src/pages/_landing_body.html.txt` (search for `voltradeai.beehiiv.com`)
- `index.html` (standalone landing — same string)

## Important Beehiiv setup steps to take in their dashboard

These aren't code things — these are publication settings you do once on
Beehiiv's site to make subscribers actually receive the newsletter properly.

### 1. Customize the welcome / confirmation pages

When users land on `voltradeai.beehiiv.com/subscribe?email=...`, Beehiiv
shows a default page. Customize it so it looks like your brand:

1. Beehiiv dashboard → **Website**
2. Edit the **Subscribe page** styling — match the dark theme + cyan
   accents (#4d9fff)
3. Edit the **Welcome message** that subscribers see after confirming —
   include a CTA like "Try the analyzer free → voltradeai.com"

### 2. Set up the welcome email

This is the email that goes to every new subscriber automatically. It's
your single best chance to convert them from newsletter reader → product
user.

1. Beehiiv dashboard → **Automations** → **New automation**
2. Trigger: "When subscriber joins"
3. Action: send email
4. Subject: something like `Welcome — what to expect from VolTradeAI Weekly`
5. Body: brief, include:
   - What the newsletter contains (3 names + the variables)
   - When it goes out (Sunday 7PM ET)
   - **A primary CTA back to voltradeai.com** with UTM tags

### 3. (Optional) Enable double opt-in

In Beehiiv: **Settings → Audience → require email confirmation**

Why: prevents spam signups, improves deliverability.
Skip for now if you're trying to grow fast — turn on once you have ~500 subscribers.

### 4. Set the From address

Beehiiv defaults to sending from a generic address. To send from
`weekly@voltradeai.com` (much better for trust):

1. Settings → **Domains** → **Email subdomain** (you saw this earlier)
2. They'll walk you through DKIM/SPF DNS setup — takes ~10 minutes once
   you have access to your domain's DNS (probably Cloudflare or wherever
   voltradeai.com is registered)
3. After verification, set the From address in **Settings → Emails**

## How to write and send a weekly issue

1. Beehiiv dashboard → **Posts → Create new post**
2. Subject line — be specific, no clickbait. Examples:
   - "Week of Nov 17 — three names worth watching"
   - "AI semis: still the trade?"
   - "What we're watching going into Fed week"
3. Body — keep it tight. Around 400-700 words. Sections:
   - Quick market read (1-2 paragraphs)
   - Three picks with the variables we like / don't like
   - One thing to watch for the week
4. **Always include this CTA at the bottom:**
   > Want the full read on every name we watch? Open the analyzer →
   > [voltradeai.com?utm_source=newsletter&utm_medium=email](https://voltradeai.com?utm_source=newsletter&utm_medium=email)
5. Schedule for **Sunday 7:00 PM ET**

## Future upgrade — bring subscribers fully on-site

The current redirect-to-Beehiiv flow is fine but adds a page transition.
If you want users to never leave voltradeai.com when they subscribe:

1. Complete Stripe Identity Verification on Beehiiv (the verification step
   you saw — required before they'll give you an API key)
2. Beehiiv → Settings → API → create an API key (starts with `pub_xxxxx`)
3. Add to Railway: `BEEHIIV_API_KEY=...` env var
4. Tell me — I'll add a `POST /api/newsletter/subscribe` endpoint that
   takes the email, calls Beehiiv's API server-side, and shows an inline
   "Subscribed ✓" message without leaving the page.

This is a 30-minute change once you have the API key.

## Pricing

| Subscribers | Beehiiv tier | Cost |
|---|---|---|
| 0 → 2,500 | Launch | **Free** |
| 2,500 → 10,000 | Scale | $39/mo |
| 10,000+ | Grow | $99/mo+ |

You're nowhere near the free tier limit. Worry about it never.
