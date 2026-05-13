# Newsletter Content

Markdown-backed newsletter archive. Each `.md` file in this folder becomes a
public issue page on the site at `voltradeai.com/newsletter/<slug>`.

## How to write a new issue

1. **Create a new file** in this folder with the naming convention:
   ```
   YYYY-MM-DD-short-slug.md
   ```
   Example: `2026-12-08-fed-week-and-the-vol-surface.md`

2. **Frontmatter** at the top of the file:
   ```markdown
   ---
   title: "Fed week and what the vol surface is saying"
   subtitle: "Optional — shows below the title on the issue page"
   slug: "fed-week-vol-surface"    # OPTIONAL — auto-derived from filename if missing
   date: "2026-12-08"              # OPTIONAL — auto-derived from filename if missing
   excerpt: "What to watch this week. 1-2 sentences."
   hero: "/newsletter/images/fed-week/hero.png"   # OPTIONAL — large image at top
   ---
   ```

3. **Body** is standard Markdown. Headings, lists, links, images, tables,
   bold/italic — all work.

4. **Images** for each issue go in `client/public/newsletter/images/<slug>/`.
   Reference them with absolute paths starting with `/newsletter/images/...`
   so they work both on the site and in Beehiiv.

5. **Save the file** and commit to GitHub. Railway deploys. The issue is live at:
   ```
   https://voltradeai.com/newsletter/<slug>
   ```

## Workflow for emailing the issue

The site is now the canonical archive — and Beehiiv handles delivery.

1. Write the issue here. Push.
2. After it's live on the site, open `https://voltradeai.com/newsletter/<slug>`
3. In Beehiiv: Posts → Create new post → paste the markdown into the editor
   (Beehiiv supports markdown), OR use the issue's HTML if you want exact
   styling.
4. At the bottom of the email, add a permanent link:
   "Read this issue on the site: https://voltradeai.com/newsletter/<slug>"
5. Send.

## Naming guidelines

- Filename slug: lowercase, dash-separated, descriptive but short
- ✅ `2026-12-08-fed-week-and-vol-surface.md`
- ✅ `2026-11-17-welcome-to-voltradeai.md`
- ❌ `Newsletter Issue 5.md` (uppercase, spaces)
- ❌ `2026.md` (no slug)

## Date handling

- The `YYYY-MM-DD` prefix in the filename is the source of truth for the
  publication date AND the sort order. Latest first.
- You CAN override with `date:` in frontmatter, but it's cleaner to just
  name the file correctly.
