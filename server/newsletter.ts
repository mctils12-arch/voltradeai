import type { Express, Request, Response } from "express";
import fs from "fs";
import path from "path";
import { marked } from "marked";
import matter from "gray-matter";

/**
 * VolTradeAI Newsletter — markdown-backed archive.
 *
 * Each issue is a markdown file in /content/newsletter/ with frontmatter:
 *
 *   ---
 *   title: "Welcome to VolTradeAI"
 *   subtitle: "A year of semiconductors..."
 *   slug: "welcome-to-voltradeai"   # used in URL; auto-derived from filename if missing
 *   date: "2026-11-17"
 *   excerpt: "What's actually moving the market..."
 *   hero: "/newsletter/images/welcome/hero.png"   # optional
 *   ---
 *
 *   ## Markdown body here
 *
 * Public endpoints:
 *   GET /api/newsletter/issues          — list (metadata only, no body)
 *   GET /api/newsletter/issues/:slug    — full content (rendered HTML)
 *
 * The content directory is resolved from process.cwd() at runtime, so it
 * works both during local dev (CWD = repo root) and in production
 * (Dockerfile copies content/ to /app/content/).
 */

const CONTENT_DIR = path.resolve(process.cwd(), "content", "newsletter");

interface IssueMeta {
  slug: string;
  title: string;
  subtitle?: string;
  date: string;
  excerpt?: string;
  hero?: string;
}

interface IssueFull extends IssueMeta {
  html: string;        // rendered markdown
  markdown: string;    // raw source
}

/** Configure marked once at module load. */
marked.setOptions({
  gfm: true,
  breaks: false,       // require explicit \n\n for paragraph breaks
});

/**
 * Read and parse a single markdown file. Returns null if the file
 * doesn't exist or is malformed (logged, not thrown — one bad file
 * shouldn't break the whole archive).
 */
function loadIssue(filename: string): IssueFull | null {
  try {
    const filePath = path.join(CONTENT_DIR, filename);
    const raw = fs.readFileSync(filePath, "utf-8");
    const { data, content } = matter(raw);

    // Auto-derive slug from filename if frontmatter doesn't specify
    // Filename format: "2026-11-17-welcome-to-voltradeai.md"
    // Default slug: "welcome-to-voltradeai" (strip date prefix + extension)
    const defaultSlug = filename
      .replace(/^\d{4}-\d{2}-\d{2}-/, "")
      .replace(/\.md$/, "");
    const slug = data.slug || defaultSlug;

    // Auto-derive date from filename if frontmatter doesn't specify
    const dateMatch = filename.match(/^(\d{4}-\d{2}-\d{2})/);
    const date = data.date || (dateMatch ? dateMatch[1] : "");

    if (!data.title) {
      console.warn(`[NEWSLETTER] ${filename} missing 'title' in frontmatter`);
      return null;
    }

    const html = marked.parse(content) as string;

    return {
      slug,
      title: data.title,
      subtitle: data.subtitle,
      date,
      excerpt: data.excerpt,
      hero: data.hero,
      html,
      markdown: content,
    };
  } catch (err) {
    console.error(`[NEWSLETTER] Failed to load ${filename}:`, err);
    return null;
  }
}

/** Get all valid issues sorted newest-first. */
function loadAllIssues(): IssueFull[] {
  if (!fs.existsSync(CONTENT_DIR)) {
    console.warn(`[NEWSLETTER] Content directory missing: ${CONTENT_DIR}`);
    return [];
  }
  const files = fs.readdirSync(CONTENT_DIR)
    .filter((f) => f.endsWith(".md"))
    .sort()       // ascending by filename (which starts with date)
    .reverse();   // newest first

  const issues: IssueFull[] = [];
  for (const f of files) {
    const issue = loadIssue(f);
    if (issue) issues.push(issue);
  }
  return issues;
}

export function registerNewsletterRoutes(app: Express) {
  /**
   * GET /api/newsletter/issues — list metadata only.
   * Used by the archive grid on the /newsletter page.
   */
  app.get("/api/newsletter/issues", (_req: Request, res: Response) => {
    try {
      const issues = loadAllIssues();
      const meta: IssueMeta[] = issues.map((i) => ({
        slug: i.slug,
        title: i.title,
        subtitle: i.subtitle,
        date: i.date,
        excerpt: i.excerpt,
        hero: i.hero,
      }));
      // Light cache — issues change infrequently
      res.setHeader("Cache-Control", "public, max-age=60");
      res.json({ issues: meta });
    } catch (err: any) {
      console.error("[NEWSLETTER] List error:", err);
      res.status(500).json({ error: "Failed to load issues" });
    }
  });

  /**
   * GET /api/newsletter/issues/:slug — full issue content.
   * Used by the /newsletter/:slug page.
   */
  app.get("/api/newsletter/issues/:slug", (req: Request, res: Response) => {
    try {
      const { slug } = req.params;
      const issues = loadAllIssues();
      const issue = issues.find((i) => i.slug === slug);
      if (!issue) {
        return res.status(404).json({ error: "Issue not found" });
      }
      res.setHeader("Cache-Control", "public, max-age=60");
      res.json(issue);
    } catch (err: any) {
      console.error("[NEWSLETTER] Get error:", err);
      res.status(500).json({ error: "Failed to load issue" });
    }
  });
}
