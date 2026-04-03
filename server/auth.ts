import { Express, Request, Response, NextFunction } from "express";
import crypto from "crypto";

// ─── Config ─────────────────────────────────────────────────────────────────
const OWNER_EMAIL = process.env.OWNER_EMAIL || "mctils12@gmail.com";
const SESSION_SECRET = process.env.SESSION_SECRET || crypto.randomBytes(32).toString("hex");

// Default password: "voltrade2026" — change via reset
let ownerPasswordHash = process.env.OWNER_PASSWORD_HASH || "";

function hashPw(pw: string): string {
  return crypto.createHash("sha256").update(pw + SESSION_SECRET).digest("hex");
}

// Set default hash on first load
if (!ownerPasswordHash) ownerPasswordHash = hashPw("voltrade2026");

// ─── Sessions (in-memory, single user) ──────────────────────────────────────
const sessions = new Map<string, { email: string; created: number }>();
const resetCodes = new Map<string, { code: string; expires: number }>();

export function requireAuth(req: Request, res: Response, next: NextFunction) {
  const sid = (req as any).cookies?.sid;
  if (!sid || !sessions.has(sid)) return res.status(401).json({ error: "Not authenticated" });
  const s = sessions.get(sid)!;
  if (Date.now() - s.created > 24 * 3600 * 1000) {
    sessions.delete(sid);
    return res.status(401).json({ error: "Session expired" });
  }
  next();
}

export function registerAuthRoutes(app: Express) {
  app.post("/api/auth/login", (req, res) => {
    const { email, password } = req.body || {};
    if (!email || !password) return res.status(400).json({ error: "Email and password required" });
    if (email.toLowerCase() !== OWNER_EMAIL.toLowerCase()) return res.status(401).json({ error: "Invalid credentials" });
    if (hashPw(password) !== ownerPasswordHash) return res.status(401).json({ error: "Invalid credentials" });

    const sid = crypto.randomBytes(32).toString("hex");
    sessions.set(sid, { email, created: Date.now() });
    res.cookie("sid", sid, { httpOnly: true, secure: process.env.NODE_ENV === "production", sameSite: "lax", maxAge: 86400000 });
    res.json({ ok: true, email });
  });

  app.post("/api/auth/logout", (req, res) => {
    const sid = (req as any).cookies?.sid;
    if (sid) sessions.delete(sid);
    res.clearCookie("sid");
    res.json({ ok: true });
  });

  app.get("/api/auth/me", (req, res) => {
    const sid = (req as any).cookies?.sid;
    if (!sid || !sessions.has(sid)) return res.json({ authenticated: false });
    const s = sessions.get(sid)!;
    if (Date.now() - s.created > 86400000) { sessions.delete(sid); return res.json({ authenticated: false }); }
    res.json({ authenticated: true, email: s.email });
  });

  app.post("/api/auth/reset-request", (req, res) => {
    const { email } = req.body || {};
    if (email?.toLowerCase() === OWNER_EMAIL.toLowerCase()) {
      const code = Math.random().toString(36).substring(2, 8).toUpperCase();
      resetCodes.set(email.toLowerCase(), { code, expires: Date.now() + 900000 });
      console.log(`[AUTH] Reset code for ${email}: ${code}`);
    }
    res.json({ ok: true });
  });

  app.post("/api/auth/reset-confirm", (req, res) => {
    const { email, code, newPassword } = req.body || {};
    const entry = resetCodes.get(email?.toLowerCase());
    if (!entry || entry.code !== code || Date.now() > entry.expires) {
      return res.status(400).json({ error: "Invalid or expired code" });
    }
    ownerPasswordHash = hashPw(newPassword);
    resetCodes.delete(email.toLowerCase());
    res.json({ ok: true });
  });
}
