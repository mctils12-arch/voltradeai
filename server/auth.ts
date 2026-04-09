import { Express, Request, Response, NextFunction } from "express";
import crypto from "crypto";
import Database from "better-sqlite3";
import bcrypt from "bcryptjs";
import path from "path";
import fs from "fs";

// ─── Config ─────────────────────────────────────────────────────────────────
const OWNER_EMAIL = process.env.OWNER_EMAIL || "mctils12@gmail.com";
// Store database on persistent volume (/data) so it survives deploys
const DATA_DIR = fs.existsSync("/data") ? "/data" : process.cwd();
const DB_PATH = process.env.DB_PATH || path.resolve(DATA_DIR, "voltrade.db");

// ─── Database setup ──────────────────────────────────────────────────────────
export const db = new Database(DB_PATH);

// Create tables if they don't exist
db.prepare(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at TEXT NOT NULL
  )
`).run();

// Ensure role column exists (in case of older DB)
try {
  db.prepare("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'").run();
} catch {} // Column already exists

db.prepare(`
  CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at INTEGER NOT NULL
  )
`).run();

db.prepare(`
  CREATE TABLE IF NOT EXISTS password_resets (
    email TEXT,
    token TEXT,
    expires TEXT
  )
`).run();

// ─── Owner bootstrap ─────────────────────────────────────────────────────────
// Set owner role for the main account (if it exists)
db.prepare("UPDATE users SET role = 'owner' WHERE email = ?").run(OWNER_EMAIL);

// If owner account doesn't exist yet, create it with the default password
const ownerExists = db.prepare("SELECT id FROM users WHERE email = ?").get(OWNER_EMAIL);
if (!ownerExists) {
  const defaultPw = process.env.OWNER_DEFAULT_PASSWORD || crypto.randomBytes(16).toString("hex");
  const defaultHash = bcrypt.hashSync(defaultPw, 10);
  db.prepare("INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, ?, ?)").run(
    OWNER_EMAIL, defaultHash, "owner", new Date().toISOString()
  );
  console.log(`[AUTH] Owner account created for ${OWNER_EMAIL} — bootstrap key: ${defaultPw.slice(0, 8)}...`);
}

// ─── Auth middleware ─────────────────────────────────────────────────────────
export function requireAuth(req: Request, res: Response, next: NextFunction) {
  const token = (req as any).cookies?.session;
  if (!token) return res.status(401).json({ error: "Not authenticated" });

  const session = db.prepare("SELECT user_id, created_at FROM sessions WHERE token = ?").get(token) as any;
  if (!session) return res.status(401).json({ error: "Not authenticated" });

  // 24-hour session expiry
  if (Date.now() - session.created_at > 24 * 3600 * 1000) {
    db.prepare("DELETE FROM sessions WHERE token = ?").run(token);
    return res.status(401).json({ error: "Session expired" });
  }

  next();
}

// ─── Route registration ──────────────────────────────────────────────────────

// ─── Login Rate Limiting ────────────────────────────────────────────────────
const loginAttempts = new Map<string, { count: number; lockedUntil: number }>();
const MAX_LOGIN_ATTEMPTS = 5;
const LOCKOUT_MINUTES = 15;

function checkRateLimit(ip: string): { allowed: boolean; remaining: number; lockedMinutes?: number } {
  const now = Date.now();
  const entry = loginAttempts.get(ip);
  if (!entry) return { allowed: true, remaining: MAX_LOGIN_ATTEMPTS };
  if (entry.lockedUntil > now) {
    const mins = Math.ceil((entry.lockedUntil - now) / 60000);
    return { allowed: false, remaining: 0, lockedMinutes: mins };
  }
  if (entry.lockedUntil > 0 && entry.lockedUntil <= now) {
    loginAttempts.delete(ip);
    return { allowed: true, remaining: MAX_LOGIN_ATTEMPTS };
  }
  return { allowed: true, remaining: MAX_LOGIN_ATTEMPTS - entry.count };
}

function recordFailedLogin(ip: string) {
  const entry = loginAttempts.get(ip) || { count: 0, lockedUntil: 0 };
  entry.count++;
  if (entry.count >= MAX_LOGIN_ATTEMPTS) {
    entry.lockedUntil = Date.now() + LOCKOUT_MINUTES * 60000;
    console.log(`[AUTH] IP ${ip} locked out for ${LOCKOUT_MINUTES} minutes after ${entry.count} failed attempts`);
  }
  loginAttempts.set(ip, entry);
}

function clearLoginAttempts(ip: string) { loginAttempts.delete(ip); }

// Clean up stale entries hourly
setInterval(() => {
  const now = Date.now();
  for (const [ip, entry] of loginAttempts.entries()) {
    if (entry.lockedUntil > 0 && entry.lockedUntil <= now) loginAttempts.delete(ip);
    else if (entry.count > 0 && entry.lockedUntil === 0) loginAttempts.delete(ip); // Bug 50: also clean non-locked stale entries
  }
  // Bug 17: Purge expired sessions (24h TTL)
  try { db.prepare("DELETE FROM sessions WHERE created_at < ?").run(Date.now() - 86400000); } catch {}
}, 3600000);

export function registerAuthRoutes(app: Express) {

  // ── Login ────────────────────────────────────────────────────────────────
  // Login — rate limited: 5 attempts then 15min lockout
  app.post("/api/auth/login", async (req, res) => {
    const clientIp = req.ip || req.socket.remoteAddress || "unknown";
    const rateCheck = checkRateLimit(clientIp);
    if (!rateCheck.allowed) {
      return res.status(429).json({
        error: `Too many login attempts. Try again in ${rateCheck.lockedMinutes} minutes.`,
        locked: true, retryAfterMinutes: rateCheck.lockedMinutes,
      });
    }

    const { email, password } = req.body || {};
    if (!email || !password) return res.status(400).json({ error: "Email and password required" });

    const user = db.prepare("SELECT id, email, password_hash, role FROM users WHERE email = ?").get(email.toLowerCase()) as any;
    if (!user) { recordFailedLogin(clientIp); return res.status(401).json({ error: "Invalid credentials", attemptsRemaining: checkRateLimit(clientIp).remaining }); }

    const valid = await bcrypt.compare(password, user.password_hash);
    if (!valid) { recordFailedLogin(clientIp); return res.status(401).json({ error: "Invalid credentials", attemptsRemaining: checkRateLimit(clientIp).remaining }); }

    clearLoginAttempts(clientIp);

    const token = crypto.randomBytes(32).toString("hex");
    db.prepare("INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)").run(token, user.id, Date.now());

    res.cookie("session", token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 86400000,
    });
    res.json({ ok: true, email: user.email, role: user.role || "user", isOwner: user.email === OWNER_EMAIL });
  });

  // ── Register ─────────────────────────────────────────────────────────────
  app.post("/api/auth/register", async (req, res) => {
    try {
      const { email, password } = req.body;
      if (!email || !password) return res.status(400).json({ error: "Email and password required" });
      if (typeof email !== "string" || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return res.status(400).json({ error: "Invalid email format" });
      if (password.length < 6) return res.status(400).json({ error: "Password must be at least 6 characters" });
      if (password.length > 72) return res.status(400).json({ error: "Password must be 72 characters or fewer" });

      const existing = db.prepare("SELECT id FROM users WHERE email = ?").get(email.toLowerCase());
      if (existing) return res.status(400).json({ error: "An account with this email already exists" });

      const hash = await bcrypt.hash(password, 10);
      const role = email.toLowerCase() === OWNER_EMAIL.toLowerCase() ? "owner" : "user";

      db.prepare("INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, ?, ?)").run(
        email.toLowerCase(), hash, role, new Date().toISOString()
      );

      // Send welcome email (non-blocking)
      const RESEND_KEY = process.env.RESEND_KEY || "";
      if (RESEND_KEY) {
        fetch("https://api.resend.com/emails", {
          method: "POST",
          headers: { "Authorization": `Bearer ${RESEND_KEY}`, "Content-Type": "application/json" },
          body: JSON.stringify({
            from: "VolTradeAI <onboarding@resend.dev>",
            to: email,
            subject: "Welcome to VolTradeAI",
            html: `<h2>Welcome to VolTradeAI</h2><p>Your account has been created successfully.</p><p>You can now save your watchlists and access all features at <a href="https://voltradeai.com">voltradeai.com</a></p>`,
          }),
        }).catch(() => {});
      }

      res.json({ ok: true, message: "Account created successfully" });
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // ── Logout ───────────────────────────────────────────────────────────────
  app.post("/api/auth/logout", (req, res) => {
    const token = (req as any).cookies?.session;
    if (token) db.prepare("DELETE FROM sessions WHERE token = ?").run(token);
    res.clearCookie("session");
    res.json({ ok: true });
  });

  // ── Me (returns role + isOwner) ──────────────────────────────────────────
  app.get("/api/auth/me", (req, res) => {
    const token = (req as any).cookies?.session;
    if (!token) return res.json({ authenticated: false });

    const session = db.prepare("SELECT user_id, created_at FROM sessions WHERE token = ?").get(token) as any;
    if (!session) return res.json({ authenticated: false });

    if (Date.now() - session.created_at > 86400000) {
      db.prepare("DELETE FROM sessions WHERE token = ?").run(token);
      return res.json({ authenticated: false });
    }

    const user = db.prepare("SELECT email, role FROM users WHERE id = ?").get(session.user_id) as any;
    if (!user) return res.json({ authenticated: false });

    res.json({
      authenticated: true,
      email: user.email,
      role: user.role || "user",
      isOwner: user.email === OWNER_EMAIL,
    });
  });

  // ── Password reset request ───────────────────────────────────────────────
  app.post("/api/auth/reset", async (req, res) => {
    try {
      const { email } = req.body;
      if (!email) return res.status(400).json({ error: "Email required" });

      const user = db.prepare("SELECT id FROM users WHERE email = ?").get(email.toLowerCase()) as any;
      if (!user) {
        // Don't reveal if email exists
        return res.json({ ok: true, message: "If that email exists, a reset link has been sent." });
      }

      const token = crypto.randomBytes(32).toString("hex");
      const expires = new Date(Date.now() + 3600000).toISOString(); // 1 hour

      db.prepare("DELETE FROM password_resets WHERE email = ?").run(email.toLowerCase());
      db.prepare("INSERT INTO password_resets (email, token, expires) VALUES (?, ?, ?)").run(
        email.toLowerCase(), token, expires
      );

      const RESEND_KEY = process.env.RESEND_KEY || "";
      const resetUrl = `${req.headers.origin || "https://voltradeai.com"}/reset?token=${token}`;

      if (RESEND_KEY) {
        try {
          await fetch("https://api.resend.com/emails", {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${RESEND_KEY}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              from: "VolTradeAI <onboarding@resend.dev>",
              to: email,
              subject: "Password Reset — VolTradeAI",
              html: `<p>Click this link to reset your password:</p><p><a href="${resetUrl}">${resetUrl}</a></p><p>This link expires in 1 hour.</p>`,
            }),
          });
        } catch (e: any) {
          console.log("[AUTH] Email send failed:", e.message);
        }
      } else {
        // No email service configured — log to console
        console.log(`[PASSWORD RESET] ${email} — Token: ${token} — URL: ${resetUrl}`);
      }

      res.json({ ok: true, message: "If that email exists, a reset link has been sent." });
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // ── Password reset confirm ───────────────────────────────────────────────
  app.post("/api/auth/reset-confirm", async (req, res) => {
    try {
      const { token, newPassword } = req.body || {};
      if (!token || !newPassword) return res.status(400).json({ error: "Token and new password required" });
      if (newPassword.length < 6) return res.status(400).json({ error: "Password must be at least 6 characters" });

      const entry = db.prepare("SELECT email, expires FROM password_resets WHERE token = ?").get(token) as any;
      if (!entry) return res.status(400).json({ error: "Invalid or expired reset link" });

      if (new Date(entry.expires) < new Date()) {
        db.prepare("DELETE FROM password_resets WHERE token = ?").run(token);
        return res.status(400).json({ error: "Reset link has expired" });
      }

      const hash = await bcrypt.hash(newPassword, 10);
      db.prepare("UPDATE users SET password_hash = ? WHERE email = ?").run(hash, entry.email);
      db.prepare("DELETE FROM password_resets WHERE token = ?").run(token);
      // Invalidate all existing sessions for this user (Bug 16)
      const resetUser = db.prepare("SELECT id FROM users WHERE email = ?").get(entry.email) as any;
      if (resetUser) db.prepare("DELETE FROM sessions WHERE user_id = ?").run(resetUser.id);

      res.json({ ok: true, message: "Password updated successfully" });
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // ── Legacy reset endpoints (kept for backwards compat) ────────────────────
  app.post("/api/auth/reset-request", (req, res) => {
    // Redirect to new reset endpoint logic
    res.json({ ok: true, message: "Use /api/auth/reset for password reset" });
  });
}
