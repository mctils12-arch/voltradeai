import { useState } from "react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import loginBgPath from "@assets/login_bg.png";

interface LoginProps {
  onLogin?: () => void;
}

// ── Login Page ───────────────────────────────────────────────────────────────

export default function LoginPage({ onLogin }: LoginProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmNewPassword, setConfirmNewPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [resetSent, setResetSent] = useState(false);
  const [signupSuccess, setSignupSuccess] = useState(false);
  const [resetComplete, setResetComplete] = useState(false);

  // Check if URL has a reset token
  const urlParams = new URLSearchParams(window.location.search);
  const resetToken = urlParams.get("token");
  const [mode, setMode] = useState<"login" | "reset" | "signup" | "reset-confirm">(resetToken ? "reset-confirm" : "login");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await apiRequest("POST", "/api/auth/login", { email, password });
      const data = await res.json();
      if (data.ok) {
        queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
        onLogin?.();
      } else {
        setError(data.error || "Invalid credentials");
      }
    } catch {
      setError("Connection failed");
    }
    setLoading(false);
  };

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await apiRequest("POST", "/api/auth/reset", { email });
      setResetSent(true);
    } catch {
      setError("Reset failed");
    }
    setLoading(false);
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }
    setLoading(true);
    try {
      const res = await apiRequest("POST", "/api/auth/register", { email, password });
      const data = await res.json();
      if (data.ok) {
        // Auto-login after registration
        const loginRes = await apiRequest("POST", "/api/auth/login", { email, password });
        const loginData = await loginRes.json();
        if (loginData.ok) {
          queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
          onLogin?.();
        } else {
          setSignupSuccess(true);
          setMode("login");
        }
      } else {
        setError(data.error || "Registration failed");
      }
    } catch {
      setError("Connection failed");
    }
    setLoading(false);
  };

  const handleResetConfirm = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (newPassword !== confirmNewPassword) {
      setError("Passwords do not match");
      return;
    }
    if (newPassword.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }
    setLoading(true);
    try {
      const res = await apiRequest("POST", "/api/auth/reset-confirm", { token: resetToken, newPassword });
      const data = await res.json();
      if (data.ok) {
        setResetComplete(true);
        // Clear the token from URL
        window.history.replaceState({}, "", window.location.pathname);
      } else {
        setError(data.error || "Reset failed — link may have expired");
      }
    } catch {
      setError("Connection failed");
    }
    setLoading(false);
  };

  const inputStyle: React.CSSProperties = {
    width: "100%", padding: "12px 14px", marginBottom: 12,
    background: "rgba(0, 15, 30, 0.7)",
    border: "1px solid rgba(0, 229, 255, 0.15)",
    borderRadius: 4, color: "#c8d6e5", fontSize: 13,
    fontFamily: "'JetBrains Mono', monospace",
    outline: "none", boxSizing: "border-box",
    letterSpacing: "0.03em",
  };

  return (
    <div style={{
      minHeight: "100dvh",
      background: "#020408",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      position: "relative",
      overflow: "hidden",
    }}>
      {/* Background image + overlay */}
      <div style={{ position: "absolute", inset: 0, zIndex: 0 }}>
        <img src={loginBgPath} alt="" style={{
          position: "absolute", inset: 0, width: "100%", height: "100%",
          objectFit: "cover", objectPosition: "center 40%",
        }} />
        <div style={{
          position: "absolute", inset: 0,
          background: "linear-gradient(180deg, rgba(3,5,10,0.6) 0%, rgba(3,5,10,0.8) 100%)",
        }} />
      </div>

      {/* Binary data streams overlay */}
      <div style={{ position: "absolute", inset: 0, overflow: "hidden", zIndex: 1, opacity: 0.08, pointerEvents: "none" }}>
        {Array.from({length: 15}, (_, i) => (
          <div key={i} style={{
            position: "absolute",
            left: `${3 + i * 6.5}%`,
            top: "-100%",
            width: "1px",
            height: "200%",
            background: "linear-gradient(180deg, transparent 0%, #00e5ff 30%, #00e5ff 50%, transparent 100%)",
            animation: `matrixFall ${3 + (i % 4) * 2}s linear infinite`,
            animationDelay: `${(i * 0.7) % 5}s`,
          }} />
        ))}
      </div>

      {/* Scan line */}
      <div style={{
        position: "absolute", inset: 0, zIndex: 2, pointerEvents: "none", overflow: "hidden",
      }}>
        <div style={{
          position: "absolute", left: 0, right: 0, height: "1px",
          background: "rgba(0, 229, 255, 0.06)",
          animation: "scanLine 6s linear infinite",
        }} />
      </div>

      {/* Corner HUD markers */}
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 20, fontSize: 12, color: "#00e5ff", opacity: 0.5, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.15em", fontWeight: 600 }}>
        VOLTRADEAI
      </div>
      <div style={{ position: "absolute", bottom: 16, left: 16, zIndex: 20, fontSize: 9, color: "#3a4a5c", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.1em" }}>
        SYS: ONLINE | ML: 36 FEATURES | ALPACA: CONNECTED
      </div>
      <div style={{ position: "absolute", bottom: 16, right: 16, zIndex: 20, fontSize: 9, color: "#3a4a5c", fontFamily: "'JetBrains Mono', monospace" }}>
        v2.0 // PAPER TRADING MODE
      </div>

      <style>{`
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }
        @keyframes matrixFall { 0% { transform: translateY(0); } 100% { transform: translateY(100vh); } }
        @keyframes scanLine { 0% { top: -2px; } 100% { top: 100%; } }
      `}</style>

      {/* Login card */}
      <div style={{
        position: "relative",
        zIndex: 10,
        width: "100%",
        maxWidth: 380,
        padding: "2.5rem 2rem",
        background: "rgba(3, 5, 10, 0.92)",
        border: "1px solid rgba(0, 229, 255, 0.12)",
        borderRadius: 6,
        boxShadow: "0 0 60px rgba(0, 0, 0, 0.8), 0 0 30px rgba(0, 229, 255, 0.03)",
        backdropFilter: "blur(12px)",
        margin: "0 1rem",
      }}>
        <div style={{ textAlign: "center", marginBottom: "0.5rem" }}>
          <span style={{ fontSize: 9, letterSpacing: "0.2em", color: "#00e5ff", opacity: 0.4, fontFamily: "'JetBrains Mono', monospace" }}>
            TOP SECRET // AUTHORIZED PERSONNEL ONLY
          </span>
        </div>

        <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
          <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: "0.08em", fontFamily: "'Inter', 'JetBrains Mono', sans-serif" }}>
            <span style={{ color: "#ffffff" }}>VolTrade</span>
            <span style={{ color: "#00e5ff" }}>AI</span>
          </div>
          <div style={{ fontSize: 11, color: "#4a5c70", letterSpacing: "0.15em", marginTop: 6, textTransform: "uppercase", fontFamily: "'JetBrains Mono', monospace" }}>
            {mode === "login" ? "SIGN IN TO YOUR ACCOUNT" : mode === "signup" ? "CREATE AN ACCOUNT" : mode === "reset-confirm" ? "SET NEW PASSWORD" : "RESET YOUR PASSWORD"}
          </div>
        </div>

        {signupSuccess && mode === "login" && (
          <div style={{ color: "#30d158", fontSize: 12, marginBottom: 12, textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>
            Account created! Sign in below.
          </div>
        )}

        {mode === "login" ? (
          <form onSubmit={handleSubmit}>
            <input
              type="email" placeholder="Email" value={email}
              onChange={e => setEmail(e.target.value)} required
              style={inputStyle}
              onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
            />
            <input
              type="password" placeholder="Password" value={password}
              onChange={e => setPassword(e.target.value)} required
              style={{ ...inputStyle, marginBottom: 16 }}
              onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
            />
            {error && <div style={{ color: "#ff3333", fontSize: 12, marginBottom: 12, textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>}
            <button type="submit" disabled={loading} style={{
              width: "100%", padding: "12px",
              background: "rgba(0, 229, 255, 0.08)",
              border: "1px solid rgba(0, 229, 255, 0.3)",
              borderRadius: 4, color: "#00e5ff", fontSize: 12,
              fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
              fontFamily: "'JetBrains Mono', monospace",
              cursor: "pointer", transition: "all 200ms ease",
            }}
              onMouseOver={e => { (e.target as HTMLElement).style.background = "rgba(0, 229, 255, 0.15)"; (e.target as HTMLElement).style.boxShadow = "0 0 20px rgba(0, 229, 255, 0.1)"; }}
              onMouseOut={e => { (e.target as HTMLElement).style.background = "rgba(0, 229, 255, 0.08)"; (e.target as HTMLElement).style.boxShadow = "none"; }}
            >
              {loading ? "AUTHENTICATING..." : "SIGN IN"}
            </button>
          </form>
        ) : mode === "signup" ? (
          <form onSubmit={handleSignup}>
            <input
              type="email" placeholder="Email" value={email}
              onChange={e => setEmail(e.target.value)} required
              style={inputStyle}
              onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
            />
            <input
              type="password" placeholder="Password (min 6 chars)" value={password}
              onChange={e => setPassword(e.target.value)} required
              style={inputStyle}
              onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
            />
            <input
              type="password" placeholder="Confirm Password" value={confirmPassword}
              onChange={e => setConfirmPassword(e.target.value)} required
              style={{ ...inputStyle, marginBottom: 16 }}
              onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
            />
            {error && <div style={{ color: "#ff3333", fontSize: 12, marginBottom: 12, textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>}
            <button type="submit" disabled={loading} style={{
              width: "100%", padding: "12px",
              background: "rgba(0, 229, 255, 0.08)",
              border: "1px solid rgba(0, 229, 255, 0.3)",
              borderRadius: 4, color: "#00e5ff", fontSize: 12,
              fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
              fontFamily: "'JetBrains Mono', monospace",
              cursor: "pointer", transition: "all 200ms ease",
            }}
              onMouseOver={e => { (e.target as HTMLElement).style.background = "rgba(0, 229, 255, 0.15)"; (e.target as HTMLElement).style.boxShadow = "0 0 20px rgba(0, 229, 255, 0.1)"; }}
              onMouseOut={e => { (e.target as HTMLElement).style.background = "rgba(0, 229, 255, 0.08)"; (e.target as HTMLElement).style.boxShadow = "none"; }}
            >
              {loading ? "CREATING ACCOUNT..." : "CREATE ACCOUNT"}
            </button>
          </form>
        ) : (
          mode === "reset-confirm" ? (
          <form onSubmit={handleResetConfirm}>
            {resetComplete ? (
              <div style={{ textAlign: "center", padding: "1rem 0" }}>
                <div style={{ color: "#30d158", fontSize: 13, marginBottom: 12, fontFamily: "'JetBrains Mono', monospace" }}>Password reset successfully</div>
                <button
                  type="button"
                  onClick={() => { setMode("login"); setResetComplete(false); setNewPassword(""); setConfirmNewPassword(""); }}
                  style={{
                    width: "100%", padding: "12px",
                    background: "rgba(0, 229, 255, 0.08)",
                    border: "1px solid rgba(0, 229, 255, 0.3)",
                    borderRadius: 4, color: "#00e5ff", fontSize: 12,
                    fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
                    fontFamily: "'JetBrains Mono', monospace", cursor: "pointer",
                  }}
                >
                  SIGN IN WITH NEW PASSWORD
                </button>
              </div>
            ) : (
              <>
                <input
                  type="password" placeholder="New Password" value={newPassword}
                  onChange={e => setNewPassword(e.target.value)} required
                  style={inputStyle}
                  onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
                  onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
                />
                <input
                  type="password" placeholder="Confirm New Password" value={confirmNewPassword}
                  onChange={e => setConfirmNewPassword(e.target.value)} required
                  style={{ ...inputStyle, marginBottom: 16 }}
                  onFocus={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"; e.target.style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.08)"; }}
                  onBlur={e => { e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"; e.target.style.boxShadow = "none"; }}
                />
                {error && <div style={{ color: "#ff453a", fontSize: 12, marginBottom: 12, textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>{error}</div>}
                <button type="submit" disabled={loading} style={{
                  width: "100%", padding: "12px",
                  background: "rgba(0, 229, 255, 0.08)",
                  border: "1px solid rgba(0, 229, 255, 0.3)",
                  borderRadius: 4, color: "#00e5ff", fontSize: 12,
                  fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
                  fontFamily: "'JetBrains Mono', monospace", cursor: "pointer",
                }}>
                  {loading ? "RESETTING..." : "RESET PASSWORD"}
                </button>
              </>
            )}
          </form>
        ) :
          <form onSubmit={handleReset}>
            {resetSent ? (
              <div style={{ textAlign: "center", color: "#00e5ff", fontSize: 13, padding: "1rem 0", fontFamily: "'JetBrains Mono', monospace" }}>
                Reset link sent to {email}
              </div>
            ) : (
              <>
                <input type="email" placeholder="Email" value={email}
                  onChange={e => setEmail(e.target.value)} required
                  style={{ ...inputStyle, marginBottom: 16 }}
                />
                {error && <div style={{ color: "#ff3333", fontSize: 12, marginBottom: 12, textAlign: "center" }}>{error}</div>}
                <button type="submit" disabled={loading} style={{
                  width: "100%", padding: "12px",
                  background: "rgba(0, 229, 255, 0.08)",
                  border: "1px solid rgba(0, 229, 255, 0.3)",
                  borderRadius: 4, color: "#00e5ff", fontSize: 12,
                  fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
                  fontFamily: "'JetBrains Mono', monospace", cursor: "pointer",
                }}>
                  {loading ? "SENDING..." : "SEND RESET LINK"}
                </button>
              </>
            )}
          </form>
        )}

        <div style={{ textAlign: "center", marginTop: 16 }}>
          <button
            onClick={() => { setMode(mode === "login" ? "reset" : "login"); setError(""); setResetSent(false); }}
            style={{
              background: "none", border: "none", color: "#d4a017",
              fontSize: 11, cursor: "pointer", letterSpacing: "0.1em",
              fontFamily: "'JetBrains Mono', monospace", opacity: 0.7,
            }}
          >
            {mode === "login" ? "Forgot Password?" : "← Back to Sign In"}
          </button>
        </div>

        <div style={{ textAlign: "center", marginTop: 12 }}>
          <button
            onClick={() => { setMode(mode === "signup" ? "login" : "signup"); setError(""); setConfirmPassword(""); }}
            style={{
              background: "none", border: "none", color: "#00e5ff",
              fontSize: 11, cursor: "pointer", letterSpacing: "0.08em",
              fontFamily: "'JetBrains Mono', monospace", opacity: 0.6,
            }}
          >
            {mode === "signup" ? "Already have an account? Sign In" : "Don't have an account? Create one"}
          </button>
        </div>
      </div>
    </div>
  );
}
