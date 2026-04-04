import { useState } from "react";
import { apiRequest } from "@/lib/queryClient";

export default function LoginPage({ onLogin }: { onLogin: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [resetMode, setResetMode] = useState(false);
  const [resetStep, setResetStep] = useState(0); // 0=enter email, 1=enter code+new pw
  const [resetCode, setResetCode] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [resetMsg, setResetMsg] = useState("");

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      const res = await apiRequest("POST", "/api/auth/login", { email, password });
      const d = await res.json();
      if (d.ok) onLogin();
      else setError(d.error || "Login failed");
    } catch { setError("Connection error"); }
    setLoading(false);
  }

  async function handleResetRequest() {
    setResetMsg(""); setLoading(true);
    try {
      await apiRequest("POST", "/api/auth/reset-request", { email });
      setResetMsg("If that email exists, a reset code was generated. Check the server logs.");
      setResetStep(1);
    } catch { setResetMsg("Error sending reset"); }
    setLoading(false);
  }

  async function handleResetConfirm() {
    setResetMsg(""); setLoading(true);
    try {
      const res = await apiRequest("POST", "/api/auth/reset-confirm", { email, code: resetCode, newPassword });
      const d = await res.json();
      if (d.ok) { setResetMsg("Password reset! You can now log in."); setResetMode(false); setResetStep(0); }
      else setResetMsg(d.error || "Reset failed");
    } catch { setResetMsg("Error"); }
    setLoading(false);
  }

  const inputStyle: React.CSSProperties = {
    width: "100%", padding: "12px 14px", background: "rgba(0, 15, 30, 0.7)",
    border: "1px solid rgba(0, 229, 255, 0.25)", borderRadius: "3px", color: "#c8d6e5",
    fontSize: "13px", outline: "none", marginBottom: "12px",
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
    letterSpacing: "0.03em",
  };
  const btnStyle: React.CSSProperties = {
    width: "100%", padding: "12px", background: "rgba(0, 229, 255, 0.15)",
    color: "#00e5ff", border: "1px solid #00e5ff", borderRadius: "3px",
    fontSize: "13px", fontWeight: 600, cursor: "pointer", marginTop: "4px",
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
    textTransform: "uppercase" as const, letterSpacing: "0.1em",
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#050a12" }}>
      <div style={{ width: "100%", maxWidth: "380px", padding: "32px", background: "rgba(0, 20, 40, 0.8)", border: "1px solid rgba(0, 229, 255, 0.25)", borderRadius: "3px", backdropFilter: "blur(20px)", boxShadow: "0 0 40px rgba(0, 229, 255, 0.05)" }}>
        
        {/* Clearance notice */}
        <div style={{ textAlign: "center", marginBottom: "16px", fontSize: "10px", color: "rgba(0, 229, 255, 0.4)", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.15em", textTransform: "uppercase" }}>
          TOP SECRET // AUTHORIZED PERSONNEL
        </div>

        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: "24px" }}>
          <div style={{ fontSize: "26px", fontWeight: 700, color: "#c8d6e5", letterSpacing: "0.05em", fontFamily: "'JetBrains Mono', monospace" }}>
            <span style={{ color: "#d4a017" }}>VolTrade</span><span style={{ color: "#00e5ff" }}>AI</span>
          </div>
          <p style={{ fontSize: "11px", color: "rgba(0, 229, 255, 0.5)", marginTop: "6px", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            {resetMode ? "Reset your password" : "Sign in to your account"}
          </p>
        </div>

        {!resetMode ? (
          <form onSubmit={handleLogin}>
            <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} style={inputStyle} required />
            <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} style={inputStyle} required />
            {error && <p style={{ color: "#ff3333", fontSize: "12px", marginBottom: "8px", fontFamily: "'JetBrains Mono', monospace" }}>{error}</p>}
            <button type="submit" style={btnStyle} disabled={loading}>{loading ? "Authenticating..." : "Sign In"}</button>
            <button type="button" onClick={() => { setResetMode(true); setResetStep(0); }} style={{ background: "none", border: "none", color: "#d4a017", fontSize: "12px", cursor: "pointer", marginTop: "12px", width: "100%", textAlign: "center", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.05em" }}>
              Forgot Password?
            </button>
          </form>
        ) : (
          <div>
            {resetStep === 0 && (
              <>
                <input type="email" placeholder="Your email" value={email} onChange={e => setEmail(e.target.value)} style={inputStyle} />
                <button onClick={handleResetRequest} style={btnStyle} disabled={loading}>Send Reset Code</button>
              </>
            )}
            {resetStep === 1 && (
              <>
                <input type="text" placeholder="Reset code" value={resetCode} onChange={e => setResetCode(e.target.value)} style={inputStyle} />
                <input type="password" placeholder="New password" value={newPassword} onChange={e => setNewPassword(e.target.value)} style={inputStyle} />
                <button onClick={handleResetConfirm} style={btnStyle} disabled={loading}>Reset Password</button>
              </>
            )}
            {resetMsg && <p style={{ color: "#7a8ba0", fontSize: "12px", marginTop: "8px", textAlign: "center", fontFamily: "'JetBrains Mono', monospace" }}>{resetMsg}</p>}
            <button onClick={() => { setResetMode(false); setResetStep(0); }} style={{ background: "none", border: "none", color: "#d4a017", fontSize: "12px", cursor: "pointer", marginTop: "12px", width: "100%", textAlign: "center", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.05em" }}>
              Back to Sign In
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
