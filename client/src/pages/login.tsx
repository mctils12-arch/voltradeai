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
    width: "100%", padding: "12px 14px", background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.12)", borderRadius: "10px", color: "#f5f5f7",
    fontSize: "14px", outline: "none", marginBottom: "12px",
  };
  const btnStyle: React.CSSProperties = {
    width: "100%", padding: "12px", background: "#0a84ff", color: "white",
    border: "none", borderRadius: "10px", fontSize: "15px", fontWeight: 600,
    cursor: "pointer", marginTop: "4px",
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#000" }}>
      <div style={{ width: "100%", maxWidth: "380px", padding: "32px", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "20px", backdropFilter: "blur(20px)" }}>
        
        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: "24px" }}>
          <div style={{ fontSize: "28px", fontWeight: 700, color: "#f5f5f7", letterSpacing: "-0.5px" }}>
            VolTrade<span style={{ color: "#0a84ff" }}>AI</span>
          </div>
          <p style={{ fontSize: "13px", color: "#6e6e73", marginTop: "4px" }}>
            {resetMode ? "Reset your password" : "Sign in to your account"}
          </p>
        </div>

        {!resetMode ? (
          <form onSubmit={handleLogin}>
            <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} style={inputStyle} required />
            <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} style={inputStyle} required />
            {error && <p style={{ color: "#ff453a", fontSize: "13px", marginBottom: "8px" }}>{error}</p>}
            <button type="submit" style={btnStyle} disabled={loading}>{loading ? "Signing in..." : "Sign In"}</button>
            <button type="button" onClick={() => { setResetMode(true); setResetStep(0); }} style={{ background: "none", border: "none", color: "#0a84ff", fontSize: "13px", cursor: "pointer", marginTop: "12px", width: "100%", textAlign: "center" }}>
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
            {resetMsg && <p style={{ color: "#a1a1a6", fontSize: "13px", marginTop: "8px", textAlign: "center" }}>{resetMsg}</p>}
            <button onClick={() => { setResetMode(false); setResetStep(0); }} style={{ background: "none", border: "none", color: "#0a84ff", fontSize: "13px", cursor: "pointer", marginTop: "12px", width: "100%", textAlign: "center" }}>
              Back to Sign In
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
