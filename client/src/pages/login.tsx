import { useState } from "react";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface LoginProps {
  onLogin?: () => void;
}

export default function LoginPage({ onLogin }: LoginProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"login" | "reset">("login");
  const [resetSent, setResetSent] = useState(false);

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

  return (
    <div style={{
      minHeight: "100dvh",
      background: "#030508",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      position: "relative",
      overflow: "hidden",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    }}>
      {/* Matrix rain + city skyline background */}
      <div style={{ position: "absolute", inset: 0, zIndex: 0 }}>
        <canvas id="matrixCanvas" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", opacity: 0.4 }} />
        {/* City skyline SVG */}
        <svg viewBox="0 0 1440 400" preserveAspectRatio="xMidYMax slice" style={{
          position: "absolute", bottom: 0, left: 0, right: 0, width: "100%", height: "50%", opacity: 0.15,
        }}>
          {/* Buildings */}
          <rect x="50" y="120" width="40" height="280" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="55" y="130" width="8" height="8" fill="#00e5ff" fillOpacity="0.15" />
          <rect x="55" y="145" width="8" height="8" fill="#00e5ff" fillOpacity="0.1" />
          <rect x="55" y="160" width="8" height="8" fill="#00e5ff" fillOpacity="0.2" />
          <rect x="70" y="140" width="8" height="8" fill="#00e5ff" fillOpacity="0.12" />
          <rect x="70" y="155" width="8" height="8" fill="#00e5ff" fillOpacity="0.08" />
          
          <rect x="120" y="80" width="60" height="320" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="128" y="90" width="6" height="6" fill="#00e5ff" fillOpacity="0.2" />
          <rect x="140" y="90" width="6" height="6" fill="#d4a017" fillOpacity="0.15" />
          <rect x="152" y="90" width="6" height="6" fill="#00e5ff" fillOpacity="0.1" />
          <rect x="128" y="105" width="6" height="6" fill="#00e5ff" fillOpacity="0.12" />
          <rect x="140" y="105" width="6" height="6" fill="#00e5ff" fillOpacity="0.18" />
          <rect x="152" y="105" width="6" height="6" fill="#d4a017" fillOpacity="0.1" />
          <rect x="128" y="120" width="6" height="6" fill="#d4a017" fillOpacity="0.12" />
          <rect x="140" y="120" width="6" height="6" fill="#00e5ff" fillOpacity="0.15" />

          <rect x="210" y="150" width="35" height="250" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="215" y="160" width="5" height="5" fill="#00e5ff" fillOpacity="0.15" />
          <rect x="225" y="160" width="5" height="5" fill="#00e5ff" fillOpacity="0.1" />
          <rect x="235" y="160" width="5" height="5" fill="#d4a017" fillOpacity="0.12" />
          
          <rect x="280" y="60" width="80" height="340" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="290" y="70" width="8" height="8" fill="#00e5ff" fillOpacity="0.2" />
          <rect x="305" y="70" width="8" height="8" fill="#d4a017" fillOpacity="0.15" />
          <rect x="320" y="70" width="8" height="8" fill="#00e5ff" fillOpacity="0.1" />
          <rect x="335" y="70" width="8" height="8" fill="#00e5ff" fillOpacity="0.18" />
          <rect x="290" y="88" width="8" height="8" fill="#d4a017" fillOpacity="0.12" />
          <rect x="305" y="88" width="8" height="8" fill="#00e5ff" fillOpacity="0.15" />
          <rect x="320" y="88" width="8" height="8" fill="#00e5ff" fillOpacity="0.08" />
          
          <rect x="390" y="100" width="45" height="300" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="460" y="130" width="55" height="270" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="540" y="50" width="70" height="350" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="640" y="90" width="40" height="310" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="710" y="110" width="50" height="290" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="790" y="40" width="85" height="360" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="900" y="100" width="50" height="300" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="980" y="70" width="65" height="330" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="1070" y="130" width="40" height="270" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="1140" y="80" width="75" height="320" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="1240" y="110" width="50" height="290" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="1310" y="90" width="60" height="310" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          <rect x="1390" y="140" width="50" height="260" fill="#0a1628" stroke="#00e5ff" strokeWidth="0.5" strokeOpacity="0.3" />
          
          {/* Random lit windows on all buildings */}
          {Array.from({length: 80}, (_, i) => {
            const bx = [50,120,210,280,390,460,540,640,710,790,900,980,1070,1140,1240,1310][i % 16];
            const bw = [40,60,35,80,45,55,70,40,50,85,50,65,40,75,50,60][i % 16];
            const by = [120,80,150,60,100,130,50,90,110,40,100,70,130,80,110,90][i % 16];
            const wx = bx + 5 + Math.floor((i * 7) % (bw - 12));
            const wy = by + 10 + Math.floor((i * 13) % 120);
            const colors = ["#00e5ff", "#d4a017", "#00e5ff", "#00e5ff", "#d4a017"];
            return <rect key={i} x={wx} y={wy} width="5" height="5" fill={colors[i % 5]} fillOpacity={0.05 + (i % 5) * 0.04} />;
          })}
        </svg>
        
        {/* Streaming data columns (Matrix-style) from building tops */}
        <div style={{ position: "absolute", inset: 0, overflow: "hidden", opacity: 0.12 }}>
          {Array.from({length: 20}, (_, i) => (
            <div key={i} className="matrix-stream" style={{
              position: "absolute",
              left: `${5 + i * 5}%`,
              top: "-100%",
              width: "1px",
              height: "200%",
              background: `linear-gradient(180deg, transparent 0%, #00e5ff 30%, #00e5ff 50%, transparent 100%)`,
              animation: `matrixFall ${3 + (i % 4) * 2}s linear infinite`,
              animationDelay: `${(i * 0.7) % 5}s`,
            }} />
          ))}
        </div>
      </div>

      {/* CSS animations */}
      <style>{`
        @keyframes matrixFall {
          0% { transform: translateY(-50%); }
          100% { transform: translateY(50%); }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        @keyframes scanline {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
      `}</style>

      {/* Scan line sweep */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: "2px",
        background: "linear-gradient(90deg, transparent, rgba(0,229,255,0.3), transparent)",
        animation: "scanline 4s linear infinite",
        zIndex: 1,
      }} />

      {/* Corner markers */}
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 2, fontSize: 10, color: "#00e5ff", opacity: 0.5, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.15em" }}>
        VOLTRADE // SI-TK // NOFORN
      </div>
      <div style={{ position: "absolute", top: 16, right: 16, zIndex: 2, fontSize: 10, color: "#ff3333", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.1em" }}>
        <span style={{ animation: "blink 2s ease infinite" }}>●</span> REC {new Date().toISOString().slice(0,19).replace('T',' ')}
      </div>
      <div style={{ position: "absolute", bottom: 16, left: 16, zIndex: 2, fontSize: 9, color: "#4a5568", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.1em" }}>
        SYS: ONLINE | ML: 36 FEATURES | ALPACA: CONNECTED
      </div>
      <div style={{ position: "absolute", bottom: 16, right: 16, zIndex: 2, fontSize: 9, color: "#4a5568", fontFamily: "'JetBrains Mono', monospace" }}>
        v2.0 // PAPER TRADING MODE
      </div>

      {/* Login card */}
      <div style={{
        position: "relative",
        zIndex: 10,
        width: "100%",
        maxWidth: 380,
        padding: "2.5rem 2rem",
        background: "rgba(3, 5, 8, 0.88)",
        border: "1px solid rgba(0, 229, 255, 0.12)",
        borderRadius: 6,
        boxShadow: "0 0 40px rgba(0, 229, 255, 0.04), 0 0 80px rgba(0, 0, 0, 0.6)",
        margin: "0 1rem",
      }}>
        {/* Classification header */}
        <div style={{ textAlign: "center", marginBottom: "0.5rem" }}>
          <span style={{ fontSize: 9, letterSpacing: "0.2em", color: "#00e5ff", opacity: 0.5 }}>
            TOP SECRET // AUTHORIZED PERSONNEL ONLY
          </span>
        </div>

        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
          <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: "0.08em", fontFamily: "'Inter', 'JetBrains Mono', sans-serif" }}>
            <span style={{ color: "#ffffff" }}>Vol</span>
            <span style={{ color: "#ffffff" }}>Trade</span>
            <span style={{ color: "#00e5ff" }}>AI</span>
          </div>
          <div style={{ fontSize: 11, color: "#5a6577", letterSpacing: "0.15em", marginTop: 6, textTransform: "uppercase" }}>
            {mode === "login" ? "SIGN IN TO YOUR ACCOUNT" : "RESET YOUR PASSWORD"}
          </div>
        </div>

        {mode === "login" ? (
          <form onSubmit={handleSubmit}>
            <input
              type="email" placeholder="Email" value={email}
              onChange={e => setEmail(e.target.value)} required
              style={{
                width: "100%", padding: "12px 14px", marginBottom: 12,
                background: "rgba(0, 15, 30, 0.6)",
                border: "1px solid rgba(0, 229, 255, 0.15)",
                borderRadius: 4, color: "#c8d6e5", fontSize: 13,
                fontFamily: "'JetBrains Mono', monospace",
                outline: "none", boxSizing: "border-box",
                letterSpacing: "0.03em",
              }}
              onFocus={e => e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"}
              onBlur={e => e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"}
            />
            <input
              type="password" placeholder="Password" value={password}
              onChange={e => setPassword(e.target.value)} required
              style={{
                width: "100%", padding: "12px 14px", marginBottom: 16,
                background: "rgba(0, 15, 30, 0.6)",
                border: "1px solid rgba(0, 229, 255, 0.15)",
                borderRadius: 4, color: "#c8d6e5", fontSize: 13,
                fontFamily: "'JetBrains Mono', monospace",
                outline: "none", boxSizing: "border-box",
                letterSpacing: "0.03em",
              }}
              onFocus={e => e.target.style.borderColor = "rgba(0, 229, 255, 0.4)"}
              onBlur={e => e.target.style.borderColor = "rgba(0, 229, 255, 0.15)"}
            />
            {error && <div style={{ color: "#ff3333", fontSize: 12, marginBottom: 12, textAlign: "center" }}>{error}</div>}
            <button type="submit" disabled={loading} style={{
              width: "100%", padding: "12px", 
              background: "rgba(0, 229, 255, 0.08)",
              border: "1px solid rgba(0, 229, 255, 0.3)",
              borderRadius: 4, color: "#00e5ff", fontSize: 12,
              fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
              fontFamily: "'JetBrains Mono', monospace",
              cursor: "pointer",
              transition: "all 200ms ease",
            }}
            onMouseOver={e => { (e.target as any).style.background = "rgba(0, 229, 255, 0.15)"; (e.target as any).style.boxShadow = "0 0 20px rgba(0, 229, 255, 0.1)"; }}
            onMouseOut={e => { (e.target as any).style.background = "rgba(0, 229, 255, 0.08)"; (e.target as any).style.boxShadow = "none"; }}
            >
              {loading ? "AUTHENTICATING..." : "SIGN IN"}
            </button>
          </form>
        ) : (
          <form onSubmit={handleReset}>
            {resetSent ? (
              <div style={{ textAlign: "center", color: "#00e5ff", fontSize: 13, padding: "1rem 0" }}>
                Reset link sent to {email}
              </div>
            ) : (
              <>
                <input
                  type="email" placeholder="Email" value={email}
                  onChange={e => setEmail(e.target.value)} required
                  style={{
                    width: "100%", padding: "12px 14px", marginBottom: 16,
                    background: "rgba(0, 15, 30, 0.6)",
                    border: "1px solid rgba(0, 229, 255, 0.15)",
                    borderRadius: 4, color: "#c8d6e5", fontSize: 13,
                    fontFamily: "'JetBrains Mono', monospace",
                    outline: "none", boxSizing: "border-box",
                  }}
                />
                {error && <div style={{ color: "#ff3333", fontSize: 12, marginBottom: 12, textAlign: "center" }}>{error}</div>}
                <button type="submit" disabled={loading} style={{
                  width: "100%", padding: "12px",
                  background: "rgba(0, 229, 255, 0.08)",
                  border: "1px solid rgba(0, 229, 255, 0.3)",
                  borderRadius: 4, color: "#00e5ff", fontSize: 12,
                  fontWeight: 600, letterSpacing: "0.15em", textTransform: "uppercase",
                  fontFamily: "'JetBrains Mono', monospace",
                  cursor: "pointer",
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
              fontFamily: "'JetBrains Mono', monospace",
              opacity: 0.7,
            }}
          >
            {mode === "login" ? "Forgot Password?" : "← Back to Sign In"}
          </button>
        </div>
      </div>
    </div>
  );
}
