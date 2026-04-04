import { useState, useEffect, useRef } from "react";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface LoginProps {
  onLogin?: () => void;
}

// ── City + Matrix Canvas Animation ───────────────────────────────────────────

function CityMatrixCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let W = window.innerWidth;
    let H = window.innerHeight;
    canvas.width = W;
    canvas.height = H;

    const handleResize = () => {
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width = W;
      canvas.height = H;
      generateCity();
    };
    window.addEventListener("resize", handleResize);

    // ── Generate city buildings ──
    interface Building {
      x: number; w: number; h: number;
      windows: { wx: number; wy: number; lit: boolean; color: string; flicker: number }[];
      antennaHeight: number;
    }
    interface Stream {
      x: number; y: number; speed: number; length: number; chars: string[];
      opacity: number; buildingTop: number;
    }

    let buildings: Building[] = [];
    let streams: Stream[] = [];

    const CHARS = "0101001101101001".split("");

    function generateCity() {
      buildings = [];
      streams = [];
      const groundY = H * 0.95;
      let x = -10;

      while (x < W + 50) {
        // NYC-style building variety
        const roll = Math.random();
        let w: number, h: number;
        
        if (roll < 0.05) {
          // Tall skyscraper (like One WTC)
          w = 30 + Math.random() * 20;
          h = H * 0.7 + Math.random() * (H * 0.2);
        } else if (roll < 0.15) {
          // Tall tower
          w = 25 + Math.random() * 35;
          h = H * 0.5 + Math.random() * (H * 0.25);
        } else if (roll < 0.35) {
          // Medium high-rise
          w = 30 + Math.random() * 50;
          h = H * 0.3 + Math.random() * (H * 0.25);
        } else if (roll < 0.6) {
          // Standard building
          w = 25 + Math.random() * 60;
          h = H * 0.15 + Math.random() * (H * 0.25);
        } else {
          // Low building
          w = 20 + Math.random() * 45;
          h = H * 0.08 + Math.random() * (H * 0.15);
        }
        const bTop = groundY - h;

        // Windows
        const windows: Building["windows"][] = [];
        const cols = Math.floor((w - 8) / 10);
        const rows = Math.floor((h - 15) / 12);
        const winArr: Building["windows"][number][] = [];
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const lit = Math.random() < 0.3;
            winArr.push({
              wx: x + 4 + c * 10,
              wy: bTop + 8 + r * 12,
              lit,
              color: Math.random() < 0.7 ? "#00e5ff" : "#d4a017",
              flicker: Math.random() * 1000,
            });
          }
        }

        const antennaHeight = Math.random() < 0.3 ? 10 + Math.random() * 25 : 0;

        buildings.push({ x, w, h, windows: winArr, antennaHeight });

        // Data streams from this building's top
        const streamCount = Math.floor(1 + Math.random() * 3);
        for (let s = 0; s < streamCount; s++) {
          streams.push({
            x: x + 5 + Math.random() * (w - 10),
            y: bTop - Math.random() * 200,
            speed: 1.5 + Math.random() * 4,
            length: 5 + Math.floor(Math.random() * 12),
            chars: Array.from({ length: 30 }, () => CHARS[Math.floor(Math.random() * CHARS.length)]),
            opacity: 0.15 + Math.random() * 0.35,
            buildingTop: bTop,
          });
        }

        x += w + Math.random() * 3; // Tight NYC spacing
      }
    }

    generateCity();

    let time = 0;

    function draw() {
      time++;
      ctx!.clearRect(0, 0, W, H);

      // Background gradient
      const grad = ctx!.createLinearGradient(0, 0, 0, H);
      grad.addColorStop(0, "#020408");
      grad.addColorStop(0.5, "#040810");
      grad.addColorStop(1, "#060c18");
      ctx!.fillStyle = grad;
      ctx!.fillRect(0, 0, W, H);

      // Subtle fog/glow at the base
      const fogGrad = ctx!.createLinearGradient(0, H * 0.7, 0, H);
      fogGrad.addColorStop(0, "transparent");
      fogGrad.addColorStop(1, "rgba(0, 229, 255, 0.02)");
      ctx!.fillStyle = fogGrad;
      ctx!.fillRect(0, H * 0.7, W, H * 0.3);

      const groundY = H * 0.95;

      // Draw buildings
      for (const b of buildings) {
        const bTop = groundY - b.h;

        // Building body — dark with slight gradient
        const bGrad = ctx!.createLinearGradient(b.x, bTop, b.x, groundY);
        bGrad.addColorStop(0, "#0a1420");
        bGrad.addColorStop(1, "#060e18");
        ctx!.fillStyle = bGrad;
        ctx!.fillRect(b.x, bTop, b.w, b.h);

        // Right edge shadow for depth
        const edgeGrad = ctx!.createLinearGradient(b.x + b.w - 8, bTop, b.x + b.w, bTop);
        edgeGrad.addColorStop(0, "transparent");
        edgeGrad.addColorStop(1, "rgba(0, 0, 0, 0.3)");
        ctx!.fillStyle = edgeGrad;
        ctx!.fillRect(b.x + b.w - 8, bTop, 8, b.h);

        // Building outline
        ctx!.strokeStyle = "rgba(0, 229, 255, 0.08)";
        ctx!.lineWidth = 0.5;
        ctx!.strokeRect(b.x, bTop, b.w, b.h);

        // Roof edge highlight
        ctx!.strokeStyle = "rgba(0, 229, 255, 0.15)";
        ctx!.lineWidth = 1;
        ctx!.beginPath();
        ctx!.moveTo(b.x, bTop);
        ctx!.lineTo(b.x + b.w, bTop);
        ctx!.stroke();

        // Antenna
        if (b.antennaHeight > 0) {
          const ax = b.x + b.w / 2;
          ctx!.strokeStyle = "rgba(0, 229, 255, 0.2)";
          ctx!.lineWidth = 1;
          ctx!.beginPath();
          ctx!.moveTo(ax, bTop);
          ctx!.lineTo(ax, bTop - b.antennaHeight);
          ctx!.stroke();
          // Blinking light
          if (Math.sin(time * 0.05 + b.x) > 0.3) {
            ctx!.fillStyle = "#ff3333";
            ctx!.beginPath();
            ctx!.arc(ax, bTop - b.antennaHeight, 1.5, 0, Math.PI * 2);
            ctx!.fill();
          }
        }

        // Windows
        for (const win of b.windows) {
          if (!win.lit) continue;
          const flick = Math.sin(time * 0.02 + win.flicker) * 0.5 + 0.5;
          const alpha = 0.05 + flick * 0.2;
          // Occasionally toggle windows
          if (Math.random() < 0.001) win.lit = !win.lit;
          ctx!.fillStyle = win.color === "#00e5ff"
            ? `rgba(0, 229, 255, ${alpha})`
            : `rgba(212, 160, 23, ${alpha})`;
          ctx!.fillRect(win.wx, win.wy, 6, 6);
        }
        // Randomly light up dark windows
        for (const win of b.windows) {
          if (win.lit) continue;
          if (Math.random() < 0.0005) win.lit = true;
        }
      }

      // Ground line
      ctx!.strokeStyle = "rgba(0, 229, 255, 0.1)";
      ctx!.lineWidth = 1;
      ctx!.beginPath();
      ctx!.moveTo(0, groundY);
      ctx!.lineTo(W, groundY);
      ctx!.stroke();

      // Draw Matrix data streams
      ctx!.font = "9px 'JetBrains Mono', 'Courier New', monospace";
      for (const s of streams) {
        s.y += s.speed;

        // Reset when stream goes below building top area
        if (s.y > s.buildingTop + 20) {
          s.y = s.buildingTop - 150 - Math.random() * 300;
          // Randomize chars
          for (let i = 0; i < s.chars.length; i++) {
            if (Math.random() < 0.3) {
              s.chars[i] = CHARS[Math.floor(Math.random() * CHARS.length)];
            }
          }
        }

        // Draw each character in the stream
        for (let i = 0; i < s.length; i++) {
          const cy = s.y + i * 10;
          if (cy < 0 || cy > s.buildingTop + 10) continue;

          const fade = 1 - (i / s.length);
          const alpha = s.opacity * fade;

          if (i === 0) {
            // Leading character is brightest (white-cyan)
            ctx!.fillStyle = `rgba(200, 255, 255, ${alpha * 1.5})`;
          } else {
            ctx!.fillStyle = `rgba(0, 229, 255, ${alpha})`;
          }
          ctx!.fillText(s.chars[i % s.chars.length], s.x, cy);

          // Randomly change characters as they fall
          if (Math.random() < 0.05) {
            s.chars[i % s.chars.length] = CHARS[Math.floor(Math.random() * CHARS.length)];
          }
        }
      }

      // Horizontal scan line
      const scanY = (time * 1.5) % (H * 2) - H * 0.5;
      ctx!.strokeStyle = "rgba(0, 229, 255, 0.06)";
      ctx!.lineWidth = 1;
      ctx!.beginPath();
      ctx!.moveTo(0, scanY);
      ctx!.lineTo(W, scanY);
      ctx!.stroke();
      // Glow around scan line
      const scanGrad = ctx!.createLinearGradient(0, scanY - 20, 0, scanY + 20);
      scanGrad.addColorStop(0, "transparent");
      scanGrad.addColorStop(0.5, "rgba(0, 229, 255, 0.015)");
      scanGrad.addColorStop(1, "transparent");
      ctx!.fillStyle = scanGrad;
      ctx!.fillRect(0, scanY - 20, W, 40);

      animId = requestAnimationFrame(draw);
    }

    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />;
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
      {/* Animated city + matrix background */}
      <CityMatrixCanvas />

      {/* Corner HUD markers */}
      <div style={{ position: "absolute", top: 16, left: 16, zIndex: 20, fontSize: 10, color: "#00e5ff", opacity: 0.5, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.15em" }}>
        VOLTRADE // SI-TK // NOFORN
      </div>
      <div style={{ position: "absolute", top: 16, right: 16, zIndex: 20, fontSize: 11, color: "#00e5ff", opacity: 0.4, fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.12em" }}>
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
