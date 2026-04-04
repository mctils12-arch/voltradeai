import { useState, useEffect, useRef } from "react";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface LoginProps {
  onLogin?: () => void;
}

// ── Matrix City Canvas ──────────────────────────────────────────────────────

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

    interface Building {
      x: number; w: number; h: number;
      windows: { wx: number; wy: number; lit: boolean; color: string; flicker: number; size: number }[];
      antennaHeight: number;
      hasCrown: boolean;
      crownType: number;
    }
    interface Stream {
      x: number; y: number; speed: number; length: number; chars: string[];
      opacity: number; buildingTop: number;
    }

    const CHARS = "01".split("");
    let buildings: Building[] = [];
    let streams: Stream[] = [];

    function generateCity() {
      buildings = [];
      streams = [];
      const groundY = H * 0.92;
      let x = -5;

      while (x < W + 20) {
        const roll = Math.random();
        let w: number, h: number;
        if (roll < 0.04) { w = 28 + Math.random() * 18; h = H * 0.72 + Math.random() * (H * 0.18); }
        else if (roll < 0.12) { w = 22 + Math.random() * 30; h = H * 0.52 + Math.random() * (H * 0.22); }
        else if (roll < 0.3) { w = 28 + Math.random() * 45; h = H * 0.32 + Math.random() * (H * 0.22); }
        else if (roll < 0.6) { w = 22 + Math.random() * 55; h = H * 0.18 + Math.random() * (H * 0.22); }
        else { w = 18 + Math.random() * 40; h = H * 0.08 + Math.random() * (H * 0.14); }

        const bTop = groundY - h;
        const cols = Math.floor((w - 6) / 9);
        const rows = Math.floor((h - 12) / 10);
        const winArr: Building["windows"][number][] = [];
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            winArr.push({
              wx: x + 3 + c * 9,
              wy: bTop + 6 + r * 10,
              lit: Math.random() < 0.35,
              color: Math.random() < 0.6 ? "cyan" : "gold",
              flicker: Math.random() * 2000,
              size: 4 + Math.random() * 2,
            });
          }
        }
        const antennaHeight = Math.random() < 0.25 ? 12 + Math.random() * 30 : 0;
        const hasCrown = h > H * 0.4 && Math.random() < 0.4;
        const crownType = Math.floor(Math.random() * 3);

        buildings.push({ x, w, h, windows: winArr, antennaHeight, hasCrown, crownType });

        // Binary streams from this building
        const streamCount = Math.max(1, Math.floor(w / 20));
        for (let s = 0; s < streamCount; s++) {
          streams.push({
            x: x + 4 + Math.random() * (w - 8),
            y: bTop - Math.random() * 400,
            speed: 1.2 + Math.random() * 3.5,
            length: 6 + Math.floor(Math.random() * 14),
            chars: Array.from({ length: 25 }, () => CHARS[Math.floor(Math.random() * CHARS.length)]),
            opacity: 0.12 + Math.random() * 0.28,
            buildingTop: bTop,
          });
        }
        x += w + Math.random() * 3;
      }
    }

    generateCity();
    let time = 0;

    function draw() {
      time++;
      ctx!.clearRect(0, 0, W, H);

      // Sky gradient
      const grad = ctx!.createLinearGradient(0, 0, 0, H);
      grad.addColorStop(0, "#010306");
      grad.addColorStop(0.4, "#030810");
      grad.addColorStop(0.7, "#04091a");
      grad.addColorStop(1, "#050c1e");
      ctx!.fillStyle = grad;
      ctx!.fillRect(0, 0, W, H);

      const groundY = H * 0.92;

      // Buildings — glass tower rendering
      for (const b of buildings) {
        const bTop = groundY - b.h;

        // Glass facade body — semi-transparent with visible structure
        const bGrad = ctx!.createLinearGradient(b.x, bTop, b.x + b.w, groundY);
        bGrad.addColorStop(0, "rgba(8, 18, 35, 0.85)");
        bGrad.addColorStop(0.3, "rgba(6, 14, 28, 0.75)");
        bGrad.addColorStop(0.7, "rgba(4, 10, 22, 0.8)");
        bGrad.addColorStop(1, "rgba(3, 8, 16, 0.9)");
        ctx!.fillStyle = bGrad;
        ctx!.fillRect(b.x, bTop, b.w, b.h);

        // Glass reflection streak (diagonal light on facade)
        const reflGrad = ctx!.createLinearGradient(b.x, bTop, b.x + b.w * 0.6, bTop + b.h * 0.4);
        reflGrad.addColorStop(0, "rgba(0, 229, 255, 0.04)");
        reflGrad.addColorStop(0.3, "rgba(0, 229, 255, 0.02)");
        reflGrad.addColorStop(1, "transparent");
        ctx!.fillStyle = reflGrad;
        ctx!.fillRect(b.x, bTop, b.w * 0.6, b.h * 0.5);

        // Horizontal floor plate lines (see-through glass structure)
        const floorHeight = 10;
        ctx!.strokeStyle = "rgba(0, 229, 255, 0.06)";
        ctx!.lineWidth = 0.5;
        for (let fy = bTop + floorHeight; fy < groundY; fy += floorHeight) {
          ctx!.beginPath();
          ctx!.moveTo(b.x, fy);
          ctx!.lineTo(b.x + b.w, fy);
          ctx!.stroke();
        }

        // Vertical mullion lines (glass panel divisions)
        const panelWidth = Math.max(8, b.w / Math.floor(b.w / 10));
        ctx!.strokeStyle = "rgba(0, 229, 255, 0.04)";
        ctx!.lineWidth = 0.5;
        for (let fx = b.x + panelWidth; fx < b.x + b.w; fx += panelWidth) {
          ctx!.beginPath();
          ctx!.moveTo(fx, bTop);
          ctx!.lineTo(fx, groundY);
          ctx!.stroke();
        }

        // Right edge — darker glass face (3D depth)
        const edgeW = Math.min(12, b.w * 0.15);
        const edgeGrad = ctx!.createLinearGradient(b.x + b.w - edgeW, bTop, b.x + b.w, bTop);
        edgeGrad.addColorStop(0, "transparent");
        edgeGrad.addColorStop(1, "rgba(0, 0, 0, 0.5)");
        ctx!.fillStyle = edgeGrad;
        ctx!.fillRect(b.x + b.w - edgeW, bTop, edgeW, b.h);

        // Left edge — cyan glass highlight
        ctx!.fillStyle = "rgba(0, 229, 255, 0.05)";
        ctx!.fillRect(b.x, bTop, 1.5, b.h);

        // Roof line glow
        ctx!.strokeStyle = "rgba(0, 229, 255, 0.18)";
        ctx!.lineWidth = 1.5;
        ctx!.beginPath();
        ctx!.moveTo(b.x, bTop);
        ctx!.lineTo(b.x + b.w, bTop);
        ctx!.stroke();

        // Building outline — subtle glass edge
        ctx!.strokeStyle = "rgba(0, 229, 255, 0.07)";
        ctx!.lineWidth = 0.8;
        ctx!.strokeRect(b.x, bTop, b.w, b.h);

        // Crown (for tall buildings)
        if (b.hasCrown) {
          ctx!.fillStyle = "rgba(0, 229, 255, 0.08)";
          if (b.crownType === 0) {
            ctx!.beginPath();
            ctx!.moveTo(b.x + b.w / 2 - 5, bTop);
            ctx!.lineTo(b.x + b.w / 2, bTop - 20);
            ctx!.lineTo(b.x + b.w / 2 + 5, bTop);
            ctx!.fill();
            // Spire glow
            ctx!.fillStyle = "rgba(0, 229, 255, 0.15)";
            ctx!.beginPath();
            ctx!.arc(b.x + b.w / 2, bTop - 20, 2, 0, Math.PI * 2);
            ctx!.fill();
          } else if (b.crownType === 1) {
            ctx!.fillRect(b.x + b.w * 0.15, bTop - 10, b.w * 0.7, 10);
            ctx!.fillRect(b.x + b.w * 0.3, bTop - 18, b.w * 0.4, 8);
          } else {
            // Angled crown
            ctx!.beginPath();
            ctx!.moveTo(b.x, bTop);
            ctx!.lineTo(b.x + b.w * 0.5, bTop - 12);
            ctx!.lineTo(b.x + b.w, bTop);
            ctx!.fill();
          }
        }

        // Antenna with blinking red light
        if (b.antennaHeight > 0) {
          const ax = b.x + b.w / 2;
          ctx!.strokeStyle = "rgba(100, 120, 140, 0.3)";
          ctx!.lineWidth = 1;
          ctx!.beginPath();
          ctx!.moveTo(ax, bTop);
          ctx!.lineTo(ax, bTop - b.antennaHeight);
          ctx!.stroke();
          const blinkPhase = Math.sin(time * 0.04 + b.x * 0.1);
          if (blinkPhase > 0) {
            const alpha = blinkPhase * 0.9;
            ctx!.fillStyle = `rgba(255, 40, 40, ${alpha})`;
            ctx!.beginPath();
            ctx!.arc(ax, bTop - b.antennaHeight, 2.5, 0, Math.PI * 2);
            ctx!.fill();
            ctx!.fillStyle = `rgba(255, 40, 40, ${alpha * 0.25})`;
            ctx!.beginPath();
            ctx!.arc(ax, bTop - b.antennaHeight, 8, 0, Math.PI * 2);
            ctx!.fill();
          }
        }

        // Windows — glass panel glow (see-through feel)
        for (const win of b.windows) {
          if (!win.lit) {
            // Unlit windows still show faint glass reflection
            ctx!.fillStyle = "rgba(0, 229, 255, 0.008)";
            ctx!.fillRect(win.wx, win.wy, win.size, win.size * 0.6);
            if (Math.random() < 0.0004) win.lit = true;
            continue;
          }
          if (Math.random() < 0.0004) { win.lit = false; continue; }
          const flick = Math.sin(time * 0.012 + win.flicker) * 0.5 + 0.5;
          const alpha = 0.08 + flick * 0.25;
          // Inner glow (warm light from inside office)
          ctx!.fillStyle = win.color === "cyan"
            ? `rgba(0, 200, 235, ${alpha})`
            : `rgba(240, 200, 80, ${alpha})`;
          ctx!.fillRect(win.wx, win.wy, win.size, win.size * 0.6);
          // Outer glow halo (light bleeding through glass)
          ctx!.fillStyle = win.color === "cyan"
            ? `rgba(0, 229, 255, ${alpha * 0.15})`
            : `rgba(240, 200, 80, ${alpha * 0.12})`;
          ctx!.fillRect(win.wx - 1, win.wy - 0.5, win.size + 2, win.size * 0.6 + 1);
        }
      }

      // Ground line + glow
      ctx!.strokeStyle = "rgba(0, 229, 255, 0.08)";
      ctx!.lineWidth = 1;
      ctx!.beginPath();
      ctx!.moveTo(0, groundY);
      ctx!.lineTo(W, groundY);
      ctx!.stroke();
      // Ground fog
      const fogGrad = ctx!.createLinearGradient(0, groundY - 30, 0, groundY + 10);
      fogGrad.addColorStop(0, "transparent");
      fogGrad.addColorStop(0.5, "rgba(0, 229, 255, 0.015)");
      fogGrad.addColorStop(1, "rgba(0, 229, 255, 0.008)");
      ctx!.fillStyle = fogGrad;
      ctx!.fillRect(0, groundY - 30, W, 40);

      // Binary streams
      ctx!.font = "9px 'JetBrains Mono', 'Courier New', monospace";
      for (const s of streams) {
        s.y += s.speed;
        if (s.y > s.buildingTop + 15) {
          s.y = s.buildingTop - 100 - Math.random() * 350;
          for (let i = 0; i < s.chars.length; i++) {
            if (Math.random() < 0.4) s.chars[i] = CHARS[Math.floor(Math.random() * 2)];
          }
        }
        for (let i = 0; i < s.length; i++) {
          const cy = s.y + i * 10;
          if (cy < 0 || cy > s.buildingTop + 8) continue;
          const fade = 1 - (i / s.length);
          const alpha = s.opacity * fade;
          ctx!.fillStyle = i === 0
            ? `rgba(180, 255, 255, ${Math.min(alpha * 1.8, 0.6)})`
            : `rgba(0, 229, 255, ${alpha})`;
          ctx!.fillText(s.chars[i % s.chars.length], s.x, cy);
          if (Math.random() < 0.04) s.chars[i % s.chars.length] = CHARS[Math.floor(Math.random() * 2)];
        }
      }

      // Scan line
      const scanY = (time * 1.2) % (H * 2) - H * 0.3;
      const scanGrad = ctx!.createLinearGradient(0, scanY - 15, 0, scanY + 15);
      scanGrad.addColorStop(0, "transparent");
      scanGrad.addColorStop(0.5, "rgba(0, 229, 255, 0.025)");
      scanGrad.addColorStop(1, "transparent");
      ctx!.fillStyle = scanGrad;
      ctx!.fillRect(0, scanY - 15, W, 30);

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
      {/* Animated Matrix city */}
      <CityMatrixCanvas />

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
