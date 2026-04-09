import { AlertCircle } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen w-full flex items-center justify-center" style={{ background: "#0a1929" }}>
      <div style={{ textAlign: "center", color: "#c8d6e5" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem", marginBottom: "1rem" }}>
          <AlertCircle style={{ width: 32, height: 32, color: "#ff453a" }} />
          <h1 style={{ fontSize: "1.5rem", fontWeight: 700 }}>404 Page Not Found</h1>
        </div>
        <p style={{ fontSize: "0.85rem", color: "#7a8ba0" }}>
          The page you're looking for doesn't exist.
        </p>
      </div>
    </div>
  );
}
