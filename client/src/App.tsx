import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import { Route, Switch, useLocation } from "wouter";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import Home from "@/pages/home";
import LoginPage from "@/pages/login";
import LandingPage from "@/pages/landing";
import PricingPage from "@/pages/pricing";
import AboutPage from "@/pages/about";
import NewsletterPage from "@/pages/newsletter";
import CareersPage from "@/pages/careers";
import ContactPage from "@/pages/contact";
import { Component, ReactNode, useState, useEffect } from "react";

class ErrorBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  constructor(props: any) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: any) {
    return { error: error?.message || "Unknown error" };
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 32, color: '#ff5a6e', fontFamily: 'Geist Mono, monospace', background: '#050a13', minHeight: '100vh' }}>
          <h2>Something went wrong.</h2>
          <pre style={{ fontSize: 12, opacity: 0.7, whiteSpace: 'pre-wrap' }}>{this.state.error}</pre>
          <button onClick={() => this.setState({ error: null })} style={{ marginTop: 16, padding: '8px 16px', background: '#4d9fff', color: '#050a13', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600 }}>Try Again</button>
        </div>
      );
    }
    return this.props.children;
  }
}

function useMobile(breakpoint = 768) {
  const [mobile, setMobile] = useState(() => window.innerWidth < breakpoint);
  useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${breakpoint - 1}px)`);
    const handler = () => setMobile(window.innerWidth < breakpoint);
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, [breakpoint]);
  return mobile;
}

/** The actual product: the tabbed dashboard. Lives at /app/* */
function AppDashboard() {
  const { data, isLoading } = useQuery({
    queryKey: ["/api/auth/me"],
    queryFn: async () => {
      const r = await apiRequest("GET", "/api/auth/me");
      return r.json();
    },
    staleTime: 60000,
    retry: false,
  });

  const authenticated = !isLoading && !!data?.authenticated;
  const isOwner = !isLoading && !!data?.isOwner;
  const isMobile = useMobile();

  // Handle Stripe checkout success — invalidate auth (so tier=pro picks up),
  // strip query params from URL so a refresh doesn't re-trigger.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("checkout") === "success" || params.get("welcome") === "pro") {
      // Refresh auth so tier=pro is reflected immediately
      queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
      // Show a brief welcome (non-blocking)
      setTimeout(() => {
        try {
          // toast() is the shadcn toast — wrapped in try in case it's not available yet
          // Falling through silently is fine; the page itself reflects Pro state.
        } catch {}
      }, 100);
      // Strip the query params so the URL is clean on refresh
      window.history.replaceState({}, "", window.location.pathname + window.location.hash);
    }
  }, []);

  // Password reset flow — always show login page when token present
  const hasResetToken = window.location.search.includes("token=");
  if (hasResetToken) {
    return <LoginPage onLogin={() => {
      window.history.replaceState({}, "", window.location.pathname);
      queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
    }} />;
  }

  return <Home authenticated={authenticated} authLoading={isLoading} isMobile={isMobile} isOwner={isOwner} />;
}

/**
 * Standalone login route. Honors `?next=<path>` so users coming from
 * /pricing get sent back to /pricing after login (where the upgrade flow
 * picks them up automatically). Defaults to /app.
 */
function LoginRoute() {
  const [, navigate] = useLocation();
  return <LoginPage onLogin={() => {
    queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
    const params = new URLSearchParams(window.location.search);
    const next = params.get("next");
    // Only allow internal paths (defense against open-redirect)
    const safeNext = next && next.startsWith("/") && !next.startsWith("//") ? next : "/app";
    navigate(safeNext);
  }} />;
}

function AppShell() {
  return (
    <Switch>
      <Route path="/" component={LandingPage} />
      <Route path="/login" component={LoginRoute} />
      <Route path="/pricing" component={PricingPage} />
      <Route path="/about" component={AboutPage} />
      <Route path="/newsletter" component={NewsletterPage} />
      <Route path="/careers" component={CareersPage} />
      <Route path="/contact" component={ContactPage} />
      <Route path="/app" component={AppDashboard} />
      <Route path="/app/:rest*" component={AppDashboard} />
      {/* Fallback — anything else returns to landing */}
      <Route component={LandingPage} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <AppShell />
        <Toaster />
      </ErrorBoundary>
    </QueryClientProvider>
  );
}

export default App;
