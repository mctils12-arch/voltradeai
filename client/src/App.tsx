import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import { Route, Switch, useLocation } from "wouter";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import Home from "@/pages/home";
import LoginPage from "@/pages/login";
import LandingPage from "@/pages/landing";
import PricingPage from "@/pages/pricing";
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

/** Standalone login route — redirects to /app on success */
function LoginRoute() {
  const [, navigate] = useLocation();
  return <LoginPage onLogin={() => {
    queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
    navigate("/app");
  }} />;
}

function AppShell() {
  return (
    <Switch>
      <Route path="/" component={LandingPage} />
      <Route path="/login" component={LoginRoute} />
      <Route path="/pricing" component={PricingPage} />
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
