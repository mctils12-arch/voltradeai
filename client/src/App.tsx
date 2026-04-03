import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Toaster } from "@/components/ui/toaster";
import Home from "@/pages/home";
import LoginPage from "@/pages/login";
import { Component, ReactNode } from "react";

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
        <div style={{ padding: 32, color: '#f43f5e', fontFamily: 'monospace', background: '#0d1117', minHeight: '100vh' }}>
          <h2>Something went wrong.</h2>
          <pre style={{ fontSize: 12, opacity: 0.7, whiteSpace: 'pre-wrap' }}>{this.state.error}</pre>
          <button onClick={() => this.setState({ error: null })} style={{ marginTop: 16, padding: '8px 16px', background: '#1e40af', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer' }}>Try Again</button>
        </div>
      );
    }
    return this.props.children;
  }
}

function AuthGate() {
  const { data, isLoading } = useQuery({
    queryKey: ["/api/auth/me"],
    queryFn: async () => {
      const r = await apiRequest("GET", "/api/auth/me");
      return r.json();
    },
    staleTime: 60000,
    retry: false,
  });

  if (isLoading) {
    return (
      <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#000", color: "#6e6e73" }}>
        Loading...
      </div>
    );
  }

  if (!data?.authenticated) {
    return <LoginPage onLogin={() => queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] })} />;
  }

  return <Home />;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <AuthGate />
        <Toaster />
      </ErrorBoundary>
    </QueryClientProvider>
  );
}

export default App;
