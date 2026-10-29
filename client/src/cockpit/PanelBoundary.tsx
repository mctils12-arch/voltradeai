// PanelBoundary — one panel's render error stays inside that panel.
//
// 2026-10-29: the Trade cockpit renders every feature body at once, so a
// body that throws on an odd payload (a research report without a verdict
// was the first) would reach the app-level boundary and blank the whole
// page. Here the panel shows the error in its own frame with a retry; the
// other panels, the look-up box and the map keep working. `resetKey`
// (the symbol) clears the error when the input that caused it changes.
import { Component, type ReactNode } from "react";

interface Props { resetKey: string; children: ReactNode }
interface State { error: string | null; forKey: string }

export class PanelBoundary extends Component<Props, State> {
  state: State = { error: null, forKey: "" };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error.message : String(error) };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    // a new symbol gets a fresh try; the same symbol keeps the note until retry
    if (state.error && state.forKey !== props.resetKey) return { error: null, forKey: props.resetKey };
    if (state.forKey !== props.resetKey) return { forKey: props.resetKey };
    return null;
  }

  componentDidCatch(error: unknown): void {
    console.error("cockpit panel failed", error);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="vt-bot-empty" role="alert">
          <span className="vt-bot-r">this panel failed to render · {this.state.error}</span>
          <span className="vt-bot-t3">the rest of the page is unaffected</span>
          <button type="button" className="vt-bot-wbtn" onClick={() => this.setState({ error: null })}>try again</button>
        </div>
      );
    }
    return this.props.children;
  }
}
