import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { Playground } from "./Playground";
import { Demo } from "./Demo";
import "./styles.css";

function App() {
  const [tab, setTab] = useState<"playground" | "demo">("playground");
  return (
    <div className="app">
      <header>
        <h1>SafeSpace</h1>
        <nav>
          <button className={tab === "playground" ? "on" : ""} onClick={() => setTab("playground")}>
            Fusion playground
          </button>
          <button className={tab === "demo" ? "on" : ""} onClick={() => setTab("demo")}>
            Live check-in
          </button>
        </nav>
      </header>
      <p className="disclaimer">
        Research demo. Not a medical or diagnostic tool. Nothing you enter is stored.
      </p>
      {tab === "playground" ? <Playground /> : <Demo />}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
