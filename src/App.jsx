import { useState } from "react";
import VigiDriverSite from "./components/VigiDriverSite";
import ModelSandbox from "./components/ModelSandbox";

function App() {
  const [page, setPage] = useState("site");

  // Inline styles that utilize your global CSS variables
  const navStyles = {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "16px 32px",
    borderBottom: "1px solid var(--border)",
    background: "var(--bg)",
  };

  const logoStyles = {
    fontSize: "20px",
    fontWeight: "600",
    color: "var(--text-h)",
    fontFamily: "var(--heading)",
  };

  const buttonGroupStyles = {
    display: "flex",
    gap: "12px",
  };

  const getButtonStyles = (isActive) => ({
    fontFamily: "var(--sans)",
    fontSize: "15px",
    fontWeight: "500",
    padding: "8px 16px",
    borderRadius: "6px",
    cursor: "pointer",
    transition: "all 0.2s ease",
    border: isActive ? "1px solid var(--accent)" : "1px solid transparent",
    background: isActive ? "var(--accent-bg)" : "transparent",
    color: isActive ? "var(--accent)" : "var(--text)",
  });

  return (
    <>
      <nav style={navStyles}>
        {/* Branding/Logo Area */}
        <div style={logoStyles}>VigiDriver</div>

        {/* Navigation Tabs */}
        <div style={buttonGroupStyles}>
          <button 
            onClick={() => setPage("site")} 
            style={getButtonStyles(page === "site")}
          >
            Site
          </button>
          <button 
            onClick={() => setPage("sandbox")} 
            style={getButtonStyles(page === "sandbox")}
          >
            Model Sandbox
          </button>
        </div>
      </nav>

      {/* Main Content Area */}
      <main style={{ flexGrow: 1 }}>
        {page === "site" ? <VigiDriverSite /> : <ModelSandbox />}
      </main>
    </>
  );
}

export default App;