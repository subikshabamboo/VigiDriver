import { BrowserRouter, Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { Activity, BarChart3, Contact, Cpu, Home, Siren, ShieldCheck } from "lucide-react";
import VigiDriverSite from "./components/VigiDriverSite.jsx";
import LiveMonitoringDashboard from "./components/LiveMonitoringDashboard.jsx";
import ModelInspectionPage from "./components/ModelInspectionPage.jsx";
import AlertNotificationScreen from "./components/AlertNotificationScreen.jsx";
import AnalyticsHistory from "./components/AnalyticsHistory.jsx";
import ContactPage from "./components/ContactPage.jsx";
import { AppPageStyle } from "./lib/ui.jsx";
import { C } from "./lib/api.js";

const NAV_LINKS = [
  { to: "/dashboard", label: "Live Dashboard", icon: <Activity size={15} /> },
  { to: "/models", label: "Model Specs", icon: <Cpu size={15} /> },
  { to: "/alerts", label: "Alerts", icon: <Siren size={15} /> },
  { to: "/analytics", label: "Analytics", icon: <BarChart3 size={15} /> },
  { to: "/contact", label: "Contact", icon: <Contact size={15} /> },
];

function AppNavbar() {
  return (
    <nav
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 24px",
        background: C.bg,
        borderBottom: `1px solid ${C.border}`,
        position: "sticky",
        top: 0,
        zIndex: 50,
        fontFamily: "'Inter', sans-serif",
        flexWrap: "wrap",
        gap: 10,
      }}
    >
      <Link to="/" style={{ display: "flex", alignItems: "center", gap: 9, textDecoration: "none" }}>
        <ShieldCheck size={20} color={C.amber} />
        <span
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 17,
            fontWeight: 700,
            color: C.navy,
            letterSpacing: 0.5,
          }}
        >
          VIGIDRIVER
        </span>
      </Link>

      <div
        style={{
          display: "flex",
          gap: 4,
          background: C.surfaceAlt,
          border: `1px solid ${C.border}`,
          borderRadius: 10,
          padding: 4,
          flexWrap: "wrap",
        }}
      >
        {NAV_LINKS.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            style={({ isActive }) => ({
              display: "flex",
              alignItems: "center",
              gap: 7,
              padding: "9px 16px",
              borderRadius: 8,
              fontSize: 13.5,
              fontWeight: 600,
              textDecoration: "none",
              background: isActive ? C.bg : "transparent",
              color: isActive ? C.navy : C.inkMuted,
              border: isActive ? `1px solid ${C.border}` : "1px solid transparent",
              boxShadow: isActive ? "0 1px 4px rgba(20,40,60,0.08)" : "none",
            })}
          >
            {l.icon}
            {l.label}
          </NavLink>
        ))}
      </div>

      <Link
        to="/"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
          fontSize: 13,
          fontWeight: 600,
          color: C.inkMuted,
          textDecoration: "none",
          border: `1px solid ${C.border}`,
          borderRadius: 8,
          padding: "9px 14px",
        }}
      >
        <Home size={14} /> VigiDriver
      </Link>
    </nav>
  );
}

function AppShell() {
  return (
    <>
      <AppPageStyle />
      <AppNavbar />
      <main style={{ flexGrow: 1, textAlign: "left" }}>
        <Routes>
          <Route path="/dashboard" element={<LiveMonitoringDashboard />} />
          <Route path="/models" element={<ModelInspectionPage />} />
          <Route path="/alerts" element={<AlertNotificationScreen />} />
          <Route path="/analytics" element={<AnalyticsHistory />} />
          <Route path="/contact" element={<ContactPage />} />
          <Route path="*" element={<LiveMonitoringDashboard />} />
        </Routes>
      </main>
    </>
  );
}

function Shell() {
  const location = useLocation();
  const isLanding = location.pathname === "/";
  return isLanding ? (
    <VigiDriverSite />
  ) : (
    <AppShell />
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  );
}
