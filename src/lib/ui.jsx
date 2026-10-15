/* Shared UI primitives for all VigiDriver app pages (components only —
 * constants and the API client live in ./api.js so HMR treats us cleanly). */

import { C } from "./api.js";

export function PageHeader({ eyebrow, title, subtitle, right }) {
  return (
    <div
      style={{
        marginBottom: 20,
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: 14,
      }}
    >
      <div>
        <div
          style={{
            fontSize: 12,
            color: C.amber,
            letterSpacing: 1,
            fontWeight: 600,
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          {eyebrow}
        </div>
        <h1
          style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: 28,
            fontWeight: 700,
            margin: "6px 0 0",
            color: C.ink,
          }}
        >
          {title}
        </h1>
        {subtitle && (
          <p style={{ color: C.inkMuted, fontSize: 14, marginTop: 4 }}>{subtitle}</p>
        )}
      </div>
      {right}
    </div>
  );
}

export function PanelLabel({ icon, text }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.navy }}>
      {icon}
      <span
        style={{
          fontSize: 12,
          fontWeight: 700,
          letterSpacing: 0.3,
          textTransform: "uppercase",
        }}
      >
        {text}
      </span>
    </div>
  );
}

export function Card({ children, style }) {
  return (
    <div
      style={{
        background: C.bg,
        border: `1px solid ${C.border}`,
        borderRadius: 12,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function StatusPill({ color, bg, children, pulse }) {
  return (
    <span
      className={pulse ? "vd-pulse" : undefined}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        background: bg,
        color,
        padding: "9px 18px",
        borderRadius: 999,
        fontWeight: 700,
        fontSize: 13,
        letterSpacing: 0.5,
      }}
    >
      <span
        style={{
          width: 9,
          height: 9,
          borderRadius: "50%",
          background: color,
          display: "inline-block",
        }}
      />
      {children}
    </span>
  );
}

/** Global styles injected once for the app pages (not the landing site). */
export function AppPageStyle() {
  return (
    <style>{`
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
      * { box-sizing: border-box; }
      .vd-root { font-family: 'Inter', sans-serif; }
      .vd-display { font-family: 'Space Grotesk', sans-serif; }
      .vd-mono { font-family: 'JetBrains Mono', monospace; }
      .vd-card { background: #FFFFFF; border: 1px solid #E2E6EC; border-radius: 12px; }
      @keyframes vd-pulse-kf { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
      .vd-pulse { animation: vd-pulse-kf 1s ease-in-out infinite; }
      @keyframes vd-siren-kf { 0%,100% { box-shadow: 0 0 0 0 rgba(196,70,63,0.55); } 50% { box-shadow: 0 0 0 14px rgba(196,70,63,0); } }
      .vd-siren { animation: vd-siren-kf 1.1s ease-out infinite; }
      @keyframes vd-blink-kf { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
      .vd-blinking { animation: vd-blink-kf 0.9s ease-in-out infinite; }
      .vd-link { color: #163A5F; text-decoration: none; font-weight: 600; font-size: 13.5px; }
      .vd-link:hover { text-decoration: underline; }
      .vd-input {
        width: 100%; padding: 10px 13px; border-radius: 8px; border: 1px solid #E2E6EC;
        font-family: 'Inter', sans-serif; font-size: 14px; color: #132133; background: #FCFDFE;
      }
      .vd-input:focus { outline: 2px solid #1F4C79; outline-offset: 1px; }
      input[type="range"].vd-slider {
        -webkit-appearance: none; width: 100%; height: 6px; border-radius: 4px;
        background: linear-gradient(to right, #C57C21 0%, #C57C21 var(--fill, 50%), #E2E6EC var(--fill, 50%), #E2E6EC 100%);
        outline: none;
      }
      input[type="range"].vd-slider::-webkit-slider-thumb {
        -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%;
        background: #fff; border: 3px solid #C57C21; cursor: pointer; box-shadow: 0 1px 4px rgba(0,0,0,0.2);
      }
      input[type="range"].vd-slider::-moz-range-thumb {
        width: 18px; height: 18px; border-radius: 50%; background: #fff; border: 3px solid #C57C21; cursor: pointer;
      }
      table.vd-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
      table.vd-table th { text-align: left; font-size: 11.5px; color: #8A93A2; text-transform: uppercase; letter-spacing: 0.4px; padding: 8px 10px; border-bottom: 1px solid #E2E6EC; }
      table.vd-table td { padding: 11px 10px; border-bottom: 1px solid #E2E6EC; color: #132133; text-align: left; }
      table.vd-table tr:hover td { background: #F4F6F9; }
      @media (max-width: 980px) {
        .vd-grid-monitor { grid-template-columns: 1fr !important; }
        .vd-grid-2 { grid-template-columns: 1fr !important; }
        .vd-form-2 { grid-template-columns: 1fr !important; }
      }
    `}</style>
  );
}
