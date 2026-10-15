/* Alert & Warning Notification Screen.
 *
 * Listens to the live monitoring state and escalates to a high-priority
 * visual + auditory banner when the backend reports a critical fatigue
 * cascade. Also shows the escalation ladder, current session alerts, and the
 * recent alert history persisted by the backend.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  BellOff,
  BellRing,
  CheckCircle2,
  History,
  Siren,
  ShieldCheck,
  Volume2,
  Zap,
} from "lucide-react";
import { C, apiGet } from "../lib/api.js";
import { Card, PageHeader, PanelLabel, StatusPill } from "../lib/ui.jsx";
import {
  notifyDesktop,
  playWarningChime,
  speak,
  startCriticalSiren,
  stopAllAudio,
  unlockAudio,
} from "../lib/alertAudio.js";

const LADDER = [
  {
    tier: "normal",
    title: "Level 1 — Baseline",
    text: "Driver classified alert. Passive monitoring only; no interruption.",
    color: C.teal,
    bg: C.tealSoft,
    icon: <ShieldCheck size={18} />,
  },
  {
    tier: "warning",
    title: "Level 2 — Warning",
    text: "Drowsiness probability above threshold for a sustained burst. Soft chime + visual cue.",
    color: C.caution,
    bg: C.cautionSoft,
    icon: <Zap size={18} />,
  },
  {
    tier: "critical",
    title: "Level 3 — Critical cascade",
    text: "Sustained fatigue cascade. Full-screen siren banner + spoken instruction.",
    color: C.critical,
    bg: C.criticalSoft,
    icon: <Siren size={18} />,
  },
];

export default function AlertNotificationScreen() {
  const [state, setState] = useState(null);
  const [backendOnline, setBackendOnline] = useState(null);
  const [muted, setMuted] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [testing, setTesting] = useState(false);
  const [demoCritical, setDemoCritical] = useState(false);
  const prevTier = useRef("normal");
  const ackedForRef = useRef(null);

  /* poll live state */
  useEffect(() => {
    let alive = true;
    let timer = null;
    const poll = async () => {
      try {
        const d = await apiGet("/api/monitor/state");
        if (!alive) return;
        setState(d);
        setBackendOnline(true);
      } catch {
        if (alive) setBackendOnline(false);
      } finally {
        if (alive) timer = setTimeout(poll, 300);
      }
    };
    poll();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, []);

  /* alert history from the audit trail */
  const refreshAlerts = useCallback(() => {
    apiGet("/api/sessions?limit=5")
      .then((d) => {
        const alerts = [];
        for (const s of d.sessions || []) {
          alerts.push({
            id: `${s.id}`,
            session_id: s.id,
            when: s.started_at,
            tier: s.critical_count > 0 ? "critical" : s.alert_count > 0 ? "warning" : "normal",
            text:
              s.critical_count > 0
                ? `Session #${s.id} (${s.driver_label}) — ${s.critical_count} critical cascade(s)`
                : s.alert_count > 0
                  ? `Session #${s.id} (${s.driver_label}) — ${s.alert_count} warning(s)`
                  : `Session #${s.id} (${s.driver_label}) — clean, no alerts`,
          });
        }
        setRecentAlerts(alerts.slice(0, 6));
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshAlerts();
    const t = setInterval(refreshAlerts, 5000);
    return () => clearInterval(t);
  }, [refreshAlerts]);

  const tier = state?.running ? state.tier || "normal" : "standby";
  const isCritical = tier === "critical" || demoCritical;
  const isWarning = !isCritical && tier === "warning";

  /* tier transitions -> audio/visual escalation */
  useEffect(() => {
    if (tier === prevTier.current) return;
    const from = prevTier.current;
    prevTier.current = tier;
    if (tier === "critical" && from !== "critical") {
      setAcknowledged(false);
      ackedForRef.current = null;
      if (!muted) {
        startCriticalSiren();
        speak("Danger. Critical drowsiness detected. Stop the vehicle safely.");
      }
      notifyDesktop("VigiDriver — CRITICAL ALERT", "Fatigue cascade detected. Pull over now.");
    } else if (tier === "warning" && from === "normal" && !muted) {
      playWarningChime();
    } else if ((tier === "normal" || tier === "standby") && from !== "normal" && from !== "standby") {
      stopAllAudio();
    }
  }, [tier, muted]);

  useEffect(() => () => stopAllAudio(), []);

  const toggleMute = () => {
    setMuted((m) => {
      if (!m) stopAllAudio();
      else if (tier === "critical") startCriticalSiren();
      return !m;
    });
    unlockAudio();
  };

  const acknowledge = () => {
    setAcknowledged(true);
    stopAllAudio();
  };

  /* demo trigger so the escalation can be verified without a drowsy driver */
  const testCascade = async () => {
    setTesting(true);
    setDemoCritical(true);
    unlockAudio();
    prevTier.current = "normal";
    if (!muted) {
      startCriticalSiren();
      speak("Demonstration. This is the critical drowsiness alert.");
    }
    notifyDesktop("VigiDriver — TEST", "Critical alert demonstration banner.");
    setTimeout(() => setTesting(false), 4000);
    setTimeout(() => {
      setDemoCritical(false);
      stopAllAudio();
      prevTier.current = tier;
    }, 4200);
  };

  return (
    <div className="vd-root" style={{ padding: "32px 24px 60px" }}>
      <PageHeader
        eyebrow="SAFETY SUPERVISION"
        title="Alert & Warning Notifications"
        subtitle="Escalation engine for fatigue cascades — visual, audible and desktop-level."
        right={
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <StatusPill
              color={backendOnline ? C.teal : C.critical}
              bg={backendOnline ? C.tealSoft : C.criticalSoft}
            >
              {backendOnline ? "LISTENING" : "OFFLINE"}
            </StatusPill>
            <button
              onClick={toggleMute}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 7,
                border: `1px solid ${C.border}`,
                background: C.bg,
                borderRadius: 8,
                padding: "9px 14px",
                cursor: "pointer",
                fontWeight: 600,
                fontSize: 13,
                color: muted ? C.critical : C.inkMuted,
              }}
            >
              {muted ? <BellOff size={15} /> : <BellRing size={15} />}
              {muted ? "Alerts muted" : "Alerts armed"}
            </button>
          </div>
        }
      />

      {/* ============ HIGH-PRIORITY BANNER ============ */}
      {isCritical ? (
        <div
          className={acknowledged ? "" : "vd-blinking vd-siren"}
          role="alert"
          style={{
            borderRadius: 14,
            padding: "34px 30px",
            marginBottom: 20,
            background: acknowledged ? C.criticalSoft : C.critical,
            color: acknowledged ? C.critical : "#FFFFFF",
            border: `2px solid ${C.critical}`,
            display: "flex",
            alignItems: "center",
            gap: 22,
            boxShadow: acknowledged ? "none" : "0 18px 44px rgba(196,70,63,0.35)",
          }}
        >
          <Siren size={54} />
          <div style={{ flex: 1 }}>
            <div
              className="vd-display"
              style={{ fontSize: 26, fontWeight: 700, letterSpacing: 0.4 }}
            >
              {acknowledged ? "CRITICAL ALERT — ACKNOWLEDGED" : "CRITICAL DROWSINESS ALERT"}
            </div>
            <div style={{ fontSize: 14.5, marginTop: 6, opacity: 0.92, lineHeight: 1.55 }}>
              A sustained fatigue cascade was detected at p = {(state?.p_drowsy ?? 0).toFixed(2)}.
              {acknowledged
                ? " Audio silenced. Re-escalates if the cascade continues."
                : " Pull over as soon as it is safe and rest before continuing."}
            </div>
          </div>
          {!acknowledged && (
            <button
              onClick={acknowledge}
              style={{
                background: "#FFFFFF",
                color: C.critical,
                border: "none",
                borderRadius: 10,
                padding: "14px 24px",
                fontWeight: 700,
                fontSize: 15,
                cursor: "pointer",
                whiteSpace: "nowrap",
              }}
            >
              Acknowledge
            </button>
          )}
        </div>
      ) : isWarning ? (
        <div
          role="status"
          style={{
            borderRadius: 12,
            padding: "20px 24px",
            marginBottom: 20,
            background: C.cautionSoft,
            color: C.caution,
            border: `1.5px solid ${C.caution}55`,
            display: "flex",
            alignItems: "center",
            gap: 14,
            fontWeight: 700,
            fontSize: 15.5,
          }}
        >
          <AlertTriangle size={22} />
          WARNING — drowsiness indicators building (p = {(state?.p_drowsy ?? 0).toFixed(2)}).
          Critical escalation triggers if the cascade sustains.
        </div>
      ) : (
        <div
          style={{
            borderRadius: 12,
            padding: "20px 24px",
            marginBottom: 20,
            background: C.surfaceAlt,
            color: C.inkMuted,
            border: `1px solid ${C.border}`,
            display: "flex",
            alignItems: "center",
            gap: 14,
            fontSize: 14.5,
          }}
        >
          <CheckCircle2 size={20} color={C.teal} />
          {tier === "standby"
            ? "No monitoring session is running. Start one from the Live Dashboard — this screen escalates automatically."
            : "All clear — the driver is classified alert and no cascade is in progress."}
        </div>
      )}

      {/* ============ ESCALATION LADDER ============ */}
      <Card style={{ padding: 20, marginBottom: 18 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <PanelLabel icon={<Zap size={15} />} text="Tiered alert escalation" />
          <button
            onClick={testCascade}
            disabled={testing}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 7,
              background: C.bg,
              border: `1px solid ${C.border}`,
              borderRadius: 8,
              padding: "8px 14px",
              cursor: "pointer",
              fontSize: 12.5,
              fontWeight: 700,
              color: C.navy,
            }}
          >
            <Volume2 size={14} /> {testing ? "Test running…" : "Test critical banner"}
          </button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14, marginTop: 16 }} className="vd-grid-2">
          {LADDER.map((l) => {
            const active = tier === l.tier || (l.tier === "critical" && isCritical);
            return (
              <div
                key={l.tier}
                style={{
                  border: `1.5px solid ${active ? l.color : C.border}`,
                  background: active ? l.bg : C.bg,
                  borderRadius: 10,
                  padding: "16px 18px",
                  opacity: active ? 1 : 0.72,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 9, color: active ? l.color : C.inkMuted }}>
                  {l.icon}
                  <span style={{ fontWeight: 700, fontSize: 13.5 }}>{l.title}</span>
                  {active && (
                    <span
                      className="vd-mono"
                      style={{
                        marginLeft: "auto",
                        fontSize: 10.5,
                        background: l.color,
                        color: "#fff",
                        borderRadius: 999,
                        padding: "2px 9px",
                        fontWeight: 700,
                      }}
                    >
                      ACTIVE
                    </span>
                  )}
                </div>
                <p style={{ fontSize: 12.5, color: C.inkMuted, marginTop: 8, lineHeight: 1.55 }}>{l.text}</p>
              </div>
            );
          })}
        </div>
      </Card>

      <div className="vd-grid-2" style={{ display: "grid", gridTemplateColumns: "1.05fr 0.95fr", gap: 18 }}>
        {/* live events */}
        <Card style={{ padding: 20 }}>
          <PanelLabel icon={<Siren size={15} />} text="Live session alert events" />
          <div
            className="vd-mono"
            style={{
              marginTop: 12,
              background: C.console,
              border: `1px solid ${C.consoleBorder}`,
              borderRadius: 8,
              padding: "12px 14px",
              height: 240,
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              gap: 8,
            }}
          >
            {!state?.running && (
              <div style={{ fontSize: 12, color: "#5A6B80" }}>
                {new Date().toTimeString().slice(0, 8)} — standby. Alerts stream here during a live session.
              </div>
            )}
            {(state?.events || []).slice().reverse().map((e, i) => (
              <div key={i} style={{ fontSize: 12, lineHeight: 1.5 }}>
                <span style={{ color: "#5A6B80" }}>{new Date(e.ts * 1000).toTimeString().slice(0, 8)} — </span>
                <span style={{ color: e.tier === "critical" ? "#FF8A80" : "#F2C245" }}>{e.text}</span>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 20, marginTop: 12, fontSize: 12.5, color: C.inkMuted }}>
            <span>Threshold: <b className="vd-mono">{state?.threshold?.toFixed(2) ?? "—"}</b></span>
            <span>p (fused): <b className="vd-mono">{state?.running ? (state?.p_drowsy ?? 0).toFixed(3) : "—"}</b></span>
            <span>Tier: <b className="vd-mono">{tier}</b></span>
          </div>
        </Card>

        {/* recent audit alerts */}
        <Card style={{ padding: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <PanelLabel icon={<History size={15} />} text="Recent sessions — audit summary" />
            <Link to="/analytics" className="vd-link">Full history →</Link>
          </div>
          <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
            {recentAlerts.length === 0 && (
              <div style={{ fontSize: 13, color: C.inkDim }}>
                No sessions recorded yet. Completed sessions appear here with their alert totals.
              </div>
            )}
            {recentAlerts.map((a) => (
              <div
                key={a.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  border: `1px solid ${C.border}`,
                  borderRadius: 9,
                  padding: "11px 14px",
                }}
              >
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    background: a.tier === "critical" ? C.critical : a.tier === "warning" ? C.amber : C.teal,
                    flexShrink: 0,
                  }}
                />
                <span style={{ fontSize: 13, color: C.ink, flex: 1 }}>{a.text}</span>
                <span className="vd-mono" style={{ fontSize: 11.5, color: C.inkDim }}>
                  {new Date(a.when * 1000).toLocaleDateString()}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
