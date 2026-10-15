/* Driver Analytics & Session History.
 *
 * Reads persisted sessions from the backend: the audit table lists every
 * completed monitoring session with its outcome classification; selecting a
 * session reveals its attention-weight trace, per-head model agreement and
 * alert log. The aggregate panel shows the drowsiness-probability
 * distribution across all recorded samples for fleet safety auditing.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  ClipboardList,
  Gauge,
  Layers,
  ShieldCheck,
  Users,
} from "lucide-react";
import { C, apiGet } from "../lib/api.js";
import { Card, PageHeader, PanelLabel } from "../lib/ui.jsx";

const OUTCOME_STYLE = {
  safe: { bg: C.tealSoft, color: C.teal, label: "Safe" },
  caution: { bg: C.cautionSoft, color: C.caution, label: "Caution" },
  unsafe: { bg: C.criticalSoft, color: C.critical, label: "Unsafe" },
  running: { bg: C.surfaceAlt2, color: C.inkMuted, label: "Running" },
};

function fmtDur(s) {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function fmtDate(ts) {
  return new Date(ts * 1000).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function AnalyticsHistory() {
  const [sessions, setSessions] = useState(null);
  const [stats, setStats] = useState(null);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);

  const refresh = useCallback(() => {
    apiGet("/api/sessions?limit=100")
      .then((d) => {
        setSessions(d.sessions || []);
        if (d.sessions?.length && selected == null) setSelected(d.sessions[0].id);
      })
      .catch((e) => setError(e.message));
    apiGet("/api/stats")
      .then(setStats)
      .catch(() => {});
  }, [selected]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    if (selected == null) return;
    apiGet(`/api/sessions/${selected}`)
      .then(setDetail)
      .catch(() => setDetail(null));
  }, [selected]);

  const maxBin = useMemo(
    () => Math.max(1, ...(stats?.attention_distribution || []).map((b) => b.count)),
    [stats]
  );

  return (
    <div className="vd-root" style={{ padding: "32px 24px 60px" }}>
      <PageHeader
        eyebrow="FLEET SAFETY AUDITING"
        title="Driver Analytics & Session History"
        subtitle="Attention-weight distributions and fatigue audit records for every monitored session."
        right={
          stats && (
            <div style={{ display: "flex", gap: 24 }}>
              <MiniStat label="Sessions" value={stats.sessions} />
              <MiniStat label="Warnings" value={stats.warnings} />
              <MiniStat label="Critical" value={stats.criticals} />
              <MiniStat label="Safety score" value={`${stats.safety_score}/100`} />
            </div>
          )
        }
      />

      {error && (
        <div
          style={{
            marginBottom: 16,
            border: `1px solid ${C.critical}44`,
            background: C.criticalSoft,
            color: C.critical,
            borderRadius: 10,
            padding: "12px 16px",
            fontSize: 13.5,
            fontWeight: 600,
          }}
        >
          Backend unreachable ({error}). Start it with <b>python backend/run.py</b>.
        </div>
      )}

      {/* summary strip */}
      {stats && (
        <div className="vd-grid-2" style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 18, marginBottom: 18 }}>
          <Card style={{ padding: 20 }}>
            <PanelLabel icon={<BarChart3 size={15} />} text="Attention-weight distribution — all sessions" />
            <p style={{ fontSize: 12, color: C.inkMuted, margin: "6px 0 12px" }}>
              Histogram of fused drowsiness probability across every analysed frame (20 bins, 0 → 1).
            </p>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 130 }}>
              {(stats.attention_distribution || []).map((b, i) => (
                <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                  <div
                    title={`${b.count} samples in [${b.bin_start.toFixed(2)}, ${b.bin_end.toFixed(2)})`}
                    style={{
                      width: "100%",
                      height: `${Math.max(2, (b.count / maxBin) * 106)}px`,
                      background:
                        b.bin_start >= 0.6 ? C.critical : b.bin_start >= 0.35 ? C.amber : C.teal,
                      borderRadius: "3px 3px 0 0",
                      transition: "height .3s ease",
                    }}
                  />
                </div>
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
              <span className="vd-mono" style={{ fontSize: 10.5, color: C.inkDim }}>0.00 — alert</span>
              <span className="vd-mono" style={{ fontSize: 10.5, color: C.inkDim }}>
                {stats.frames_analyzed.toLocaleString()} frames analysed
              </span>
              <span className="vd-mono" style={{ fontSize: 10.5, color: C.inkDim }}>drowsy — 1.00</span>
            </div>
          </Card>

          <Card style={{ padding: 20 }}>
            <PanelLabel icon={<ShieldCheck size={15} />} text="Outcome breakdown" />
            <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 12 }}>
              {["safe", "caution", "unsafe"].map((k) => {
                const n = stats.outcome_counts?.[k] || 0;
                const total = Math.max(1, (stats.outcome_counts?.safe || 0) + (stats.outcome_counts?.caution || 0) + (stats.outcome_counts?.unsafe || 0));
                return (
                  <div key={k}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 4 }}>
                      <span style={{ color: OUTCOME_STYLE[k].color, fontWeight: 700 }}>{OUTCOME_STYLE[k].label}</span>
                      <span className="vd-mono" style={{ color: C.inkMuted }}>{n} session(s)</span>
                    </div>
                    <div style={{ height: 7, background: C.surfaceAlt2, borderRadius: 4, overflow: "hidden" }}>
                      <div
                        style={{
                          width: `${(n / total) * 100}%`,
                          height: "100%",
                          background: OUTCOME_STYLE[k].color,
                          borderRadius: 4,
                          transition: "width .3s ease",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
              <SummaryRow label="Avg session duration" value={fmtDur(stats.avg_duration_s)} />
              <SummaryRow label="Frames analysed" value={stats.frames_analyzed.toLocaleString()} />
            </div>
          </Card>
        </div>
      )}

      {/* audit table + detail */}
      <div className="vd-grid-2" style={{ display: "grid", gridTemplateColumns: "1.05fr 0.95fr", gap: 18, alignItems: "start" }}>
        <Card style={{ padding: 20 }}>
          <PanelLabel icon={<ClipboardList size={15} />} text="Historical audit table" />
          <div style={{ marginTop: 14, overflowX: "auto" }}>
            <table className="vd-table">
              <thead>
                <tr>
                  <th>Session</th>
                  <th>Driver</th>
                  <th>Started</th>
                  <th>Duration</th>
                  <th>Alerts</th>
                  <th>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {sessions === null && (
                  <tr><td colSpan={6} style={{ color: C.inkDim, textAlign: "center", padding: "22px 0" }}>loading…</td></tr>
                )}
                {sessions?.length === 0 && (
                  <tr>
                    <td colSpan={6} style={{ color: C.inkDim, textAlign: "center", padding: "22px 0" }}>
                      No sessions recorded yet — run a live monitoring session first.
                    </td>
                  </tr>
                )}
                {sessions?.map((s) => {
                  const st = s.ended_at ? OUTCOME_STYLE[s.outcome] : OUTCOME_STYLE.running;
                  const isSel = selected === s.id;
                  return (
                    <tr
                      key={s.id}
                      onClick={() => setSelected(s.id)}
                      style={{ cursor: "pointer", background: isSel ? C.surfaceAlt : "transparent" }}
                    >
                      <td className="vd-mono">#{s.id}</td>
                      <td>{s.driver_label}</td>
                      <td className="vd-mono" style={{ fontSize: 12 }}>{fmtDate(s.started_at)}</td>
                      <td className="vd-mono">{s.ended_at ? fmtDur(s.duration_s) : "live"}</td>
                      <td className="vd-mono">
                        {s.alert_count + s.critical_count > 0 ? (
                          <>
                            {s.critical_count > 0 && <span style={{ color: C.critical, fontWeight: 700 }}>{s.critical_count}✕ </span>}
                            {s.alert_count > 0 && <span style={{ color: C.caution }}>{s.alert_count}⚠</span>}
                          </>
                        ) : (
                          "0"
                        )}
                      </td>
                      <td>
                        <span
                          style={{
                            background: st.bg,
                            color: st.color,
                            padding: "3px 10px",
                            borderRadius: 999,
                            fontSize: 12,
                            fontWeight: 700,
                          }}
                        >
                          {st.label}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <Card style={{ padding: 20 }}>
            <PanelLabel icon={<Activity size={15} />} text={`Session #${selected ?? "—"} — attention trace`} />
            {!detail && (
              <div style={{ marginTop: 14, color: C.inkDim, fontSize: 13 }}>
                Select a completed session from the audit table.
              </div>
            )}
            {detail?.session?.attention_hist?.length > 1 ? (
              <>
                <TraceGraph data={detail.session.attention_hist} />
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 14 }}>
                  <SummaryRow label="Avg drowsiness" value={(detail.session.avg_drowsiness ?? 0).toFixed(3)} />
                  <SummaryRow label="Peak drowsiness" value={(detail.session.peak_drowsiness ?? 0).toFixed(3)} />
                  <SummaryRow label="Frames analysed" value={detail.session.frames_total} />
                  <SummaryRow label="Above threshold" value={`${detail.session.frames_drowsy} (${detail.session.frames_total ? Math.round((100 * detail.session.frames_drowsy) / detail.session.frames_total) : 0}%)`} />
                </div>
              </>
            ) : detail?.session ? (
              <div style={{ marginTop: 14, color: C.inkDim, fontSize: 13 }}>
                {detail.session.ended_at
                  ? "This session stored no attention samples (no face was detected)."
                  : "Session still running — the trace is written when it ends."}
              </div>
            ) : null}
          </Card>

          {detail?.session && (
            <Card style={{ padding: 20 }}>
              <PanelLabel icon={<Users size={15} />} text="Session record" />
              <div style={{ marginTop: 14 }}>
                <SummaryRow label="Driver" value={detail.session.driver_label} />
                <SummaryRow label="Model" value={detail.session.model_used} />
                <SummaryRow label="Threshold" value={detail.session.threshold?.toFixed(2)} />
                <SummaryRow
                  label="Outcome"
                  value={detail.session.ended_at ? OUTCOME_STYLE[detail.session.outcome].label : "running"}
                />
              </div>
              {detail.alerts?.length > 0 && (
                <>
                  <div style={{ marginTop: 16, marginBottom: 8, display: "flex", alignItems: "center", gap: 8, color: C.navy }}>
                    <Layers size={14} />
                    <span style={{ fontSize: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.3 }}>Alert log</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 170, overflowY: "auto" }}>
                    {detail.alerts.map((a) => (
                      <div key={a.id} style={{ display: "flex", gap: 10, fontSize: 12, alignItems: "center" }}>
                        <span className="vd-mono" style={{ color: C.inkDim }}>{fmtDate(a.ts)}</span>
                        <span
                          style={{
                            background: a.tier === "critical" ? C.criticalSoft : C.cautionSoft,
                            color: a.tier === "critical" ? C.critical : C.caution,
                            borderRadius: 999,
                            padding: "1px 8px",
                            fontSize: 11,
                            fontWeight: 700,
                          }}
                        >
                          {a.tier}
                        </span>
                        <span className="vd-mono" style={{ color: C.inkMuted }}>p={a.p_drowsy.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </Card>
          )}

          {stats && (
            <Card style={{ padding: "14px 16px", display: "flex", gap: 10, alignItems: "center", background: C.surfaceAlt }}>
              <Gauge size={16} color={C.navy} />
              <span style={{ fontSize: 12.5, color: C.inkMuted }}>
                Composite fleet safety score: <b className="vd-mono" style={{ color: C.navy }}>{stats.safety_score}/100</b>{" "}
                (safe = 1.0, caution = 0.5, unsafe = 0 weighting).
              </span>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------------- helpers ---------------- */

function MiniStat({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: 10.5, color: C.inkDim, textTransform: "uppercase", letterSpacing: 0.4 }}>{label}</div>
      <div className="vd-mono" style={{ fontSize: 16, fontWeight: 700, color: C.navy }}>{value}</div>
    </div>
  );
}

function SummaryRow({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", borderBottom: `1px solid ${C.border}`, paddingBottom: 8, marginBottom: 8 }}>
      <span style={{ fontSize: 12.5, color: C.inkMuted }}>{label}</span>
      <span className="vd-mono" style={{ fontSize: 12.5, fontWeight: 700, color: C.ink }}>{value}</span>
    </div>
  );
}

function TraceGraph({ data }) {
  const w = 420;
  const h = 110;
  const pts = data
    .map((v, i) => {
      const x = (i / Math.max(1, data.length - 1)) * w;
      const y = h - Math.max(0, Math.min(1, v)) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ marginTop: 12 }}>
      <line x1="0" y1={h * 0.38} x2={w} y2={h * 0.38} stroke={C.critical} strokeOpacity="0.4" strokeDasharray="4 5" />
      <polygon points={`0,${h} ${pts} ${w},${h}`} fill={C.navy} fillOpacity="0.08" />
      <polyline points={pts} fill="none" stroke={C.navy} strokeWidth="2" />
    </svg>
  );
}
