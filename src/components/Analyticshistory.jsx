// pages/AnalyticsHistory.jsx
import React, { useMemo, useState } from 'react';
import { Users, BarChart3, ClipboardList, ChevronDown } from 'lucide-react';

const C = {
  bg: '#FFFFFF',
  surfaceAlt: '#F4F6F9',
  border: '#E2E6EC',
  ink: '#132133',
  inkMuted: '#5B6572',
  inkDim: '#8A93A2',
  navy: '#163A5F',
  amber: '#C57C21',
  teal: '#1F7A63',
  tealSoft: '#E4F2EC',
  critical: '#C4463F',
  criticalSoft: '#F9E4E2',
  cautionSoft: '#FBEDD5',
  caution: '#946B0C',
};

const SUBJECTS = [
  { id: 'Subject_01', tag: 'Unseen Test', incidents: [3, 1, 4, 0, 2, 1], score: 88 },
  { id: 'Subject_02', tag: 'Train', incidents: [1, 2, 1, 3, 1, 0], score: 93 },
  { id: 'Subject_03', tag: 'Unseen Test', incidents: [5, 4, 6, 3, 5, 4], score: 71 },
  { id: 'Subject_04', tag: 'Train', incidents: [0, 1, 0, 1, 0, 1], score: 97 },
];

const AUDIT_LOG = [
  { session: 'SES-2201', subject: 'Subject_01', clip: 'trip_014_night.mp4', duration: '42:10', label: 'Caution' },
  { session: 'SES-2198', subject: 'Subject_02', clip: 'trip_009_day.mp4', duration: '38:02', label: 'Safe' },
  { session: 'SES-2194', subject: 'Subject_03', clip: 'trip_022_night.mp4', duration: '51:47', label: 'Unsafe' },
  { session: 'SES-2189', subject: 'Subject_01', clip: 'trip_011_dusk.mp4', duration: '29:55', label: 'Safe' },
  { session: 'SES-2183', subject: 'Subject_04', clip: 'trip_006_day.mp4', duration: '33:20', label: 'Safe' },
  { session: 'SES-2177', subject: 'Subject_03', clip: 'trip_018_night.mp4', duration: '47:03', label: 'Caution' },
];

const LABEL_STYLE = {
  Safe: { bg: C.tealSoft, color: C.teal },
  Caution: { bg: C.cautionSoft, color: C.caution },
  Unsafe: { bg: C.criticalSoft, color: C.critical },
};

export default function AnalyticsHistory() {
  const [subjectId, setSubjectId] = useState('Subject_01');
  const subject = useMemo(() => SUBJECTS.find((s) => s.id === subjectId), [subjectId]);
  const rows = useMemo(() => AUDIT_LOG.filter((r) => r.subject === subjectId), [subjectId]);
  const maxIncident = Math.max(...subject.incidents, 1);

  return (
    <div style={{ width: '100%', minHeight: '100vh', background: C.bg, color: C.ink, padding: '36px 24px 60px' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        * { box-sizing: border-box; }
        .ah-root { font-family: 'Inter', sans-serif; }
        .ah-display { font-family: 'Space Grotesk', sans-serif; }
        .ah-mono { font-family: 'JetBrains Mono', monospace; }
        .ah-card { background: ${C.bg}; border: 1px solid ${C.border}; border-radius: 12px; }
        .ah-select {
          appearance: none; background: ${C.bg}; border: 1px solid ${C.border}; border-radius: 8px;
          padding: 10px 34px 10px 14px; font-size: 14px; font-family: 'Inter', sans-serif; color: ${C.ink};
          cursor: pointer;
        }
        table.ah-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
        table.ah-table th { text-align: left; font-size: 11.5px; color: ${C.inkDim}; text-transform: uppercase; letter-spacing: 0.4px; padding: 8px 10px; border-bottom: 1px solid ${C.border}; }
        table.ah-table td { padding: 11px 10px; border-bottom: 1px solid ${C.border}; color: ${C.ink}; }
      `}</style>

      <div className="ah-root" style={{ maxWidth: 1180, margin: '0 auto' }}>
        <div style={{ marginBottom: 22 }}>
          <div className="ah-mono" style={{ fontSize: 12, color: C.amber, letterSpacing: 1, fontWeight: 600 }}>
            SUBJECT-ISOLATED EVALUATION
          </div>
          <h1 className="ah-display" style={{ fontSize: 28, fontWeight: 700, margin: '6px 0 0' }}>
            Analytics &amp; Driver History
          </h1>
          <p style={{ color: C.inkMuted, fontSize: 14, marginTop: 4 }}>
            Cross-validated performance and trip records, per subject.
          </p>
        </div>

        {/* Subject selector */}
        <div
          className="ah-card"
          style={{
            padding: '16px 20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: 18,
            flexWrap: 'wrap',
            gap: 14,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: C.navy }}>
            <Users size={17} />
            <span style={{ fontSize: 13.5, fontWeight: 700 }}>Viewing subject</span>
          </div>
          <div style={{ position: 'relative' }}>
            <select className="ah-select" value={subjectId} onChange={(e) => setSubjectId(e.target.value)}>
              {SUBJECTS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id} [{s.tag}]
                </option>
              ))}
            </select>
            <ChevronDown
              size={15}
              style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: C.inkDim }}
            />
          </div>
          <div style={{ display: 'flex', gap: 24 }}>
            <MiniStat label="Safety score" value={subject.score} />
            <MiniStat label="Split" value={subject.tag} />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: 18, marginBottom: 18 }}>
          {/* bar chart */}
          <div className="ah-card" style={{ padding: 20 }}>
            <PanelLabel icon={<BarChart3 size={15} />} text={`Fatigue incidents per trip — ${subjectId}`} />
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16, height: 160, marginTop: 20, paddingLeft: 6 }}>
              {subject.incidents.map((v, i) => (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, flex: 1 }}>
                  <div className="ah-mono" style={{ fontSize: 11.5, color: C.inkMuted }}>
                    {v}
                  </div>
                  <div
                    style={{
                      width: '100%',
                      maxWidth: 34,
                      borderRadius: '5px 5px 0 0',
                      background: v >= maxIncident * 0.75 ? C.critical : v >= maxIncident * 0.4 ? C.amber : C.teal,
                      height: `${Math.max(6, (v / maxIncident) * 120)}px`,
                      transition: 'height .3s ease',
                    }}
                  />
                  <div style={{ fontSize: 11, color: C.inkDim }}>Trip {i + 1}</div>
                </div>
              ))}
            </div>
          </div>

          {/* summary card */}
          <div className="ah-card" style={{ padding: 20 }}>
            <PanelLabel icon={<Users size={15} />} text="Subject summary" />
            <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <SummaryRow label="Cross-validation split" value={subject.tag} />
              <SummaryRow label="Total incidents (6 trips)" value={subject.incidents.reduce((a, b) => a + b, 0)} />
              <SummaryRow label="Avg incidents / trip" value={(subject.incidents.reduce((a, b) => a + b, 0) / subject.incidents.length).toFixed(1)} />
              <SummaryRow label="Composite safety score" value={`${subject.score} / 100`} />
            </div>
          </div>
        </div>

        {/* audit table */}
        <div className="ah-card" style={{ padding: 20 }}>
          <PanelLabel icon={<ClipboardList size={15} />} text="Historical audit table" />
          <div style={{ marginTop: 14, overflowX: 'auto' }}>
            <table className="ah-table">
              <thead>
                <tr>
                  <th>Session</th>
                  <th>Video clip</th>
                  <th>Duration</th>
                  <th>Classification</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={4} style={{ color: C.inkDim, textAlign: 'center', padding: '24px 0' }}>
                      No sessions recorded for this subject.
                    </td>
                  </tr>
                ) : (
                  rows.map((r) => (
                    <tr key={r.session}>
                      <td className="ah-mono">{r.session}</td>
                      <td className="ah-mono">{r.clip}</td>
                      <td className="ah-mono">{r.duration}</td>
                      <td>
                        <span
                          style={{
                            background: LABEL_STYLE[r.label].bg,
                            color: LABEL_STYLE[r.label].color,
                            padding: '3px 10px',
                            borderRadius: 999,
                            fontSize: 12,
                            fontWeight: 700,
                          }}
                        >
                          {r.label}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function PanelLabel({ icon, text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: C.navy }}>
      {icon}
      <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: 0.3, textTransform: 'uppercase' }}>{text}</span>
    </div>
  );
}

function MiniStat({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: 10.5, color: C.inkDim, textTransform: 'uppercase' }}>{label}</div>
      <div className="ah-mono" style={{ fontSize: 15, fontWeight: 700, color: C.navy }}>
        {value}
      </div>
    </div>
  );
}

function SummaryRow({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: `1px solid ${C.border}`, paddingBottom: 10 }}>
      <span style={{ fontSize: 13, color: C.inkMuted }}>{label}</span>
      <span className="ah-mono" style={{ fontSize: 13, fontWeight: 700 }}>
        {value}
      </span>
    </div>
  );
}