// pages/LiveMonitoringDashboard.jsx
import React, { useEffect, useRef, useState } from 'react';
import { Video, Activity, Timer, Layers, Radio, AlertTriangle } from 'lucide-react';

const C = {
  bg: '#FFFFFF',
  surfaceAlt: '#F4F6F9',
  surfaceAlt2: '#EEF1F6',
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
  console: '#0E1620',
  consoleBorder: '#1F2C3A',
};

function nowClock() {
  return new Date().toTimeString().slice(0, 8);
}

function formatDuration(sec) {
  const m = Math.floor(sec / 60)
    .toString()
    .padStart(2, '0');
  const s = Math.floor(sec % 60)
    .toString()
    .padStart(2, '0');
  return `${m}:${s}`;
}

export default function LiveMonitoringDashboard() {
  const [attention, setAttention] = useState(() => Array(30).fill(0.92));
  const [confidence, setConfidence] = useState(0.94);
  const [alertState, setAlertState] = useState('normal');
  const [fps, setFps] = useState(29);
  const [windowId, setWindowId] = useState(1042);
  const [cmosStatus, setCmosStatus] = useState('Nominal');
  const [seconds, setSeconds] = useState(0);
  const [logs, setLogs] = useState([
    { id: 1, time: nowClock(), text: 'Session initialized — Subject_01 stream connected.' },
  ]);

  const tickRef = useRef(0);
  const sustainedLowRef = useRef(0);
  const logIdRef = useRef(2);

  function pushLog(text) {
    setLogs((prev) => [{ id: logIdRef.current++, time: nowClock(), text }, ...prev].slice(0, 8));
  }

  useEffect(() => {
    const id = setInterval(() => {
      tickRef.current += 1;
      const n = tickRef.current;
      setSeconds((s) => s + 1);
      setWindowId((w) => w + 1);
      setFps(+(27 + Math.random() * 3.5).toFixed(1));
      setCmosStatus(Math.random() > 0.94 ? 'Denoising active' : 'Nominal');

      const dipPhase = n % 40;
      const inDip = dipPhase >= 26 && dipPhase <= 35;
      const dipAmt = inDip ? Math.sin(((dipPhase - 26) / 9) * Math.PI) * 0.62 : 0;
      const nextConf = Math.max(0.05, Math.min(0.99, 0.93 - dipAmt + (Math.random() - 0.5) * 0.03));

      setConfidence(nextConf);
      setAttention((a) => [...a.slice(1), nextConf]);

      if (nextConf < 0.4) {
        sustainedLowRef.current += 1;
      } else {
        sustainedLowRef.current = 0;
      }

      if (sustainedLowRef.current >= 5 && alertState !== 'drowsy') {
        setAlertState('drowsy');
        pushLog('Sustained eye closure detected over 30 frames — DROWSY ALERT DETECTED.');
      } else if (sustainedLowRef.current === 0 && alertState === 'drowsy') {
        setAlertState('normal');
        pushLog('Alertness restored — returning to baseline monitoring.');
      }
    }, 550);
    return () => clearInterval(id);
  }, [alertState]);

  const isDrowsy = alertState === 'drowsy';

  return (
    <div style={{ width: '100%', minHeight: '100vh', background: C.bg, color: C.ink, padding: '36px 24px 60px' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        * { box-sizing: border-box; }
        .lmd-root { font-family: 'Inter', sans-serif; }
        .lmd-display { font-family: 'Space Grotesk', sans-serif; }
        .lmd-mono { font-family: 'JetBrains Mono', monospace; }
        .lmd-card { background: ${C.bg}; border: 1px solid ${C.border}; border-radius: 12px; }
        @keyframes lmd-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
        .lmd-blinking { animation: lmd-blink 0.9s ease-in-out infinite; }
        .lmd-grid { display: grid; grid-template-columns: 1.05fr 1.15fr 0.95fr; gap: 18px; align-items: start; }
        @media (max-width: 980px) {
          .lmd-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>

      <div className="lmd-root" style={{ maxWidth: 1240, margin: '0 auto' }}>
        <div style={{ marginBottom: 20 }}>
          <div className="lmd-mono" style={{ fontSize: 12, color: C.amber, letterSpacing: 1, fontWeight: 600 }}>
            CORE PRODUCT VIEW
          </div>
          <h1 className="lmd-display" style={{ fontSize: 28, fontWeight: 700, margin: '6px 0 0' }}>
            Live Monitoring Dashboard
          </h1>
          <p style={{ color: C.inkMuted, fontSize: 14, marginTop: 4 }}>
            Real-time execution of the CNN-LSTM-Attention drowsiness model.
          </p>
        </div>

        {/* Status banner */}
        <div
          className={isDrowsy ? 'lmd-blinking' : ''}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            background: isDrowsy ? C.criticalSoft : C.tealSoft,
            color: isDrowsy ? C.critical : C.teal,
            border: `1px solid ${isDrowsy ? C.critical : C.teal}33`,
            borderRadius: 10,
            padding: '12px 18px',
            fontWeight: 700,
            fontSize: 14.5,
            marginBottom: 18,
          }}
        >
          <AlertTriangle size={18} />
          {isDrowsy ? '🚨 DROWSY ALERT DETECTED' : '✅ MONITORING NORMAL — ALERT STANDBY'}
        </div>

        <div className="lmd-grid">
          {/* LEFT: live feed */}
          <div className="lmd-card" style={{ padding: 16 }}>
            <PanelLabel icon={<Video size={15} />} text="Cabin feed — Subject_01" />
            <div
              style={{
                marginTop: 12,
                position: 'relative',
                borderRadius: 8,
                overflow: 'hidden',
                background: '#F0F2F5',
                border: `1px solid ${C.border}`,
                height: 280,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <FeedIllustration color={isDrowsy ? C.critical : C.navy} drowsy={isDrowsy} />
              <div
                style={{
                  position: 'absolute',
                  top: 10,
                  left: 10,
                  background: 'rgba(255,255,255,0.9)',
                  border: `1px solid ${C.border}`,
                  borderRadius: 6,
                  padding: '3px 8px',
                }}
              >
                <span className="lmd-mono" style={{ fontSize: 11, color: C.ink }}>
                  Subject_01 · {confidence.toFixed(2)}
                </span>
              </div>
              <div className="lmd-mono" style={{ position: 'absolute', bottom: 10, right: 12, fontSize: 10.5, color: C.inkDim }}>
                {nowClock()}
              </div>
            </div>
          </div>

          {/* CENTER: attention graph + stats */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <div className="lmd-card" style={{ padding: 18 }}>
              <PanelLabel icon={<Activity size={15} />} text="Temporal attention (rolling 30-frame window)" />
              <div style={{ marginTop: 12 }}>
                <AttentionGraph data={attention} color={isDrowsy ? C.critical : C.navy} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
                <span style={{ fontSize: 12, color: C.inkMuted }}>Confidence score</span>
                <span
                  className="lmd-mono"
                  style={{ fontSize: 13, fontWeight: 700, color: isDrowsy ? C.critical : C.navy }}
                >
                  {confidence.toFixed(2)}
                </span>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
              <StatCard icon={<Radio size={15} />} label="Current FPS" value={fps.toFixed(1)} />
              <StatCard icon={<Layers size={15} />} label="Active window ID" value={`#${windowId}`} />
              <StatCard icon={<Activity size={15} />} label="CMOS noise sim." value={cmosStatus} small />
              <StatCard icon={<Timer size={15} />} label="Session duration" value={formatDuration(seconds)} />
            </div>
          </div>

          {/* RIGHT: log feed */}
          <div className="lmd-card" style={{ padding: 18 }}>
            <PanelLabel icon={<AlertTriangle size={15} />} text="Emergency override / log feed" />
            <div
              className="lmd-mono"
              style={{
                marginTop: 12,
                background: C.console,
                border: `1px solid ${C.consoleBorder}`,
                borderRadius: 8,
                padding: '12px 14px',
                height: 300,
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
              }}
            >
              {logs.map((l) => (
                <div key={l.id} style={{ fontSize: 12, lineHeight: 1.5 }}>
                  <span style={{ color: '#5A6B80' }}>{l.time} — </span>
                  <span style={{ color: l.text.includes('DROWSY') ? '#FF8A80' : '#B7C4D4' }}>{l.text}</span>
                </div>
              ))}
            </div>
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

function StatCard({ icon, label, value, small }) {
  return (
    <div style={{ background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 9, padding: '13px 15px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: C.inkMuted, fontSize: 11.5 }}>
        {icon}
        {label}
      </div>
      <div className="lmd-mono" style={{ fontSize: small ? 15 : 20, fontWeight: 600, color: C.ink, marginTop: 4 }}>
        {value}
      </div>
    </div>
  );
}

function AttentionGraph({ data, color }) {
  const w = 420,
    h = 90;
  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - Math.max(0, Math.min(1, v)) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  const areaPts = `0,${h} ${pts} ${w},${h}`;
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <polygon points={areaPts} fill={color} fillOpacity="0.08" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2.2" />
      <line x1="0" y1={h * 0.6} x2={w} y2={h * 0.6} stroke={C.border} strokeDasharray="3 5" />
    </svg>
  );
}

function FeedIllustration({ color, drowsy }) {
  return (
    <svg width="220" height="240" viewBox="0 0 220 240">
      {/* bounding box */}
      <rect x="55" y="35" width="110" height="150" fill="none" stroke={color} strokeWidth="1.6" strokeDasharray={drowsy ? '5 4' : '0'} />
      {[
        [55, 35],
        [165, 35],
        [55, 185],
        [165, 185],
      ].map(([x, y], i) => (
        <g key={i}>
          <line x1={x} y1={y} x2={x + (x < 110 ? 14 : -14)} y2={y} stroke={color} strokeWidth="3" />
          <line x1={x} y1={y} x2={x} y2={y + (y < 110 ? 14 : -14)} stroke={color} strokeWidth="3" />
        </g>
      ))}
      {/* face */}
      <ellipse cx="110" cy="110" rx="42" ry="56" fill="none" stroke={color} strokeOpacity="0.7" strokeWidth="1.4" />
      <ellipse cx="90" cy="100" rx="8" ry={drowsy ? 1.2 : 4} fill="none" stroke={color} strokeWidth="1.6" />
      <ellipse cx="130" cy="100" rx="8" ry={drowsy ? 1.2 : 4} fill="none" stroke={color} strokeWidth="1.6" />
      <ellipse cx="110" cy="135" rx={drowsy ? 11 : 7} ry={drowsy ? 7 : 3.5} fill="none" stroke={color} strokeWidth="1.6" />
      <text x="60" y="200" fontFamily="JetBrains Mono, monospace" fontSize="10" fill={color}>
        bbox: 55,35,110,150
      </text>
    </svg>
  );
}