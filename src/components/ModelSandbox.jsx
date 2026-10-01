// pages/ModelSandbox.jsx
import React, { useEffect, useRef, useState } from 'react';
import {
  Cpu,
  UploadCloud,
  FileCheck2,
  Activity,
  Gauge,
  Terminal,
  CheckCircle2,
  Circle,
  Loader2,
  Wifi,
  Sliders,
} from 'lucide-react';

const C = {
  bg: '#FFFFFF',
  surfaceAlt: '#F4F6F9',
  surfaceAlt2: '#EEF1F6',
  border: '#E2E6EC',
  ink: '#132133',
  inkMuted: '#5B6572',
  inkDim: '#8A93A2',
  navy: '#163A5F',
  navySoft: '#1F4C79',
  amber: '#C57C21',
  amberSoft: '#FBEDD5',
  teal: '#1F7A63',
  tealSoft: '#E4F2EC',
  critical: '#C4463F',
  criticalSoft: '#F9E4E2',
  console: '#0E1620',
  consoleBorder: '#1F2C3A',
};

const STATUS_META = {
  idle: { label: 'IDLE', color: C.inkDim, bg: '#EDEFF2' },
  validating: { label: 'VALIDATING', color: '#946B0C', bg: '#FBF0D6' },
  ready: { label: 'READY', color: C.teal, bg: C.tealSoft },
};

function nowClock() {
  const d = new Date();
  return d.toTimeString().slice(0, 8);
}

export default function ModelSandbox() {
  const [status, setStatus] = useState('idle');
  const [fileName, setFileName] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [checks, setChecks] = useState({ features: false, arch: false, shape: false });

  const [threshold, setThreshold] = useState(0.85);
  const [latency, setLatency] = useState(14);
  const [cpu, setCpu] = useState(38);
  const [ram, setRam] = useState(46);

  const [earL, setEarL] = useState(0.28);
  const [earR, setEarR] = useState(0.27);
  const [mar, setMar] = useState(0.18);
  const [frameCount, setFrameCount] = useState(30);
  const [earHistory, setEarHistory] = useState(Array(40).fill(0.28));
  const [marHistory, setMarHistory] = useState(Array(40).fill(0.18));

  const [logs, setLogs] = useState([
    { id: 1, time: '10:51:22', level: 'info', text: 'Model v1.2 loaded successfully.' },
    { id: 2, time: '10:51:24', level: 'info', text: 'Preprocessing pipeline: YUV-CLAHE active.' },
    { id: 3, time: '10:51:25', level: 'info', text: 'WebSocket: connection established.' },
  ]);

  const tickRef = useRef(0);
  const alertedRef = useRef(false);
  const logIdRef = useRef(4);

  function pushLog(level, text) {
    setLogs((prev) => [...prev, { id: logIdRef.current++, time: nowClock(), level, text }].slice(-9));
  }

  function handleFile(name) {
    if (!name.toLowerCase().endsWith('.pkl')) {
      pushLog('warn', `Rejected upload "${name}" — expected a .pkl model file.`);
      return;
    }
    setFileName(name);
    setStatus('validating');
    setChecks({ features: false, arch: false, shape: false });
    pushLog('info', `Received "${name}" — starting pre-flight validation.`);

    setTimeout(() => {
      setChecks((c) => ({ ...c, features: true }));
      pushLog('info', 'Input schema scan complete: 12 features detected.');
    }, 650);
    setTimeout(() => {
      setChecks((c) => ({ ...c, arch: true }));
      pushLog('info', 'Model architecture matches CNN-LSTM signature.');
    }, 1300);
    setTimeout(() => {
      setChecks((c) => ({ ...c, shape: true }));
      setStatus('ready');
      pushLog('info', `"${name}" passed validation — now serving inference.`);
    }, 1900);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f.name);
  }

  function onFileInput(e) {
    const f = e.target.files?.[0];
    if (f) handleFile(f.name);
  }

  useEffect(() => {
    const id = setInterval(() => {
      tickRef.current += 1;
      const n = tickRef.current;

      const dipPhase = n % 34;
      const inDip = dipPhase >= 24 && dipPhase <= 31;
      const dipAmt = inDip ? Math.sin(((dipPhase - 24) / 7) * Math.PI) * 0.16 : 0;

      const nextEarL = +(0.30 - dipAmt + (Math.random() - 0.5) * 0.01).toFixed(3);
      const nextEarR = +(0.29 - dipAmt + (Math.random() - 0.5) * 0.01).toFixed(3);
      const nextMar = +(0.17 + dipAmt * 1.4 + (Math.random() - 0.5) * 0.01).toFixed(3);

      setEarL(nextEarL);
      setEarR(nextEarR);
      setMar(nextMar);
      setEarHistory((h) => [...h.slice(1), (nextEarL + nextEarR) / 2]);
      setMarHistory((h) => [...h.slice(1), nextMar]);

      setFrameCount((f) => (inDip ? Math.max(6, 30 - Math.round(dipAmt * 90)) : 30));

      setLatency(+(12 + Math.random() * 5).toFixed(1));
      setCpu((c) => Math.max(22, Math.min(68, c + (Math.random() - 0.5) * 6)));
      setRam((r) => Math.max(30, Math.min(72, r + (Math.random() - 0.5) * 5)));

      const probability = Math.min(0.99, Math.max(0.02, dipAmt * 5.8 + Math.random() * 0.05));
      if (probability > threshold && !alertedRef.current && status === 'ready') {
        alertedRef.current = true;
        pushLog('critical', `Drowsiness probability ${probability.toFixed(2)} exceeded threshold ${threshold.toFixed(2)}.`);
      }
      if (!inDip) alertedRef.current = false;
    }, 600);
    return () => clearInterval(id);
  }, [threshold, status]);

  const meta = STATUS_META[status];

  return (
    <div style={{ width: '100%', minHeight: '100vh', background: C.bg, color: C.ink, padding: '40px 24px 60px' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        * { box-sizing: border-box; }
        .ms-root { font-family: 'Inter', sans-serif; }
        .ms-display { font-family: 'Space Grotesk', sans-serif; }
        .ms-mono { font-family: 'JetBrains Mono', monospace; }
        .ms-card { background: ${C.bg}; border: 1px solid ${C.border}; border-radius: 12px; }
        @keyframes ms-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        .ms-pulse { animation: ms-pulse 1.3s ease-in-out infinite; }
        @keyframes ms-spin { to { transform: rotate(360deg); } }
        .ms-spin { animation: ms-spin 1s linear infinite; }
        input[type="range"].ms-slider {
          -webkit-appearance: none; width: 100%; height: 6px; border-radius: 4px;
          background: linear-gradient(to right, ${C.amber} 0%, ${C.amber} var(--fill, 85%), ${C.border} var(--fill, 85%), ${C.border} 100%);
          outline: none;
        }
        input[type="range"].ms-slider::-webkit-slider-thumb {
          -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%;
          background: #fff; border: 3px solid ${C.amber}; cursor: pointer; box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        }
        input[type="range"].ms-slider::-moz-range-thumb {
          width: 18px; height: 18px; border-radius: 50%;
          background: #fff; border: 3px solid ${C.amber}; cursor: pointer;
        }
        .ms-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        .ms-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; }
        @media (max-width: 880px) {
          .ms-grid-2, .ms-grid-3 { grid-template-columns: 1fr !important; }
        }
      `}</style>

      <div className="ms-root" style={{ maxWidth: 1180, margin: '0 auto' }}>
        {/* ===== HEADER ===== */}
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 16,
            marginBottom: 26,
          }}
        >
          <div>
            <div className="ms-mono" style={{ fontSize: 12, color: C.amber, letterSpacing: 1, fontWeight: 600 }}>
              MODEL OPERATIONS
            </div>
            <h1 className="ms-display" style={{ fontSize: 30, fontWeight: 700, margin: '6px 0 0' }}>
              Model Sandbox
            </h1>
            <p style={{ color: C.inkMuted, fontSize: 14, marginTop: 4 }}>
              Validate, tune and observe the CNN-LSTM drowsiness engine before it goes live.
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div
              className="ms-mono"
              style={{
                fontSize: 12.5,
                color: C.inkMuted,
                border: `1px solid ${C.border}`,
                padding: '7px 12px',
                borderRadius: 8,
                background: C.surfaceAlt,
              }}
            >
              {fileName || 'vigi_hybrid_v1.pkl'}
            </div>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                background: meta.bg,
                color: meta.color,
                padding: '9px 18px',
                borderRadius: 999,
                fontWeight: 700,
                fontSize: 13,
                letterSpacing: 0.5,
              }}
            >
              <span
                className={status === 'validating' ? 'ms-pulse' : ''}
                style={{ width: 9, height: 9, borderRadius: '50%', background: meta.color, display: 'inline-block' }}
              />
              {meta.label}
            </div>
          </div>
        </div>

        {/* ===== VALIDATION SECTION ===== */}
        <div className="ms-grid-2" style={{ marginBottom: 18 }}>
          <div className="ms-card" style={{ padding: 22 }}>
            <SectionLabel icon={<UploadCloud size={15} />} text="Pre-flight upload" />
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              style={{
                marginTop: 12,
                border: `2px dashed ${dragOver ? C.amber : C.border}`,
                borderRadius: 10,
                padding: '30px 20px',
                textAlign: 'center',
                background: dragOver ? C.amberSoft : C.surfaceAlt,
                transition: 'all .15s ease',
              }}
            >
              <UploadCloud size={26} color={dragOver ? C.amber : C.inkDim} style={{ marginBottom: 8 }} />
              <div style={{ fontSize: 13.5, color: C.ink, fontWeight: 600 }}>
                Drop a <span className="ms-mono">.pkl</span> model file here
              </div>
              <div style={{ fontSize: 12.5, color: C.inkDim, marginTop: 4 }}>or</div>
              <label style={{ display: 'inline-block', marginTop: 8, cursor: 'pointer' }}>
                <span
                  className="ms-mono"
                  style={{
                    fontSize: 12.5,
                    color: C.navy,
                    border: `1px solid ${C.border}`,
                    background: C.bg,
                    padding: '7px 14px',
                    borderRadius: 7,
                  }}
                >
                  Browse file
                </span>
                <input type="file" accept=".pkl" onChange={onFileInput} style={{ display: 'none' }} />
              </label>
            </div>
          </div>

          <div className="ms-card" style={{ padding: 22 }}>
            <SectionLabel icon={<FileCheck2 size={15} />} text="Automated validation report" />
            <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <ReportRow
                label="Input features detected"
                value="12 features (EAR, MAR, blink velocity, ...)"
                done={checks.features}
                validating={status === 'validating' && !checks.features}
              />
              <CheckRow
                label="Model architecture matches CNN-LSTM signature"
                done={checks.arch}
                validating={status === 'validating' && checks.features && !checks.arch}
              />
              <CheckRow
                label="Input shape validated against expected tensor"
                done={checks.shape}
                validating={status === 'validating' && checks.arch && !checks.shape}
              />
            </div>
            {status === 'idle' && (
              <div style={{ fontSize: 12.5, color: C.inkDim, marginTop: 14 }}>
                Awaiting a model upload — no validation has run yet.
              </div>
            )}
          </div>
        </div>

        {/* ===== INFERENCE CONTROL PANEL ===== */}
        <div className="ms-card" style={{ padding: 22, marginBottom: 18 }}>
          <SectionLabel icon={<Sliders size={15} />} text="Live inference control panel" />
          <div className="ms-grid-3" style={{ marginTop: 16, alignItems: 'start' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>Confidence threshold</span>
                <span className="ms-mono" style={{ fontSize: 13, color: C.amber, fontWeight: 600 }}>
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={threshold}
                className="ms-slider"
                style={{ '--fill': `${threshold * 100}%` }}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
              />
              <div style={{ fontSize: 12, color: C.inkMuted, marginTop: 8, lineHeight: 1.5 }}>
                Alert triggers when drowsiness probability exceeds <b className="ms-mono">{threshold.toFixed(2)}</b>.
              </div>
            </div>

            <MetricCard icon={<Activity size={16} />} label="Inference latency" value={`${latency.toFixed(1)} ms`} sub="per-frame, on-device" />

            <div>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>Processing load</div>
              <LoadBar label="CPU" value={cpu} color={C.navy} />
              <LoadBar label="RAM" value={ram} color={C.teal} />
            </div>
          </div>
        </div>

        {/* ===== DATA DEBUGGER ===== */}
        <div className="ms-grid-2" style={{ marginBottom: 18 }}>
          <div className="ms-card" style={{ padding: 22 }}>
            <SectionLabel icon={<Gauge size={15} />} text="Live feature stream" />
            <div style={{ marginTop: 14 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <tbody>
                  <FeatureRow label="EAR (left / right)" value={`${earL.toFixed(2)} / ${earR.toFixed(2)}`} />
                  <FeatureRow label="MAR (mouth)" value={mar.toFixed(2)} />
                  <FeatureRow label="Temporal window state" value={`${frameCount}/30 frames filled`} />
                </tbody>
              </table>
              <div style={{ fontSize: 11.5, color: C.inkDim, marginTop: 10 }}>Streaming at ~30 fps to the CNN-LSTM input buffer.</div>
            </div>
          </div>

          <div className="ms-card" style={{ padding: 22 }}>
            <SectionLabel icon={<Activity size={15} />} text="EAR / MAR sparkline" />
            <div style={{ marginTop: 14 }}>
              <Sparkline data={earHistory} color={C.navy} domain={[0.05, 0.35]} label="EAR" />
              <div style={{ height: 10 }} />
              <Sparkline data={marHistory} color={C.amber} domain={[0.1, 0.5]} label="MAR" />
            </div>
          </div>
        </div>

        {/* ===== DEPLOYMENT / EMERGENCY LOG ===== */}
        <div className="ms-card" style={{ padding: 22 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <SectionLabel icon={<Terminal size={15} />} text="Deployment & event log" />
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: C.teal, fontSize: 12 }}>
              <Wifi size={13} /> connected
            </div>
          </div>
          <div
            className="ms-mono"
            style={{
              marginTop: 14,
              background: C.console,
              border: `1px solid ${C.consoleBorder}`,
              borderRadius: 8,
              padding: '14px 16px',
              height: 190,
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 6,
            }}
          >
            {logs.map((l) => (
              <div key={l.id} style={{ display: 'flex', gap: 10, fontSize: 12.5 }}>
                <span style={{ color: '#5A6B80' }}>[{l.time}]</span>
                <span
                  style={{
                    color: l.level === 'critical' ? '#FF8A80' : l.level === 'warn' ? '#F2C245' : '#B7C4D4',
                  }}
                >
                  {l.text}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- helpers ---------- */

function SectionLabel({ icon, text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: C.navy }}>
      {icon}
      <span style={{ fontSize: 12.5, fontWeight: 700, letterSpacing: 0.3, textTransform: 'uppercase' }}>{text}</span>
    </div>
  );
}

function CheckRow({ label, done, validating }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      {done ? (
        <CheckCircle2 size={17} color={C.teal} />
      ) : validating ? (
        <Loader2 size={17} color={C.amber} className="ms-spin" />
      ) : (
        <Circle size={17} color={C.border} />
      )}
      <span style={{ fontSize: 13.5, color: done ? C.ink : C.inkMuted }}>{label}</span>
    </div>
  );
}

function ReportRow({ label, value, done, validating }) {
  return (
    <div>
      <CheckRow label={label} done={done} validating={validating} />
      {done && (
        <div className="ms-mono" style={{ fontSize: 12, color: C.inkDim, marginLeft: 27, marginTop: 2 }}>
          {value}
        </div>
      )}
    </div>
  );
}

function MetricCard({ icon, label, value, sub }) {
  return (
    <div style={{ background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 9, padding: '14px 16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7, color: C.inkMuted, fontSize: 12 }}>
        {icon} {label}
      </div>
      <div className="ms-mono" style={{ fontSize: 22, fontWeight: 600, color: C.ink, marginTop: 4 }}>
        {value}
      </div>
      <div style={{ fontSize: 11.5, color: C.inkDim, marginTop: 2 }}>{sub}</div>
    </div>
  );
}

function LoadBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: C.inkMuted, marginBottom: 4 }}>
        <span>{label}</span>
        <span className="ms-mono">{Math.round(value)}%</span>
      </div>
      <div style={{ height: 7, borderRadius: 4, background: C.surfaceAlt2, overflow: 'hidden' }}>
        <div style={{ width: `${value}%`, height: '100%', background: color, borderRadius: 4, transition: 'width .4s ease' }} />
      </div>
    </div>
  );
}

function FeatureRow({ label, value }) {
  return (
    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
      <td style={{ padding: '9px 0', color: C.inkMuted, fontSize: 12.5 }}>{label}</td>
      <td className="ms-mono" style={{ padding: '9px 0', textAlign: 'right', fontWeight: 600, color: C.ink }}>
        {value}
      </td>
    </tr>
  );
}

function Sparkline({ data, color, domain, label }) {
  const w = 400,
    h = 46;
  const [lo, hi] = domain;
  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const norm = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
      const y = h - norm * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <div>
      <div style={{ fontSize: 11, color: C.inkDim, marginBottom: 3 }}>{label}</div>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        <polyline points={pts} fill="none" stroke={color} strokeWidth="2" />
      </svg>
    </div>
  );
}