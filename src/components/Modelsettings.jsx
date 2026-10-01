// pages/ModelSettings.jsx
import React, { useState } from 'react';
import { Sliders, Camera, Bell, Save, CheckCircle2 } from 'lucide-react';

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
};

const CAMERAS = ['Built-in webcam', 'External USB camera', 'IR night-vision camera'];
const STRIDES = [1, 2, 4, 8];

export default function ModelSettings() {
  const [sensitivity, setSensitivity] = useState(0.85);
  const [stride, setStride] = useState(4);
  const [camera, setCamera] = useState(CAMERAS[0]);
  const [toggles, setToggles] = useState({
    chime: true,
    push: true,
    flash: false,
    email: false,
  });
  const [saved, setSaved] = useState(false);

  function toggle(key) {
    setToggles((t) => ({ ...t, [key]: !t[key] }));
    setSaved(false);
  }

  function handleSave() {
    setSaved(true);
    setTimeout(() => setSaved(false), 2600);
  }

  return (
    <div style={{ width: '100%', minHeight: '100vh', background: C.bg, color: C.ink, padding: '36px 24px 60px' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        * { box-sizing: border-box; }
        .cfg-root { font-family: 'Inter', sans-serif; }
        .cfg-display { font-family: 'Space Grotesk', sans-serif; }
        .cfg-mono { font-family: 'JetBrains Mono', monospace; }
        .cfg-card { background: ${C.bg}; border: 1px solid ${C.border}; border-radius: 12px; }
        input[type="range"].cfg-slider {
          -webkit-appearance: none; width: 100%; height: 6px; border-radius: 4px;
          background: linear-gradient(to right, ${C.amber} 0%, ${C.amber} var(--fill, 50%), ${C.border} var(--fill, 50%), ${C.border} 100%);
          outline: none;
        }
        input[type="range"].cfg-slider::-webkit-slider-thumb {
          -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%;
          background: #fff; border: 3px solid ${C.amber}; cursor: pointer; box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        }
        input[type="range"].cfg-slider::-moz-range-thumb {
          width: 18px; height: 18px; border-radius: 50%; background: #fff; border: 3px solid ${C.amber}; cursor: pointer;
        }
        .cfg-select {
          appearance: none; background: ${C.bg}; border: 1px solid ${C.border}; border-radius: 8px;
          padding: 10px 34px 10px 14px; font-size: 14px; font-family: 'Inter', sans-serif; color: ${C.ink};
          cursor: pointer; width: 100%;
        }
        .cfg-stride-btn {
          padding: 8px 0; border-radius: 8px; border: 1px solid ${C.border}; background: ${C.bg};
          font-family: 'JetBrains Mono', monospace; font-size: 13px; cursor: pointer; flex: 1; text-align: center;
        }
        .cfg-toggle {
          width: 42px; height: 24px; border-radius: 999px; border: none; cursor: pointer; position: relative; flex-shrink: 0;
          transition: background .15s ease;
        }
        .cfg-toggle-knob {
          position: absolute; top: 3px; width: 18px; height: 18px; border-radius: 50%; background: #fff;
          transition: left .15s ease; box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        .cfg-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        @media (max-width: 880px) { .cfg-grid-2 { grid-template-columns: 1fr !important; } }
      `}</style>

      <div className="cfg-root" style={{ maxWidth: 900, margin: '0 auto' }}>
        <div style={{ marginBottom: 22 }}>
          <div className="cfg-mono" style={{ fontSize: 12, color: C.amber, letterSpacing: 1, fontWeight: 600 }}>
            SYSTEM CONFIGURATION
          </div>
          <h1 className="cfg-display" style={{ fontSize: 28, fontWeight: 700, margin: '6px 0 0' }}>
            Model Settings &amp; Configuration
          </h1>
          <p style={{ color: C.inkMuted, fontSize: 14, marginTop: 4 }}>
            Tune detection sensitivity and input sources without touching the model code.
          </p>
        </div>

        <div className="cfg-grid-2" style={{ marginBottom: 18 }}>
          {/* sensitivity + stride */}
          <div className="cfg-card" style={{ padding: 22 }}>
            <SectionLabel icon={<Sliders size={15} />} text="Detection tuning" />

            <div style={{ marginTop: 18 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>Alert sensitivity</span>
                <span className="cfg-mono" style={{ fontSize: 13, color: C.amber, fontWeight: 700 }}>
                  {sensitivity.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={sensitivity}
                className="cfg-slider"
                style={{ '--fill': `${sensitivity * 100}%` }}
                onChange={(e) => {
                  setSensitivity(parseFloat(e.target.value));
                  setSaved(false);
                }}
              />
              <div style={{ fontSize: 12, color: C.inkMuted, marginTop: 6 }}>
                Higher values reduce false alarms but react more slowly.
              </div>
            </div>

            <div style={{ marginTop: 24 }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Window stride (frames)</div>
              <div style={{ display: 'flex', gap: 8 }}>
                {STRIDES.map((s) => (
                  <button
                    key={s}
                    className="cfg-stride-btn"
                    style={{
                      background: stride === s ? C.navy : C.bg,
                      color: stride === s ? '#fff' : C.ink,
                      borderColor: stride === s ? C.navy : C.border,
                    }}
                    onClick={() => {
                      setStride(s);
                      setSaved(false);
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
              <div style={{ fontSize: 12, color: C.inkMuted, marginTop: 6 }}>
                Smaller strides run inference more often, at higher compute cost.
              </div>
            </div>
          </div>

          {/* camera + notifications */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <div className="cfg-card" style={{ padding: 22 }}>
              <SectionLabel icon={<Camera size={15} />} text="Camera selection" />
              <div style={{ marginTop: 14, position: 'relative' }}>
                <select
                  className="cfg-select"
                  value={camera}
                  onChange={(e) => {
                    setCamera(e.target.value);
                    setSaved(false);
                  }}
                >
                  {CAMERAS.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="cfg-card" style={{ padding: 22 }}>
              <SectionLabel icon={<Bell size={15} />} text="Notification toggles" />
              <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 14 }}>
                <ToggleRow label="Audible chime on tier 2+" active={toggles.chime} onClick={() => toggle('chime')} />
                <ToggleRow label="Push notification to phone" active={toggles.push} onClick={() => toggle('push')} />
                <ToggleRow label="Dashboard visual flash" active={toggles.flash} onClick={() => toggle('flash')} />
                <ToggleRow label="Email report on critical event" active={toggles.email} onClick={() => toggle('email')} />
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleSave}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            background: saved ? C.teal : C.amber,
            color: saved ? '#fff' : '#221304',
            fontWeight: 700,
            fontSize: 14.5,
            padding: '12px 24px',
            borderRadius: 8,
            border: 'none',
            cursor: 'pointer',
            transition: 'background .2s ease',
          }}
        >
          {saved ? <CheckCircle2 size={17} /> : <Save size={17} />}
          {saved ? 'Settings saved' : 'Save configuration'}
        </button>
      </div>
    </div>
  );
}

function SectionLabel({ icon, text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: C.navy }}>
      {icon}
      <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: 0.3, textTransform: 'uppercase' }}>{text}</span>
    </div>
  );
}

function ToggleRow({ label, active, onClick }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <span style={{ fontSize: 13.5, color: C.ink }}>{label}</span>
      <button className="cfg-toggle" onClick={onClick} style={{ background: active ? C.teal : '#D6DBE2' }}>
        <span className="cfg-toggle-knob" style={{ left: active ? 21 : 3 }} />
      </button>
    </div>
  );
}