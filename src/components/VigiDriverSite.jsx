import { useState } from "react";
import { Link } from "react-router-dom";
import {
  ShieldCheck,
  Eye,
  Moon,
  Cpu,
  Layers,
  ScanFace,
  BellRing,
  WifiOff,
  Compass,
  Mail,
  Phone,
  MapPin,
  Code2,
  Share2,
  ArrowRight,
  CheckCircle2,
  Menu,
  X,
} from "lucide-react";

/* TOKENS ---------------------------------------------------------
   Light, formal palette: white/near-white surfaces throughout,
   deep navy reserved for headlines, borders and icons (not big
   background blocks), amber for the product's tiered-alert accent,
   sage-teal for reassurance/benefit checkmarks.
------------------------------------------------------------------- */
const C = {
  bg: "#FFFFFF",
  surfaceAlt: "#F4F6F9",
  surfaceAlt2: "#EEF1F6",
  border: "#E2E6EC",
  ink: "#132133",
  inkMuted: "#5B6572",
  inkDim: "#8A93A2",
  navy: "#163A5F",
  navySoft: "#1F4C79",
  amber: "#C57C21",
  amberSoft: "#FBEDD5",
  teal: "#1F7A63",
  tealSoft: "#E4F2EC",
  critical: "#C4463F",
};

const NAV_LINKS = [
  { id: "overview", label: "Overview" },
  { id: "components", label: "How it works" },
  { id: "features", label: "Features" },
  { id: "benefits", label: "Benefits" },
  { id: "contact", label: "Contact" },
];

export default function VigiDriverSite() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [form, setForm] = useState({ name: "", email: "", message: "" });
  const [sent, setSent] = useState(false);

  function handleSubmit(e) {
    e.preventDefault();
    const subject = encodeURIComponent("VigiDriver — enquiry from " + (form.name || "website visitor"));
    const body = encodeURIComponent(
      `Name: ${form.name}\nEmail: ${form.email}\n\n${form.message}`
    );
    window.open(`mailto:team7.vigidriver@amrita.edu?subject=${subject}&body=${body}`, "_blank");
    setSent(true);
  }

  return (
    <div style={{ width: "100%", minHeight: "100vh", background: C.bg, color: C.ink }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
        html, body { margin: 0; padding: 0; scroll-behavior: smooth; }
        * { box-sizing: border-box; }
        .vgs-root { font-family: 'Inter', sans-serif; width: 100%; }
        .vgs-display { font-family: 'Space Grotesk', sans-serif; }
        .vgs-mono { font-family: 'JetBrains Mono', monospace; }
        .vgs-nav-link { color: ${C.inkMuted}; text-decoration: none; font-size: 14px; font-weight: 500; transition: color .15s; }
        .vgs-nav-link:hover { color: ${C.navy}; }
        .vgs-card { background: ${C.bg}; border: 1px solid ${C.border}; border-radius: 12px; }
        .vgs-btn-primary {
          display: inline-flex; align-items: center; gap: 8px;
          background: ${C.amber}; color: #221304; font-weight: 600; font-size: 14.5px;
          padding: 12px 22px; border-radius: 8px; border: none; cursor: pointer;
          text-decoration: none; transition: transform .12s ease, box-shadow .12s ease;
        }
        .vgs-btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(197,124,33,0.30); }
        .vgs-btn-ghost {
          display: inline-flex; align-items: center; gap: 8px;
          background: transparent; color: ${C.navy}; font-weight: 500; font-size: 14.5px;
          padding: 11px 20px; border-radius: 8px; border: 1px solid ${C.border};
          text-decoration: none; transition: border-color .15s ease;
        }
        .vgs-btn-ghost:hover { border-color: ${C.navy}; }
        .vgs-input {
          width: 100%; padding: 11px 13px; border-radius: 8px; border: 1px solid ${C.border};
          font-family: 'Inter', sans-serif; font-size: 14px; color: ${C.ink}; background: #FCFDFE;
        }
        .vgs-input:focus { outline: 2px solid ${C.navySoft}; outline-offset: 1px; }
        @media (max-width: 880px) {
          .vgs-hero-grid { grid-template-columns: 1fr !important; }
          .vgs-grid-4 { grid-template-columns: 1fr 1fr !important; }
          .vgs-grid-3 { grid-template-columns: 1fr !important; }
          .vgs-benefits-grid { grid-template-columns: 1fr !important; }
          .vgs-nav-links { display: none !important; }
          .vgs-menu-btn { display: inline-flex !important; }
          .vgs-contact-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>

      <div className="vgs-root">
        {/* ================= NAV ================= */}
        <header style={{ background: C.bg, borderBottom: `1px solid ${C.border}`, position: "sticky", top: 0, zIndex: 20 }}>
          <div
            style={{
              maxWidth: 1180, margin: "0 auto", padding: "16px 24px",
              display: "flex", alignItems: "center", justifyContent: "space-between",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
              <ShieldCheck size={20} color={C.amber} />
              <span className="vgs-display" style={{ color: C.navy, fontSize: 18, fontWeight: 700, letterSpacing: 0.5 }}>
                VIGIDRIVER
              </span>
            </div>
            <nav className="vgs-nav-links" style={{ display: "flex", gap: 28 }}>
              {NAV_LINKS.map((l) => (
                <a key={l.id} href={`#${l.id}`} className="vgs-nav-link">{l.label}</a>
              ))}
            </nav>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Link to="/models" style={{ textDecoration: "none", color: C.navy, fontSize: 13.5, fontWeight: 600, padding: "8px 14px", border: `1px solid ${C.border}`, borderRadius: 8 }}>
                Model Specs
              </Link>
              <Link to="/dashboard" className="vgs-btn-primary" style={{ padding: "9px 16px", fontSize: 13.5, textDecoration: "none" }}>
                Live Dashboard
              </Link>
              <button
                className="vgs-menu-btn"
                onClick={() => setMenuOpen((o) => !o)}
                style={{ display: "none", background: "transparent", border: "none", color: C.navy, cursor: "pointer" }}
              >
                {menuOpen ? <X size={22} /> : <Menu size={22} />}
              </button>
            </div>
          </div>
          {menuOpen && (
            <div style={{ background: C.surfaceAlt, padding: "10px 24px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
              {NAV_LINKS.map((l) => (
                <a key={l.id} href={`#${l.id}`} className="vgs-nav-link" onClick={() => setMenuOpen(false)}>{l.label}</a>
              ))}
              <Link to="/models" className="vgs-nav-link" onClick={() => setMenuOpen(false)}>Model Specs</Link>
              <Link to="/dashboard" className="vgs-nav-link" onClick={() => setMenuOpen(false)}>Live Dashboard</Link>
            </div>
          )}
        </header>

        {/* ================= HERO ================= */}
        <section id="overview" style={{ background: C.surfaceAlt, padding: "60px 24px 74px" }}>
          <div className="vgs-hero-grid" style={{ maxWidth: 1180, margin: "0 auto", display: "grid", gridTemplateColumns: "1fr", justifyItems: "center", gap: 56, alignItems: "center" }}>
            <div>
              <div style={{ display: "flex", gap: 8, marginBottom: 18, flexWrap: "wrap", justifyContent: "center" }}>
                {["Driver monitoring", "CNN-LSTM", "Edge AI", "23CSE473"].map((tag) => (
                  <span key={tag} className="vgs-mono" style={{ fontSize: 11, color: C.navySoft, border: `1px solid ${C.border}`, background: C.bg, padding: "3px 9px", borderRadius: 20 }}>
                    #{tag}
                  </span>
                ))}
              </div>
              <h1 className="vgs-display" style={{ fontSize: 42, lineHeight: 1.15, fontWeight: 700, margin: 0, color: C.ink, textAlign: "center" }}>
                Catching micro-sleep before it becomes a crash
              </h1>
              <p style={{ fontSize: 16.5, color: C.inkMuted, lineHeight: 1.65, marginTop: 18, maxWidth: 640, textAlign: "center" }}>
                VigiDriver is a vision-based driver monitoring system that tracks how fatigue
                actually builds — frame by frame, blink by blink — powered strictly by our
                5-Conv2D Deep Neural Network checkpoint.
              </p>
              <div style={{ display: "flex", gap: 12, marginTop: 30, justifyContent: "center", flexWrap: "wrap" }}>
                <Link to="/dashboard" className="vgs-btn-primary" style={{ textDecoration: "none" }}>
                  Launch Live Dashboard <ArrowRight size={16} />
                </Link>
                <Link to="/models" className="vgs-btn-ghost" style={{ textDecoration: "none" }}>
                  Inspect Model &amp; Weights
                </Link>
              </div>

              <div style={{ display: "flex", gap: 36, marginTop: 44, justifyContent: "center", flexWrap: "wrap" }}>
                <Stat value="2" label="Behavioral signals fused: EAR + MAR" />
                <Stat value="<1s" label="On-device alert latency, no cloud round trip" />
                <Stat value="0" label="Specialized sensors required" />
              </div>
            </div>

            <HeroArt />
          </div>
        </section>

        {/* ================= SCENARIO CALLOUT ================= */}
        <section style={{ maxWidth: 1180, margin: "-30px auto 0", padding: "0 24px", position: "relative", zIndex: 2 }}>
          <div className="vgs-card" style={{ padding: "26px 30px", display: "flex", gap: 20, alignItems: "flex-start", boxShadow: "0 12px 30px rgba(20,40,60,0.08)" }}>
            <Moon size={26} color={C.amber} style={{ flexShrink: 0, marginTop: 3 }} />
            <div>
              <div className="vgs-display" style={{ fontWeight: 700, fontSize: 15.5, color: C.ink }}>3:00 AM, an empty highway</div>
              <p style={{ color: C.inkMuted, fontSize: 14.5, lineHeight: 1.65, marginTop: 6, marginBottom: 0 }}>
                A long-haul driver's cabin is lit only by the dashboard glow. Standard cameras
                lose the face in the dark, and a cloud-connected system stalls the moment
                cellular coverage drops — right as a two-second micro-sleep begins. VigiDriver
                was built for exactly this moment: local inference, low-light recovery, and
                zero dependency on a network connection.
              </p>
            </div>
          </div>
        </section>

        {/* ================= SYSTEM COMPONENTS ================= */}
        <section id="components" style={{ padding: "84px 24px 20px" }}>
          <div style={{ maxWidth: 1180, margin: "0 auto" }}>
            <SectionEyebrow>System components</SectionEyebrow>
            <h2 className="vgs-display" style={{ fontSize: 30, fontWeight: 700, margin: "8px 0 34px", color: C.ink }}>
              A four-stage on-device pipeline
            </h2>
            <div className="vgs-grid-4" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
              {[
                { n: "01", icon: <Compass size={20} />, title: "Low-light optimization", text: "YUV-CLAHE and adaptive gamma recover facial detail in near-total darkness." },
                { n: "02", icon: <ScanFace size={20} />, title: "Geometric abstraction", text: "MediaPipe FaceMesh reduces the face to light-agnostic eye and mouth ratios." },
                { n: "03", icon: <Layers size={20} />, title: "Sequential tracking", text: "A CNN-LSTM reads those ratios over a sliding window to judge speed, not snapshots." },
                { n: "04", icon: <WifiOff size={20} />, title: "Offline edge deployment", text: "The compressed model runs fully on-device — sub-second alerts, zero cloud dependency." },
              ].map((s) => (
                <div key={s.n} className="vgs-card" style={{ padding: "22px 20px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <div style={{ color: C.navy, background: C.surfaceAlt, borderRadius: 8, width: 38, height: 38, display: "flex", alignItems: "center", justifyContent: "center" }}>
                      {s.icon}
                    </div>
                    <span className="vgs-mono" style={{ fontSize: 12, color: C.inkDim }}>{s.n}</span>
                  </div>
                  <div className="vgs-display" style={{ fontWeight: 700, fontSize: 15.5, marginTop: 14, color: C.ink }}>{s.title}</div>
                  <p style={{ fontSize: 13.5, color: C.inkMuted, lineHeight: 1.6, marginTop: 6 }}>{s.text}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ================= PRODUCT FEATURES ================= */}
        <section id="features" style={{ padding: "80px 24px", background: C.surfaceAlt }}>
          <div style={{ maxWidth: 1180, margin: "0 auto" }}>
            <SectionEyebrow>Product features</SectionEyebrow>
            <h2 className="vgs-display" style={{ fontSize: 30, fontWeight: 700, margin: "8px 0 34px", color: C.ink }}>
              Built to tell a blink from a warning sign
            </h2>
            <div className="vgs-grid-3" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 18 }}>
              <FeatureCard
                icon={<Layers size={20} />}
                title="Spatiotemporal CNN-LSTM core"
                text="Reads eyelid and yawn behavior across a rolling window of frames, so it judges duration and velocity — not just a single closed-eye frame."
              />
              <FeatureCard
                icon={<Eye size={20} />}
                title="EAR / MAR feature tracking"
                text="Eye Aspect Ratio and Mouth Aspect Ratio replace raw pixels with stable, lighting-independent measurements of eye closure and yawning."
              />
              <FeatureCard
                icon={<Compass size={20} />}
                title="YUV-CLAHE night vision"
                text="Boosts contrast on the luminance channel only, recovering a trackable face in a pitch-black cabin without introducing color noise."
              />
              <FeatureCard
                icon={<BellRing size={20} />}
                title="Tiered alert escalation"
                text="A soft visual cue, then an audible chime, then an urgent alarm — calibrated to avoid the false-alarm fatigue that gets safety systems switched off."
              />
              <FeatureCard
                icon={<WifiOff size={20} />}
                title="Fully offline inference"
                text="No round trip to a server. Response time depends only on onboard compute, so the system still works in a cellular dead zone."
              />
              <FeatureCard
                icon={<Cpu size={20} />}
                title="Runs on commodity hardware"
                text="A standard laptop or in-cabin webcam is enough — no infrared sensors, no proprietary driver-facing camera, no retrofit."
              />
            </div>
          </div>
        </section>

        {/* ================= OPERATION ================= */}
        <section style={{ padding: "80px 28px", textAlign: "center" }}>
          <div style={{ maxWidth: 1180, margin: "0 auto", width: "100%" }}>
            <div className="vgs-card" style={{ padding: "66px 84px", textAlign: "center", boxShadow: "0 20px 50px rgba(22,45,70,0.13)", borderRadius: 16 }}>
              <SectionEyebrow>Operation principle</SectionEyebrow>
              <h2 className="vgs-display" style={{ fontSize: 28, fontWeight: 700, margin: "14px 0 24px", color: C.ink, textAlign: "center" }}>
                From raw frame to a calibrated alert
              </h2>
              <p style={{ color: C.inkMuted, fontSize: 17.5, lineHeight: 1.85, maxWidth: 880, textAlign: "center" }}>
                Each frame is brightness-corrected, reduced to eye and mouth ratios, then handed
                to the LSTM alongside recent history. A normal blink and the start of a
                micro-sleep can look identical in one frame — the difference only appears once
                you track how long the eye stays closed and how quickly it reopens. When that
                trend crosses a threshold, the tiered-alert layer decides how urgently to warn
                the driver, weighing severity against the risk of crying wolf.
              </p>
            </div>
          </div>
        </section>

        {/* ================= BENEFITS ================= */}
        <section id="benefits" style={{ padding: "0 24px 84px" }}>
          <div style={{ maxWidth: 1180, margin: "0 auto" }}>
            <SectionEyebrow>System benefits</SectionEyebrow>
            <h2 className="vgs-display" style={{ fontSize: 30, fontWeight: 700, margin: "8px 0 34px", color: C.ink }}>
              Built around two audiences
            </h2>
            <div className="vgs-benefits-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <BenefitCard
                title="For drivers"
                items={[
                  "An early, calibrated warning before a micro-sleep — not after one starts",
                  "Escalation stays proportional, so alerts stay trustworthy instead of becoming noise",
                  "Nothing leaves the vehicle: video is processed on-device, never uploaded",
                ]}
              />
              <BenefitCard
                title="For fleet operators & researchers"
                items={[
                  "No infrared cameras or connectivity contracts to deploy or maintain",
                  "Continuous behavioral modeling instead of brittle, single-frame thresholds",
                  "A dataset and architecture designed specifically for low-light, occluded driving conditions",
                ]}
              />
            </div>
          </div>
        </section>

        {/* ================= CONTACT ================= */}
        <section id="contact" style={{ background: C.surfaceAlt, padding: "80px 24px" }}>
          <div style={{ maxWidth: 1180, margin: "0 auto" }}>
            <div style={{ textAlign: "center", marginBottom: 42 }}>
              <span className="vgs-mono" style={{ fontSize: 12, color: C.amber, letterSpacing: 1, fontWeight: 600 }}>GET IN TOUCH</span>
              <h2 className="vgs-display" style={{ fontSize: 30, fontWeight: 700, margin: "10px 0 10px", color: C.ink }}>Contact us</h2>
              <p style={{ color: C.inkMuted, fontSize: 14.5, maxWidth: 520, margin: "0 auto" }}>
                Questions about the architecture, the dataset, or a collaboration? Team 7 would love to hear from you.
              </p>
            </div>

            <div className="vgs-contact-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 30 }}>
              <div className="vgs-card" style={{ padding: 28 }}>
                {sent ? (
                  <div style={{ textAlign: "center", padding: "34px 10px" }}>
                    <CheckCircle2 size={34} color={C.teal} />
                    <div className="vgs-display" style={{ fontWeight: 700, fontSize: 18, marginTop: 12, color: C.ink }}>Message ready to send</div>
                    <p style={{ color: C.inkMuted, fontSize: 13.5, marginTop: 6 }}>
                      Your email client should have opened with your note pre-filled. If not,
                      write to us directly at team7.vigidriver@amrita.edu.
                    </p>
                    <button className="vgs-btn-ghost" style={{ marginTop: 14 }} onClick={() => setSent(false)}>
                      Send another message
                    </button>
                  </div>
                ) : (
                  <form onSubmit={handleSubmit}>
                    <div style={{ marginBottom: 14 }}>
                      <label style={{ fontSize: 13, fontWeight: 600, color: C.ink }}>Name</label>
                      <input required className="vgs-input" style={{ marginTop: 6 }} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="Your name" />
                    </div>
                    <div style={{ marginBottom: 14 }}>
                      <label style={{ fontSize: 13, fontWeight: 600, color: C.ink }}>Email</label>
                      <input required type="email" className="vgs-input" style={{ marginTop: 6 }} value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} placeholder="you@example.com" />
                    </div>
                    <div style={{ marginBottom: 18 }}>
                      <label style={{ fontSize: 13, fontWeight: 600, color: C.ink }}>Message</label>
                      <textarea required className="vgs-input" style={{ marginTop: 6, minHeight: 100, resize: "vertical" }} value={form.message} onChange={(e) => setForm({ ...form, message: e.target.value })} placeholder="Tell us what you'd like to know" />
                    </div>
                    <button type="submit" className="vgs-btn-primary" style={{ width: "100%", justifyContent: "center" }}>
                      Send message <ArrowRight size={16} />
                    </button>
                  </form>
                )}
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <ContactRow icon={<Mail size={18} />} title="Email" text="team7.vigidriver@amrita.edu" />
                <ContactRow icon={<Phone size={18} />} title="Faculty guide" text="Available via department office hours" />
                <ContactRow icon={<MapPin size={18} />} title="Department" text="Computer Science & Engineering, Amrita School of Computing" />
                <div style={{ display: "flex", gap: 12, marginTop: 6 }}>
                  <a href="#" className="vgs-btn-ghost"><Code2 size={16} /> Repository</a>
                  <a href="#" className="vgs-btn-ghost"><Share2 size={16} /> LinkedIn</a>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ================= FOOTER ================= */}
        <footer style={{ padding: "22px 24px", background: C.bg, borderTop: `1px solid ${C.border}` }}>
          <div style={{ maxWidth: 1180, margin: "0 auto", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
            <span style={{ fontSize: 12.5, color: C.inkDim }}>© 2026 VigiDriver · Team 7 · 23CSE473 Neural Networks and Deep Learning</span>
            <span style={{ fontSize: 12.5, color: C.inkDim }}>Amrita Vishwa Vidyapeetham</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

/* ---------- small components ---------- */

function SectionEyebrow({ children }) {
  return (
    <span className="vgs-mono" style={{ fontSize: 12, color: C.amber, letterSpacing: 1, fontWeight: 600 }}>
      {children.toUpperCase()}
    </span>
  );
}

function Stat({ value, label }) {
  return (
    <div>
      <div className="vgs-mono" style={{ fontSize: 26, fontWeight: 600, color: C.navy }}>{value}</div>
      <div style={{ fontSize: 12.5, color: C.inkMuted, marginTop: 4, maxWidth: 140, lineHeight: 1.4 }}>{label}</div>
    </div>
  );
}

function FeatureCard({ icon, title, text }) {
  return (
    <div className="vgs-card" style={{ padding: "24px 22px" }}>
      <div style={{ color: C.teal, background: C.tealSoft, borderRadius: 8, width: 40, height: 40, display: "flex", alignItems: "center", justifyContent: "center" }}>
        {icon}
      </div>
      <div className="vgs-display" style={{ fontWeight: 700, fontSize: 16, marginTop: 14, color: C.ink }}>{title}</div>
      <p style={{ fontSize: 13.5, color: C.inkMuted, lineHeight: 1.6, marginTop: 7 }}>{text}</p>
    </div>
  );
}

function BenefitCard({ title, items }) {
  return (
    <div className="vgs-card" style={{ padding: "26px 26px" }}>
      <div className="vgs-display" style={{ fontWeight: 700, fontSize: 17, marginBottom: 16, color: C.ink }}>{title}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {items.map((it, i) => (
          <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
            <CheckCircle2 size={17} color={C.teal} style={{ flexShrink: 0, marginTop: 2 }} />
            <span style={{ fontSize: 13.5, color: C.inkMuted, lineHeight: 1.55 }}>{it}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ContactRow({ icon, title, text }) {
  return (
    <div className="vgs-card" style={{ padding: "16px 18px", display: "flex", gap: 12, alignItems: "center" }}>
      <div style={{ color: C.amber }}>{icon}</div>
      <div>
        <div style={{ fontSize: 12, color: C.inkDim }}>{title}</div>
        <div style={{ fontSize: 14, color: C.ink, fontWeight: 500 }}>{text}</div>
      </div>
    </div>
  );
}

function HeroArt() {
  return (
    <div style={{ display: "flex", justifyContent: "center" }}>
      <img
        src={"/src/assets/hero.png"}
        alt="Car dashboard at night with illuminated steering wheel"
        style={{ width: "100%", maxWidth: 520, height: "auto", borderRadius: 18, objectFit: "cover", display: "block" }}
      />
    </div>
  );
}