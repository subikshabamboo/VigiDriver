/* Contact & Support portal.
 *
 * Stores enquiries and technical-support requests through the backend API
 * (SQLite-persisted). This is the standalone communication page; the landing
 * page keeps its own original contact section untouched.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  Bug,
  CheckCircle2,
  Mail,
  MapPin,
  MessageSquarePlus,
  Phone,
  Send,
  ShieldCheck,
} from "lucide-react";
import { API_BASE, C, apiPost } from "../lib/api.js";
import { Card, PageHeader, PanelLabel } from "../lib/ui.jsx";

const SUBJECTS = [
  "General enquiry",
  "Technical support",
  "Model / research question",
  "Fleet deployment",
  "Bug report",
];

export default function ContactPage() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    subject: SUBJECTS[0],
    message: "",
  });
  const [sending, setSending] = useState(false);
  const [sentId, setSentId] = useState(null);
  const [error, setError] = useState(null);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  async function handleSubmit(e) {
    e.preventDefault();
    setSending(true);
    setError(null);
    try {
      const res = await apiPost("/api/contact", form);
      setSentId(res.id);
      setForm({ name: "", email: "", subject: SUBJECTS[0], message: "" });
    } catch (err) {
      setError(err.message);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="vd-root" style={{ padding: "32px 24px 60px" }}>
      <PageHeader
        eyebrow="COMMUNICATION PORTAL"
        title="Contact & Technical Support"
        subtitle="User enquiries and technical support requests — answered by the VigiDriver team."
        right={
          <div style={{ display: "flex", alignItems: "center", gap: 9, color: C.navy }}>
            <ShieldCheck size={20} color={C.amber} />
            <span className="vd-display" style={{ fontSize: 17, fontWeight: 700, letterSpacing: 0.5 }}>
              VIGIDRIVER
            </span>
          </div>
        }
      />

      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: 30 }} className="vd-grid-2">
        {/* ---------- form ---------- */}
        <Card style={{ padding: 28 }}>
          {sentId ? (
            <div style={{ textAlign: "center", padding: "38px 10px" }}>
              <CheckCircle2 size={40} color={C.teal} />
              <div className="vd-display" style={{ fontWeight: 700, fontSize: 19, marginTop: 12, color: C.ink }}>
                Message received — ticket #{sentId}
              </div>
              <p style={{ color: C.inkMuted, fontSize: 13.5, marginTop: 8, lineHeight: 1.6, maxWidth: 420, margin: "8px auto 0" }}>
                Your enquiry was logged in the support inbox. The team typically responds
                within two working days; technical-support tickets include the model build
                (Eye-State CNN + MAR yawn detector) automatically.
              </p>
              <button
                className="vd-btn-ghost"
                onClick={() => setSentId(null)}
                style={{
                  marginTop: 18,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 8,
                  background: "transparent",
                  color: C.navy,
                  fontWeight: 600,
                  fontSize: 14,
                  padding: "11px 20px",
                  borderRadius: 8,
                  border: `1px solid ${C.border}`,
                  cursor: "pointer",
                }}
              >
                <MessageSquarePlus size={16} /> Send another message
              </button>
            </div>
          ) : (
            <form onSubmit={handleSubmit}>
              <PanelLabel icon={<Send size={15} />} text="New enquiry" />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 16 }} className="vd-form-2">
                <Field label="Name">
                  <input
                    required
                    minLength={2}
                    className="vd-input"
                    value={form.name}
                    onChange={set("name")}
                    placeholder="Your name"
                  />
                </Field>
                <Field label="Email">
                  <input
                    required
                    type="email"
                    className="vd-input"
                    value={form.email}
                    onChange={set("email")}
                    placeholder="you@example.com"
                  />
                </Field>
              </div>
              <Field label="Topic" style={{ marginTop: 14 }}>
                <select className="vd-input" value={form.subject} onChange={set("subject")}>
                  {SUBJECTS.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </Field>
              <Field label="Message" style={{ marginTop: 14 }}>
                <textarea
                  required
                  minLength={5}
                  className="vd-input"
                  style={{ minHeight: 130, resize: "vertical" }}
                  value={form.message}
                  onChange={set("message")}
                  placeholder={
                    form.subject === "Technical support" || form.subject === "Bug report"
                      ? "Describe the issue, the page it occurred on, and what you expected to happen."
                      : "Tell us what you'd like to know."
                  }
                />
              </Field>
              {error && (
                <div
                  style={{
                    marginTop: 12,
                    background: C.criticalSoft,
                    color: C.critical,
                    border: `1px solid ${C.critical}44`,
                    borderRadius: 8,
                    padding: "10px 14px",
                    fontSize: 13,
                    fontWeight: 600,
                  }}
                >
                  Could not send: {error}
                </div>
              )}
              <button
                type="submit"
                disabled={sending}
                style={{
                  marginTop: 18,
                  width: "100%",
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 8,
                  background: C.amber,
                  color: "#221304",
                  fontWeight: 700,
                  fontSize: 14.5,
                  padding: "13px 22px",
                  borderRadius: 8,
                  border: "none",
                  cursor: sending ? "wait" : "pointer",
                }}
              >
                {sending ? "Sending…" : "Send message"} <ArrowRight size={16} />
              </button>
            </form>
          )}
        </Card>

        {/* ---------- info column ---------- */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <Card style={{ padding: 22 }}>
            <PanelLabel icon={<Bug size={15} />} text="Support channels" />
            <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 12 }}>
              <ContactRow icon={<Mail size={17} />} title="Email" text="team7.vigidriver@amrita.edu" />
              <ContactRow icon={<Phone size={17} />} title="Faculty guide" text="Available via department office hours" />
              <ContactRow
                icon={<MapPin size={17} />}
                title="Department"
                text="Computer Science & Engineering, Amrita School of Computing"
              />
            </div>
          </Card>

          <Card style={{ padding: 22, background: C.surfaceAlt }}>
            <PanelLabel icon={<Bug size={15} />} text="Before filing a bug" />
            <ul style={{ margin: "12px 0 0", paddingLeft: 18, display: "flex", flexDirection: "column", gap: 8 }}>
              {[
                "Confirm the backend is running (python backend/run.py) — the dashboard shows BACKEND ONLINE when connected.",
                "Check that both model checkpoints exist under backend/weights/.",
                "If the camera feed is blank, verify no other app holds the webcam and re-select the camera index.",
                "Include the session id from the Analytics page when reporting alert behaviour.",
              ].map((t, i) => (
                <li key={i} style={{ fontSize: 13, color: C.inkMuted, lineHeight: 1.55 }}>{t}</li>
              ))}
            </ul>
          </Card>

          <Card style={{ padding: "14px 16px", display: "flex", gap: 10, alignItems: "center" }}>
            <Mail size={16} color={C.navy} />
            <span style={{ fontSize: 12.5, color: C.inkMuted }}>
              API base: <span className="vd-mono">{API_BASE}</span> — enquiries POST to{" "}
              <span className="vd-mono">/api/contact</span>.
            </span>
          </Card>

          <div style={{ display: "flex", gap: 12 }}>
            <Link to="/" className="vd-btn-ghost" style={{
              flex: 1,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
              background: "transparent",
              color: C.navy,
              fontWeight: 600,
              fontSize: 13.5,
              padding: "11px 16px",
              borderRadius: 8,
              border: `1px solid ${C.border}`,
              textDecoration: "none",
            }}>
              ← Back to overview
            </Link>
            <Link to="/dashboard" className="vd-btn-ghost" style={{
              flex: 1,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
              background: "transparent",
              color: C.navy,
              fontWeight: 600,
              fontSize: 13.5,
              padding: "11px 16px",
              borderRadius: 8,
              border: `1px solid ${C.border}`,
              textDecoration: "none",
            }}>
              Open dashboard →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children, style }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 6, ...style }}>
      <span style={{ fontSize: 13, fontWeight: 600, color: C.ink }}>{label}</span>
      {children}
    </label>
  );
}

function ContactRow({ icon, title, text }) {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center", border: `1px solid ${C.border}`, borderRadius: 9, padding: "12px 14px" }}>
      <div style={{ color: C.amber }}>{icon}</div>
      <div>
        <div style={{ fontSize: 11.5, color: C.inkDim }}>{title}</div>
        <div style={{ fontSize: 13.5, color: C.ink, fontWeight: 500 }}>{text}</div>
      </div>
    </div>
  );
}
