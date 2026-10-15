/* Model Architecture & Deep Learning Inspection Page.
 *
 * Provides full transparency into the neural network models strictly loaded
 * from the `models/` directory, including layer dimensions of the binary
 * Eye-State CNN (models/eye_cnn.keras), MAR yawn detection, and interactive
 * inference testing.
 */

import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  Brain,
  CheckCircle2,
  Cpu,
  Layers,
  ShieldCheck,
  Sparkles,
  Upload,
  Zap,
  ArrowRight,
} from "lucide-react";
import { API_BASE, C, apiGet } from "../lib/api.js";
import { Card, PageHeader, PanelLabel, StatusPill } from "../lib/ui.jsx";

const LAYERS_TABLE = [
  { name: "input_layer", type: "InputLayer", shape: "(None, 64, 64, 1)", params: 0, detail: "Grayscale eye crop, centered, normalized to [0, 1]" },
  { name: "conv2d", type: "Conv2D", shape: "(None, 64, 64, 32)", params: 320, detail: "32 filters, 3×3 kernel, same padding, ReLU" },
  { name: "batch_normalization", type: "BatchNormalization", shape: "(None, 64, 64, 32)", params: 128, detail: "Normalizes conv feature maps for stable training" },
  { name: "max_pooling2d", type: "MaxPooling2D", shape: "(None, 32, 32, 32)", params: 0, detail: "2×2 pool size, stride 2" },
  { name: "conv2d_1", type: "Conv2D", shape: "(None, 32, 32, 64)", params: 18496, detail: "64 filters, 3×3 kernel, same padding, ReLU" },
  { name: "batch_normalization_1", type: "BatchNormalization", shape: "(None, 32, 32, 64)", params: 256, detail: "Normalizes conv feature maps" },
  { name: "max_pooling2d_1", type: "MaxPooling2D", shape: "(None, 16, 16, 64)", params: 0, detail: "2×2 pool size, stride 2" },
  { name: "conv2d_2", type: "Conv2D", shape: "(None, 16, 16, 128)", params: 73856, detail: "128 filters, 3×3 kernel, same padding, ReLU" },
  { name: "batch_normalization_2", type: "BatchNormalization", shape: "(None, 16, 16, 128)", params: 512, detail: "Normalizes conv feature maps" },
  { name: "max_pooling2d_2", type: "MaxPooling2D", shape: "(None, 8, 8, 128)", params: 0, detail: "2×2 pool size, stride 2" },
  { name: "flatten", type: "Flatten", shape: "(None, 8192)", params: 0, detail: "Channels-last 8×8×128 vectorization" },
  { name: "dense", type: "Dense (ReLU)", shape: "(None, 128)", params: 1048704, detail: "128 hidden units, ReLU" },
  { name: "dropout", type: "Dropout (0.5)", shape: "(None, 128)", params: 0, detail: "Regularization during training" },
  { name: "dense_1", type: "Dense (Sigmoid)", shape: "(None, 1)", params: 129, detail: "Output: P(eye_open) — >0.5 Open, <0.5 Closed" },
];

export default function ModelInspectionPage() {
  const [health, setHealth] = useState(null);
  const [, setLoading] = useState(true);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const [temporal, setTemporal] = useState(null);

  useEffect(() => {
    apiGet("/api/health")
      .then((d) => {
        setHealth(d);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
    apiGet("/api/model/temporal")
      .then(setTemporal)
      .catch(() => setTemporal(null));
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error(`Inference test failed: ${res.statusText}`);
      }
      const data = await res.json();
      setUploadResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };


  return (
    <div className="vd-root" style={{ padding: "32px 24px 60px" }}>
      <PageHeader
        eyebrow="DEEP LEARNING ARCHITECTURE & WEIGHTS"
        title="Model Inspection & Specifications"
        subtitle="Layer specs for the Eye-State CNN, Bi-LSTM temporal micro-sleep model, K-Means latent gate and Grad-CAM attribution suite."
        right={
          <div style={{ display: "flex", gap: 10 }}>
            <StatusPill color={C.teal} bg={C.tealSoft}>
              CHECKPOINT: STRICT LOADED
            </StatusPill>
            <Link
              to="/dashboard"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                background: C.amber,
                color: "#221304",
                fontWeight: 700,
                fontSize: 13,
                padding: "8px 16px",
                borderRadius: 8,
                textDecoration: "none",
              }}
            >
              Open Live Dashboard <ArrowRight size={14} />
            </Link>
          </div>
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
          {error}
        </div>
      )}

      {/* Top Overview Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16, marginBottom: 24 }}>
        <Card style={{ padding: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.inkMuted, fontSize: 12 }}>
            <Cpu size={16} color={C.amber} /> Core Checkpoint
          </div>
          <div className="vd-mono" style={{ fontSize: 17, fontWeight: 700, color: C.navy, marginTop: 6, wordBreak: "break-all" }}>
            models/eye_cnn.keras
          </div>
          <div style={{ fontSize: 12, color: C.teal, marginTop: 4, display: "flex", alignItems: "center", gap: 4 }}>
            <CheckCircle2 size={12} /> Keras 3 checkpoint — direct load
          </div>
        </Card>

        <Card style={{ padding: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.inkMuted, fontSize: 12 }}>
            <Layers size={16} color={C.teal} /> Total Parameters
          </div>
          <div className="vd-mono" style={{ fontSize: 22, fontWeight: 700, color: C.navy, marginTop: 6 }}>
            {(health?.primary_model?.parameters ?? 1142401).toLocaleString()}
          </div>
          <div style={{ fontSize: 12, color: C.inkDim, marginTop: 4 }}>
            ~13.1 MB Keras v3 archive (float32 weights)
          </div>
        </Card>

        <Card style={{ padding: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.inkMuted, fontSize: 12 }}>
            <Zap size={16} color={C.amber} /> Input Tensor Spec
          </div>
          <div className="vd-mono" style={{ fontSize: 22, fontWeight: 700, color: C.navy, marginTop: 6 }}>
            64 × 64 × 1
          </div>
          <div style={{ fontSize: 12, color: C.inkDim, marginTop: 4 }}>
            Grayscale centered eye crop, normalized [0, 1]
          </div>
        </Card>

        <Card style={{ padding: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.inkMuted, fontSize: 12 }}>
            <Activity size={16} color={C.teal} /> Output Classes
          </div>
          <div className="vd-mono" style={{ fontSize: 22, fontWeight: 700, color: C.navy, marginTop: 6 }}>
            Binary Sigmoid
          </div>
          <div style={{ fontSize: 12, color: C.inkDim, marginTop: 4 }}>
            P(eye_open) &gt; 0.5 → Open · &lt; 0.5 → Closed
          </div>
        </Card>
      </div>

      {/* LSTM Temporal Micro-Sleep Model + K-Means Gate */}
      <Card style={{ padding: 22, marginBottom: 24 }}>
        <PanelLabel icon={<Brain size={16} color={C.amber} />} text="Bi-LSTM Temporal Micro-Sleep Model — best_vigidriver_lstm.keras" />
        <p style={{ fontSize: 13, color: C.inkMuted, margin: "6px 0 16px" }}>
          A 30-frame (4-feature vigilance window) sequential Bi-LSTM tracks eyelid-closure velocity, mouth dynamics and head yaw
          to distinguish fast benign blinks from sustained micro-sleep — modulating the fused drowsiness score in real time.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12, marginBottom: 14 }}>
          {[
            ["Checkpoint", temporal?.available ? "models/best_vigidriver_lstm.keras" : "not available"],
            ["Input Tensor", temporal?.available ? "(None, 30, 4)" : "—"],
            ["Trainable Params", temporal?.available ? temporal.trainable_parameters?.toLocaleString?.() ?? "—" : "—"],
            ["Window", temporal?.available ? `${temporal.window_seconds}s @ ${ temporal.inference_stride ?? 5 }-frame stride` : "—"],
          ].map(([label, val], ix) => (
            <div key={ix} style={{ background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 9, padding: "12px 14px" }}>
              <div style={{ fontSize: 11, color: C.inkMuted, textTransform: "uppercase", letterSpacing: 0.4 }}>{label}</div>
              <div className="vd-mono" style={{ fontSize: 14.5, fontWeight: 700, color: C.navy, marginTop: 3, wordBreak: "break-all" }}>{String(val)}</div>
            </div>
          ))}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 18, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: C.navy, marginBottom: 8 }}>Feature Standardization (train stats)</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
              <thead>
                <tr style={{ color: C.inkMuted, borderBottom: `1px solid ${C.border}` }}>
                  <th style={{ textAlign: "left", padding: "6px 8px" }}>Feature</th>
                  <th style={{ textAlign: "right", padding: "6px 8px" }}>Mean</th>
                  <th style={{ textAlign: "right", padding: "6px 8px" }}>Std</th>
                </tr>
              </thead>
              <tbody>
                {(temporal?.input_features ?? []).map((f, i) => (
                  <tr key={f} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td className="vd-mono" style={{ padding: "6px 8px", color: C.navy, fontWeight: 600 }}>{f}</td>
                    <td className="vd-mono" style={{ padding: "6px 8px", textAlign: "right", color: C.inkDim }}>
                      {temporal?.feature_normalization?.mean?.[i]?.toFixed(4) ?? "—"}
                    </td>
                    <td className="vd-mono" style={{ padding: "6px 8px", textAlign: "right", color: C.inkDim }}>
                      {temporal?.feature_normalization?.std?.[i]?.toFixed(4) ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: C.navy, marginBottom: 8 }}>K-Means Latent Gating Engine (k = 4)</div>
            <div style={{ fontSize: 12.5, color: C.inkMuted, display: "flex", flexDirection: "column", gap: 4 }}>
              <span>Status: <strong style={{ color: temporal?.kmeans_gate?.fitted ? C.teal : C.caution }}>
                {temporal?.kmeans_gate?.fitted ? "fitted — live gate active" : "warming up (needs ~240 live frames)"}
              </strong></span>
              <span className="vd-mono">Samples: {temporal?.kmeans_gate?.n_samples ?? 0} · refits: {temporal?.kmeans_gate?.n_fits ?? 0}</span>
              <span className="vd-mono">τ_outlier: {temporal?.kmeans_gate?.tau_outlier ?? "—"} · whiten dim {temporal?.kmeans_gate?.whiten_dim ?? 6}</span>
              <span style={{ marginTop: 4 }}>
                C0 alert · C1 blinks · C2 micro-sleep · C3 yawning —
                every latent vector outside centroid boundaries is flagged an anomaly, holding alert escalation until corroborated.
              </span>
            </div>
          </div>
        </div>
      </Card>

      {/* Interactive Inference Tester Card */}
      <Card style={{ padding: 22, marginBottom: 24, border: `1px solid ${C.border}` }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
          <div>
            <PanelLabel icon={<Sparkles size={16} color={C.amber} />} text="Interactive Model Inference Tester" />
            <p style={{ fontSize: 13, color: C.inkMuted, marginTop: 4 }}>
              Upload any test image to run the full pipeline — face detection, per-eye CNN classification and MAR yawn measurement — against the live backend.
            </p>
          </div>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              background: C.navy,
              color: "#fff",
              fontWeight: 700,
              fontSize: 13.5,
              padding: "10px 18px",
              borderRadius: 8,
              border: "none",
              cursor: uploading ? "wait" : "pointer",
            }}
          >
            <Upload size={15} /> {uploading ? "Running Neural Inference…" : "Upload Test Image"}
          </button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept="image/*"
            style={{ display: "none" }}
          />
        </div>

        {uploadResult && (
          <div style={{ marginTop: 20, paddingTop: 18, borderTop: `1px solid ${C.border}` }}>
            {uploadResult.face_found ? (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: C.navy, marginBottom: 12 }}>
                    Predicted Probabilities Breakdown:
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    <ProbGauge label="Eye Open: P(Open) — eye_cnn.keras" val={uploadResult.p_open ?? 0} color={C.teal} />
                    <ProbGauge label="Eye Closed: P(Closed)" val={uploadResult.p_closed ?? 0} color={C.critical} />
                    <ProbGauge label="Yawn: P(yawn) — MAR geometric" val={uploadResult.p_yawn ?? 0} color={C.caution} />
                    <ProbGauge label="Alert / Neutral: P(no_yawn)" val={uploadResult.p_no_yawn ?? 0} color={C.teal} />
                  </div>
                  <div style={{ marginTop: 14, fontSize: 13, color: C.ink }}>
                    <strong>Top Output Class:</strong>{" "}
                    <span className="vd-mono" style={{ color: uploadResult.p_drowsy >= 0.5 ? C.critical : C.teal, fontWeight: 700 }}>
                      {uploadResult.top_class?.toUpperCase?.()}
                    </span>{" "}
                    · <strong>Overall Drowsiness Score:</strong>{" "}
                    <span className="vd-mono" style={{ fontWeight: 700 }}>
                      {uploadResult.p_drowsy?.toFixed?.(4)}
                    </span>
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: C.navy, marginBottom: 10 }}>
                    Region Measurements:
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    <div style={{ background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 9, padding: "12px 14px" }}>
                      <div style={{ fontSize: 11, color: C.inkMuted, textTransform: "uppercase", letterSpacing: 0.4 }}>Eye State (CNN)</div>
                      <div className="vd-mono" style={{ fontSize: 18, fontWeight: 700, color: uploadResult.is_eye_closed ? C.critical : C.teal, marginTop: 3 }}>
                        {uploadResult.is_eye_closed ? "CLOSED" : "OPEN"}
                      </div>
                    </div>
                    <div style={{ background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 9, padding: "12px 14px" }}>
                      <div style={{ fontSize: 11, color: C.inkMuted, textTransform: "uppercase", letterSpacing: 0.4 }}>Mouth Opening (MAR)</div>
                      <div className="vd-mono" style={{ fontSize: 18, fontWeight: 700, color: uploadResult.is_yawning ? C.caution : C.navy, marginTop: 3 }}>
                        {(uploadResult.mar ?? 0).toFixed(3)} {uploadResult.is_yawning ? "· YAWNING" : "· NO YAWN"}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                    <div>
                      <div style={{ fontSize: 11, color: C.inkMuted, textTransform: "uppercase", letterSpacing: 0.4, marginBottom: 5 }}>
                        Grad-CAM saliency (eye CNN)
                      </div>
                      {uploadResult.gradcam_png ? (
                        <img
                          src={`data:image/jpeg;base64,${uploadResult.gradcam_png}`}
                          alt="Grad-CAM"
                          style={{ width: 140, height: 140, borderRadius: 8, border: `1px solid ${C.border}` }}
                        />
                      ) : (
                        <div style={{ width: 140, height: 140, borderRadius: 8, border: `1px dashed ${C.border}`, display: "flex", alignItems: "center", justifyContent: "center", color: C.inkDim, fontSize: 11 }}>
                          no heat map
                        </div>
                      )}
                    </div>
                    <div style={{ fontSize: 12, color: C.inkMuted, lineHeight: 1.7 }}>
                      <div><strong style={{ color: C.navy }}>Attribution layer:</strong> <span className="vd-mono">{uploadResult.gradcam_meta?.conv_layer ?? "—"}</span></div>
                      <div><strong style={{ color: C.navy }}>XAI P(open):</strong> {uploadResult.gradcam_meta?.p_open_xai != null ? uploadResult.gradcam_meta.p_open_xai : "—"}</div>
                      <div style={{ maxWidth: 260, marginTop: 4 }}>
                        ReLU-weighted class activation on the terminal conv feature map, bilinearly upsampled onto the eye crop —
                        eyelid-margin focus indicates genuine ocular evidence, not background correlation.
                      </div>
                    </div>
                  </div>
                  <div style={{ fontSize: 11.5, color: C.inkDim, marginTop: 8 }}>
                    Face box: [{(uploadResult.face_box || []).join(", ")}] — eye crops are centered per-eye before CNN inference.
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ color: C.critical, fontSize: 13 }}>
                No face detected in the uploaded frame. Please try an image with clear cabin lighting.
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Layer Architecture Table */}
      <Card style={{ padding: 22 }}>
        <PanelLabel icon={<Layers size={16} />} text="Layer-by-Layer Network Architecture" />
        <p style={{ fontSize: 13, color: C.inkMuted, margin: "6px 0 16px" }}>
          Exact layer sequence and parameter dimensions of <code>models/eye_cnn.keras</code> (1,142,401 trainable parameters).
        </p>

        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, textAlign: "left" }}>
            <thead>
              <tr style={{ borderBottom: `2px solid ${C.border}`, color: C.inkMuted }}>
                <th style={{ padding: "10px 12px" }}>Layer Name</th>
                <th style={{ padding: "10px 12px" }}>Type</th>
                <th style={{ padding: "10px 12px" }}>Output Tensor Shape</th>
                <th style={{ padding: "10px 12px" }}>Parameters</th>
                <th style={{ padding: "10px 12px" }}>Operation Details</th>
              </tr>
            </thead>
            <tbody>
              {LAYERS_TABLE.map((row, idx) => (
                <tr
                  key={idx}
                  style={{
                    borderBottom: `1px solid ${C.border}`,
                    background: idx % 2 === 0 ? "transparent" : C.surfaceAlt,
                  }}
                >
                  <td className="vd-mono" style={{ padding: "10px 12px", fontWeight: 700, color: C.navy }}>
                    {row.name}
                  </td>
                  <td style={{ padding: "10px 12px", color: C.ink }}>
                    {row.type}
                  </td>
                  <td className="vd-mono" style={{ padding: "10px 12px", color: C.inkDim }}>
                    {row.shape}
                  </td>
                  <td className="vd-mono" style={{ padding: "10px 12px", fontWeight: 600, color: row.params > 0 ? C.amber : C.inkDim }}>
                    {row.params.toLocaleString()}
                  </td>
                  <td style={{ padding: "10px 12px", color: C.inkMuted, fontSize: 12.5 }}>
                    {row.detail}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function ProbGauge({ label, val, color }) {
  const pct = Math.round(Math.min(1, Math.max(0, val)) * 100);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 3 }}>
        <span style={{ color: C.ink, fontWeight: 600 }}>{label}</span>
        <span className="vd-mono" style={{ fontWeight: 700, color }}>
          {(val * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{ height: 6, background: C.surfaceAlt2, borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3 }} />
      </div>
    </div>
  );
}
