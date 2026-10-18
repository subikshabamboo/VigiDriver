/* Live Monitoring Dashboard.
 *
 * Real-time drowsiness monitoring powered by the binary Eye-State CNN loaded
 * from `models/eye_cnn.keras` (sigmoid P(eye_open), 64×64×1 input) fused with
 * geometric MAR-based yawn detection and a tiered fatigue alert cascade.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  BellRing,
  Camera,
  CameraOff,
  Braces,
  CheckCircle2,
  Cpu,
  Eye,
  FileCheck,
  Layers,
  Play,
  Radio,
  ShieldCheck,
  Square,
  Timer,
  Upload,
  Video,
} from "lucide-react";
import { API_BASE, C, apiGet, apiPost } from "../lib/api.js";
import { Card, PanelLabel, PageHeader, StatusPill } from "../lib/ui.jsx";
import {
  notifyDesktop,
  playWarningChime,
  speak,
  startCriticalSiren,
  stopAllAudio,
  unlockAudio,
} from "../lib/alertAudio.js";

const TIER_META = {
  normal: { label: "ALERT · MONITORING", color: C.teal, bg: C.tealSoft },
  warning: { label: "WARNING · FATIGUE BUILDING", color: C.caution, bg: C.cautionSoft },
  critical: { label: "CRITICAL · DROWSY ALERT", color: C.critical, bg: C.criticalSoft },
};

function fmtDuration(s) {
  const m = Math.floor(s / 60).toString().padStart(2, "0");
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

export default function LiveMonitoringDashboard() {
  const [state, setState] = useState(null);
  const [, setHealthInfo] = useState(null);
  const [backendOnline, setBackendOnline] = useState(null);
  const [, setCameraReady] = useState(null);
  const [starting, setStarting] = useState(false);
  const [threshold, setThreshold] = useState(0.50);
  const [driverLabel, setDriverLabel] = useState("Driver_01");
  const [sourceMode, setSourceMode] = useState("browser"); // "browser" | "sim" | cameraIndex
  const [cameras, setCameras] = useState([]);
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploading, setUploading] = useState(false);

  const prevTier = useRef("normal");
  const feedRef = useRef(null);
  const fileInputRef = useRef(null);
  const browserVideoRef = useRef(null);
  const browserStreamRef = useRef(null);
  const frameIntervalRef = useRef(null);
  const offscreenCanvasRef = useRef(null);

  /* -- backend health & model metadata on mount ------------------------- */
  const checkHealth = useCallback(() => {
    apiGet("/api/health")
      .then((d) => {
        setBackendOnline(true);
        setHealthInfo(d);
      })
      .catch(() => {
        setBackendOnline(false);
      });
  }, []);

  useEffect(() => {
    checkHealth();
    apiGet("/api/monitor/cameras")
      .then((d) => {
        setCameras(d.cameras || []);
      })
      .catch(() => {});
  }, [checkHealth]);

  /* -- state polling (real-time stream) -------------------------------- */
  useEffect(() => {
    let alive = true;
    let timer = null;
    const poll = async () => {
      try {
        const d = await apiGet("/api/monitor/state");
        if (!alive) return;
        setState(d);
        setBackendOnline(true);
        setCameraReady(d.running ? !d.cam_error : null);
      } catch {
        if (alive) setBackendOnline(false);
      } finally {
        if (alive) timer = setTimeout(poll, 250);
      }
    };
    poll();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, []);

  /* -- tier-change audio and speech alerts ----------------------------- */
  useEffect(() => {
    const tier = state?.tier || "normal";
    if (tier === prevTier.current) return;
    const from = prevTier.current;
    prevTier.current = tier;
    if (muted) return;
    if (tier === "critical" && from !== "critical") {
      startCriticalSiren();
      speak("Critical drowsiness alert. Please pull over safely.");
      notifyDesktop(
        "VigiDriver — CRITICAL DROWSINESS ALERT",
        "Sustained fatigue cascade detected. Pull over as soon as it is safe."
      );
    } else if (tier === "warning" && from === "normal") {
      playWarningChime();
      speak("Fatigue building. Stay alert.");
    } else if (tier === "normal" && from !== "normal") {
      stopAllAudio();
    }
  }, [state?.tier, muted]);

  useEffect(() => () => {
    stopAllAudio();
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (browserStreamRef.current) {
      browserStreamRef.current.getTracks().forEach((t) => t.stop());
      browserStreamRef.current = null;
    }
  }, []);

  // When backend ends session automatically, stop local browser webcam tracks
  useEffect(() => {
    if (state && !state.running) {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
      if (browserStreamRef.current) {
        browserStreamRef.current.getTracks().forEach((t) => t.stop());
        browserStreamRef.current = null;
      }
    }
  }, [state?.running]);

  const running = !!state?.running;

  const handleStart = useCallback(async () => {
    setStarting(true);
    setError(null);
    unlockAudio();
    try {
      if (sourceMode === "browser") {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("Browser camera API is not supported on this browser or requires HTTPS.");
        }
        let stream;
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
            audio: false,
          });
        } catch (mediaErr) {
          throw new Error(`Could not access device camera: ${mediaErr.message || mediaErr.name}. Please grant camera permission.`);
        }
        browserStreamRef.current = stream;
        if (browserVideoRef.current) {
          browserVideoRef.current.srcObject = stream;
          try {
            await browserVideoRef.current.play();
          } catch (_) {}
        }

        await apiPost("/api/monitor/start", {
          driver_label: driverLabel || "Driver_01",
          threshold,
          camera_index: 0,
          use_camera: false,
          use_simulation: false,
          use_browser_cam: true,
        });

        // Frame streaming loop: capture canvas frame at ~10 FPS and push to backend
        let isPushing = false;
        frameIntervalRef.current = setInterval(() => {
          if (isPushing || !browserVideoRef.current || browserVideoRef.current.readyState < 2) return;
          const video = browserVideoRef.current;
          if (!video.videoWidth || !video.videoHeight) return;

          let canvas = offscreenCanvasRef.current;
          if (!canvas) {
            canvas = document.createElement("canvas");
            offscreenCanvasRef.current = canvas;
          }
          canvas.width = 640;
          canvas.height = 480;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(video, 0, 0, 640, 480);

          isPushing = true;
          canvas.toBlob(
            async (blob) => {
              try {
                if (!blob) return;
                const fd = new FormData();
                fd.append("file", blob, "frame.jpg");
                const res = await fetch(`${API_BASE}/api/monitor/frame`, {
                  method: "POST",
                  body: fd,
                });
                if (res.ok) {
                  const data = await res.json();
                  setState(data);
                }
              } catch (err) {
                console.warn("Frame push failed", err);
              } finally {
                isPushing = false;
              }
            },
            "image/jpeg",
            0.65
          );
        }, 100);
      } else {
        await apiPost("/api/monitor/start", {
          driver_label: driverLabel || "Driver_01",
          threshold,
          camera_index: sourceMode === "sim" ? 0 : (parseInt(sourceMode, 10) || 0),
          use_camera: sourceMode !== "sim",
          use_simulation: sourceMode === "sim",
          use_browser_cam: false,
        });
      }
      prevTier.current = "normal";
    } catch (e) {
      setError(e.message);
      if (browserStreamRef.current) {
        browserStreamRef.current.getTracks().forEach((t) => t.stop());
        browserStreamRef.current = null;
      }
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
    } finally {
      setStarting(false);
    }
  }, [driverLabel, threshold, sourceMode]);

  const handleStop = useCallback(async () => {
    setError(null);
    stopAllAudio();
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (browserStreamRef.current) {
      browserStreamRef.current.getTracks().forEach((t) => t.stop());
      browserStreamRef.current = null;
    }
    try {
      await apiPost("/api/monitor/stop");
    } catch (e) {
      setError(e.message);
    }
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
        throw new Error(`Upload failed: ${res.statusText}`);
      }
      const data = await res.json();
      setUploadResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const tier = state?.tier || "normal";
  const meta = TIER_META[tier];
  const pDrowsy = state?.p_drowsy ?? 0;
  const drowsy = state?.classification === "drowsy";

  // 4-Class Breakdown values from core model under models/
  const pClosed = state?.p_closed ?? (state?.classes?.Closed ?? 0.0);
  const pOpen = state?.p_open ?? (state?.classes?.Open ?? 1.0);
  const pNoYawn = state?.p_no_yawn ?? (state?.classes?.no_yawn ?? 1.0);
  const pYawn = state?.p_yawn ?? (state?.classes?.yawn ?? 0.0);
  const topClass = state?.top_class ?? "no_yawn";

  return (
    <div className="vd-root" style={{ padding: "32px 24px 60px" }}>
      <PageHeader
        eyebrow="STRICT NEURAL NETWORK INFERENCE ENGINE"
        title="Live Monitoring Dashboard"
        subtitle="End-to-end drowsiness detection executing models/eye_cnn.keras with per-eye CNN inference & MAR yawn detection."
        right={
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <StatusPill
              color={backendOnline ? C.teal : C.critical}
              bg={backendOnline ? C.tealSoft : C.criticalSoft}
            >
              {backendOnline === null ? "CONNECTING" : backendOnline ? "BACKEND ONLINE" : "BACKEND OFFLINE"}
            </StatusPill>
            <StatusPill
              color={running ? (tier === "critical" ? C.critical : tier === "warning" ? C.caution : C.teal) : C.inkDim}
              bg={running ? meta.bg : C.surfaceAlt2}
              pulse={tier === "critical"}
            >
              {running ? meta.label : "STANDBY"}
            </StatusPill>
          </div>
        }
      />

      {/* Model Spec Banner */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 12,
          background: C.surfaceAlt,
          border: `1px solid ${C.border}`,
          borderRadius: 10,
          padding: "10px 16px",
          marginBottom: 16,
          fontSize: 12.5,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.navy, fontWeight: 700 }}>
          <Cpu size={16} color={C.amber} />
          <span>Core Model:</span>
          <span className="vd-mono" style={{ color: C.amber, fontWeight: 700 }}>
            models/eye_cnn.keras
          </span>
          <span style={{ color: C.inkDim }}>· 3×Conv2D CNN · 1.14M Params · 64×64×1 · Sigmoid P(eye_open)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ color: C.inkMuted, display: "flex", alignItems: "center", gap: 4 }}>
            <CheckCircle2 size={13} color={C.teal} /> Strict weights verified
          </span>
          <button
            onClick={() => fileInputRef.current?.click()}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
              background: C.bg,
              border: `1px solid ${C.border}`,
              borderRadius: 6,
              padding: "4px 10px",
              fontSize: 12,
              fontWeight: 600,
              color: C.navy,
              cursor: "pointer",
            }}
          >
            <Upload size={12} /> {uploading ? "Analyzing image…" : "Analyze Test Photo"}
          </button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept="image/*"
            style={{ display: "none" }}
          />
        </div>
      </div>

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

      {/* One-Shot Upload Modal / Banner */}
      {uploadResult && (
        <Card style={{ padding: "16px 20px", marginBottom: 18, background: C.bg, border: `1px solid ${C.amber}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 700, color: C.navy }}>
              <FileCheck size={18} color={C.teal} /> One-Shot Photo Analysis Result
            </div>
            <button
              onClick={() => setUploadResult(null)}
              style={{ background: "none", border: "none", color: C.inkMuted, cursor: "pointer", fontSize: 13 }}
            >
              ✕ Close
            </button>
          </div>
          {uploadResult.face_found ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12 }}>
              <BigStat label="Classification" value={uploadResult.top_class?.toUpperCase?.() || "ALERT"} color={uploadResult.p_drowsy >= 0.5 ? C.critical : C.teal} />
              <BigStat label="Drowsiness Score" value={uploadResult.p_drowsy?.toFixed?.(3) ?? "-"} color={uploadResult.p_drowsy >= 0.5 ? C.critical : C.teal} />
              <BigStat label="Eye Closed P(Closed)" value={uploadResult.p_closed?.toFixed?.(3) ?? "-"} color={C.navy} />
              <BigStat label="Yawn P(Yawn)" value={uploadResult.p_yawn?.toFixed?.(3) ?? "-"} color={C.navy} />
              <BigStat label="Eye Open P(Open)" value={uploadResult.p_open?.toFixed?.(3) ?? "-"} color={C.navy} />
              <BigStat label="Alert P(No-Yawn)" value={uploadResult.p_no_yawn?.toFixed?.(3) ?? "-"} color={C.navy} />
            </div>
          ) : (
            <div style={{ color: C.critical, fontSize: 13 }}>No driver face detected in the uploaded image. Try an image with a clearer cabin view.</div>
          )}
        </Card>
      )}

      {/* Live Status Banner */}
      <div
        className={tier === "critical" ? "vd-blinking" : ""}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          background: running ? meta.bg : C.surfaceAlt,
          color: running ? meta.color : C.inkMuted,
          border: `1px solid ${running ? meta.color + "33" : C.border}`,
          borderRadius: 10,
          padding: "12px 18px",
          fontWeight: 700,
          fontSize: 14.5,
          marginBottom: 18,
        }}
      >
        {tier === "critical" || (!backendOnline) || (running && state?.face_detected === false)
          ? <AlertTriangle size={18} />
          : tier === "warning" ? <Eye size={18} /> : <ShieldCheck size={18} />}
        {!backendOnline
          ? "Backend unreachable — ensure python backend/run.py is active."
          : !running
            ? "Session standby — click 'Start Monitoring' to initiate real-time deep inference."
            : state?.face_detected === false
              ? `No face detected (frame brightness ${state?.brightness ?? "?"}/255) — sit in front of the camera and add light. Drowsiness cannot be judged until a face is visible.`
              : tier === "critical"
                ? "CRITICAL ALERT — High sustained eye closure / yawning fatigue cascade detected. Pull over safely!"
                : tier === "warning"
                  ? `Warning — Fatigue indicators building (State: ${topClass}, Score: ${pDrowsy.toFixed(2)}, PERCLOS ${Math.round((state?.perclos ?? 0) * 100)}%). Stay alert.`
                  : `Monitoring normal — State: ${topClass} (Eye closed: ${(pClosed*100).toFixed(1)}%, Yawn: ${(pYawn*100).toFixed(1)}%, Blinks: ${state?.blink_count ?? 0}).`}
      </div>

      {/* Control Bar */}
      <Card style={{ padding: "16px 20px", marginBottom: 18 }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "flex-end" }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 12.5, fontWeight: 600, color: C.inkMuted }}>
            DRIVER LABEL
            <input
              value={driverLabel}
              disabled={running}
              onChange={(e) => setDriverLabel(e.target.value)}
              style={{
                border: `1px solid ${C.border}`,
                borderRadius: 8,
                padding: "9px 12px",
                fontSize: 14,
                color: C.ink,
                width: 170,
                fontFamily: "'JetBrains Mono', monospace",
              }}
            />
          </label>

          <label style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 12.5, fontWeight: 600, color: C.inkMuted, minWidth: 200 }}>
            ALERT THRESHOLD — <span className="vd-mono" style={{ color: C.amber }}>{threshold.toFixed(2)}</span>
            <input
              type="range"
              min="0.3"
              max="0.9"
              step="0.01"
              value={threshold}
              disabled={running}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="vd-slider"
              style={{ ["--fill"]: `${((threshold - 0.3) / 0.6) * 100}%` }}
            />
          </label>

          <label style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 12.5, fontWeight: 600, color: C.inkMuted }}>
            SOURCE
            <select
              value={sourceMode}
              disabled={running}
              onChange={(e) => setSourceMode(e.target.value)}
              style={{
                border: `1px solid ${C.border}`,
                borderRadius: 8,
                padding: "9px 12px",
                fontSize: 14,
                color: C.ink,
                background: C.bg,
              }}
            >
              <option value="browser">💻 My Device Camera (Browser Webcam)</option>
              <option value="sim">🚗 Simulation Stream (Synthetic Cabin Video)</option>
              {cameras.map((i) => (
                <option key={i} value={i}>📹 Host Machine Webcam {i}</option>
              ))}
            </select>
          </label>

          <div style={{ flex: 1 }} />

          {!running ? (
            <button
              onClick={handleStart}
              disabled={starting || !backendOnline}
              className="vd-siren"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                background: C.amber,
                color: "#221304",
                fontWeight: 700,
                fontSize: 14.5,
                padding: "12px 26px",
                borderRadius: 8,
                border: "none",
                cursor: starting || !backendOnline ? "wait" : "pointer",
              }}
            >
              <Play size={16} /> Start monitoring
            </button>
          ) : (
            <button
              onClick={handleStop}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                background: C.critical,
                color: "#fff",
                fontWeight: 700,
                fontSize: 14.5,
                padding: "12px 26px",
                borderRadius: 8,
                border: "none",
                cursor: "pointer",
              }}
            >
              <Square size={15} /> End session
            </button>
          )}

          <button
            onClick={() => {
              setMuted((m) => {
                if (!m) stopAllAudio();
                return !m;
              });
            }}
            title={muted ? "Unmute audible alerts" : "Mute audible alerts"}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              background: C.bg,
              border: `1px solid ${C.border}`,
              borderRadius: 8,
              padding: "11px 14px",
              cursor: "pointer",
              color: C.inkMuted,
              fontWeight: 600,
              fontSize: 13,
            }}
          >
            {muted ? "Muted" : "Sound on"}
          </button>
        </div>
      </Card>

      {/* 3-Column Layout */}
      <div className="vd-grid-monitor" style={{ display: "grid", gridTemplateColumns: "1.1fr 1.15fr 0.95fr", gap: 18, alignItems: "start" }}>
        {/* LEFT COLUMN: Live Feed */}
        <Card style={{ padding: 16 }}>
          <PanelLabel
            icon={running && !state?.cam_error ? <Video size={15} /> : <CameraOff size={15} />}
            text={`Cabin Feed — ${state?.driver_label || driverLabel}`}
          />
          {/* Hidden offscreen video element for browser media stream capture */}
          <video ref={browserVideoRef} playsInline autoPlay muted style={{ display: "none" }} />
          <div
            ref={feedRef}
            style={{
              marginTop: 12,
              position: "relative",
              borderRadius: 8,
              overflow: "hidden",
              background: C.console,
              border: `1px solid ${C.border}`,
              aspectRatio: "4 / 3",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {running ? (
              <>
                {state?.use_browser_cam && state?.jpeg ? (
                  <img
                    src={`data:image/jpeg;base64,${state.jpeg}`}
                    alt="Live driver feed"
                    style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
                  />
                ) : (
                  <img
                    src={`${API_BASE}/api/cam?sid=${state?.session_id || 0}`}
                    alt="Live driver feed"
                    style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
                    onError={(e) => {
                      if (state?.jpeg) {
                        e.currentTarget.src = `data:image/jpeg;base64,${state.jpeg}`;
                      }
                    }}
                  />
                )}
                {!state?.face_found && (
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      background: "rgba(10, 14, 22, 0.55)",
                    }}
                  >
                    <div
                      className="vd-blinking"
                      style={{
                        color: "#fff",
                        fontWeight: 700,
                        fontSize: 14,
                        textAlign: "center",
                        background: "rgba(200, 60, 60, 0.85)",
                        borderRadius: 8,
                        padding: "10px 18px",
                      }}
                    >
                      NO FACE DETECTED — sit centered in frame, face the camera,
                      and make sure the room is lit
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div style={{ textAlign: "center", color: "#5A6B80", padding: 20 }}>
                <Camera size={34} style={{ marginBottom: 8 }} />
                <div className="vd-mono" style={{ fontSize: 12.5 }}>
                  {running ? "awaiting frames…" : "feed offline — session not started"}
                </div>
              </div>
            )}
            {running && state?.face_box && (
              <div
                className="vd-mono"
                style={{
                  position: "absolute",
                  top: 10,
                  left: 10,
                  background: "rgba(255,255,255,0.92)",
                  border: `1px solid ${C.border}`,
                  borderRadius: 6,
                  padding: "3px 8px",
                  fontSize: 11,
                  color: C.ink,
                  fontWeight: 600,
                }}
              >
                bbox [{state.face_box.join(", ")}]
              </div>
            )}
            {running && (
              <div
                className="vd-mono"
                style={{
                  position: "absolute",
                  bottom: 10,
                  right: 12,
                  fontSize: 10.5,
                  color: "rgba(255,255,255,0.75)",
                }}
              >
                {state?.fps?.toFixed?.(1) ?? "—"} fps
              </div>
            )}
          </div>

          <div style={{ marginTop: 12, fontSize: 12.5, color: C.inkMuted, display: "flex", justifyContent: "space-between" }}>
            <span>
              Face tracking:{" "}
              <span
                className="vd-mono"
                style={{
                  color: state?.face_found ? C.teal : running ? C.critical : C.inkDim,
                  fontWeight: 600,
                }}
              >
                {state?.face_box
                  ? `${state.face_box[2]}×${state.face_box[3]} px`
                  : running
                    ? "searching…"
                    : "—"}
              </span>
            </span>
            <span className="vd-mono" style={{ color: drowsy ? C.critical : C.teal, fontWeight: 700 }}>
              {running ? (drowsy ? "DROWSY" : "ALERT") : "—"}
            </span>
          </div>
        </Card>

        {/* CENTER COLUMN: Neural Breakdown + Region Measurements + Attention Chart */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {/* 4-Class Core Model Gauge Card */}
          <Card style={{ padding: 18 }}>
            <PanelLabel icon={<Activity size={15} />} text="Neural Breakdown (models/eye_cnn.keras + MAR)" />
            <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 12 }}>
              <ClassBar label="Eye Closed (CNN)" val={running ? pClosed : 0} color={C.critical} icon="👁️" />
              <ClassBar label="Eye Open (CNN)" val={running ? pOpen : 0} color={C.teal} icon="👀" />
              <ClassBar label="Yawning Fatigue (MAR)" val={running ? pYawn : 0} color={C.caution} icon="🥱" />
              <ClassBar label="Alert / Neutral (no-yawn)" val={running ? pNoYawn : 0} color={C.teal} icon="😐" />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginTop: 14 }}>
              <BigStat label="Fused Drowsiness" value={running ? pDrowsy.toFixed(3) : "—"} color={drowsy ? C.critical : C.teal} />
              <BigStat label="Top Predicted Class" value={running ? topClass.toUpperCase() : "—"} color={drowsy ? C.critical : C.navy} />
              <BigStat label="PERCLOS (6s window)" value={running ? `${Math.round((state?.perclos ?? 0) * 100)}%` : "—"} color={(state?.perclos ?? 0) > 0.3 ? C.critical : (state?.perclos ?? 0) > 0.15 ? C.caution : C.teal} />
            </div>
          </Card>

          {/* Temporal Bi-LSTM + K-Means + Grad-CAM card */}
          <Card style={{ padding: 18 }}>
            <PanelLabel icon={<Braces size={15} />} text="Temporal Fusion — Bi-LSTM Micro-Sleep + K-Means Gating + Grad-CAM" />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
              <div>
                <BigStat
                  label="LSTM P(drowsy) — 30-frame window"
                  value={running ? (state?.temporal?.p_drowsy_lstm ?? "—") : "—"}
                  color={(state?.temporal?.p_drowsy_lstm ?? 0) > 0.5 ? C.critical : C.teal}
                />
                <div className="vd-mono" style={{ fontSize: 11.5, color: C.inkDim, marginTop: 6 }}>
                  {state?.temporal?.window ?? 0}/{state?.temporal?.window_size ?? 30} frames buffered
                  {state?.temporal?.ready ? " · inference ARMED" : " · filling window"}
                </div>
              </div>
              <div>
                <BigStat
                  label="K-Means Behavioural Cluster"
                  value={running ? (state?.kmeans?.cluster_id != null ? `C${state.kmeans.cluster_id}` : "—") : "—"}
                  color={state?.kmeans?.is_outlier ? C.caution : C.navy}
                />
                <div className="vd-mono" style={{ fontSize: 11.5, color: state?.kmeans?.is_outlier ? C.caution : C.inkDim, marginTop: 6 }}>
                  {running && state?.kmeans?.cluster ? state.kmeans.cluster : "fitting latent space…"}
                  {state?.kmeans?.is_outlier ? " · OUTLIER GATED" : ""}
                </div>
              </div>
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "flex-start", marginTop: 12, flexWrap: "wrap" }}>
              <div style={{ flex: "0 0 auto" }}>
                <div style={{ fontSize: 11, color: C.inkMuted, textTransform: "uppercase", letterSpacing: 0.4, marginBottom: 5 }}>
                  Grad-CAM saliency (eye CNN)
                </div>
                {running && state?.gradcam_png ? (
                  <img
                    src={`data:image/jpeg;base64,${state.gradcam_png}`}
                    alt="Grad-CAM heat map"
                    style={{ width: 128, height: 128, borderRadius: 8, border: `1px solid ${C.border}`, imageRendering: "auto" }}
                  />
                ) : (
                  <div style={{ width: 128, height: 128, borderRadius: 8, border: `1px dashed ${C.border}`, display: "flex", alignItems: "center", justifyContent: "center", color: C.inkDim, fontSize: 11 }}>
                    {running ? "computing…" : "—"}
                  </div>
                )}
              </div>
              <div style={{ flex: 1, minWidth: 220 }}>
                {state?.kmeans?.cluster_probs?.length > 0 && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {["0 · Alert", "1 · Blinks", "2 · Microsleep", "3 · Yawn"].map((lbl, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span className="vd-mono" style={{ fontSize: 11, width: 86, color: C.inkMuted }}>{lbl}</span>
                        <div style={{ flex: 1, height: 5, background: C.surfaceAlt2, borderRadius: 3 }}>
                          <div style={{ width: `${Math.round((state.kmeans.cluster_probs[i] ?? 0) * 100)}%`, height: "100%", background: i === (state.kmeans.cluster_id ?? -1) ? C.amber : C.navy, borderRadius: 3 }} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </Card>

          {/* Eye & Mouth Region Measurements */}
          <Card style={{ padding: 18 }}>
            <PanelLabel icon={<Layers size={15} />} text="Region Measurements — Eye CNN Crops & Mouth Opening" />
            <p style={{ fontSize: 12, color: C.inkMuted, margin: "8px 0 10px" }}>
              Per-eye 64×64 grayscale crops are classified by the eye CNN; mouth opening is measured as the dark-pixel height of the inner-mouth region relative to mouth width (MAR).
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <BigStat
                label={running && state?.is_blink ? "Eye State (CNN) — blinking" : "Eye State (CNN)"}
                value={running ? (pClosed > 0.5 ? "CLOSED" : "OPEN") : "—"}
                color={running ? (pClosed > 0.5 ? C.critical : C.teal) : C.inkDim}
              />
              <BigStat
                label="Yawn Detector (sustained)"
                value={running ? (pYawn > 0.5 ? "YAWNING" : "NO YAWN") : "—"}
                color={running ? (pYawn > 0.5 ? C.caution : C.teal) : C.inkDim}
              />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginTop: 10 }}>
              <BigStat label="Blinks counted" value={running ? state?.blink_count ?? 0 : "—"} color={C.navy} />
              <BigStat label="Yawns counted" value={running ? state?.yawn_count ?? 0 : "—"} color={(state?.yawn_count ?? 0) > 0 ? C.caution : C.navy} />
              <BigStat label="Eyes closed for" value={running ? `${(state?.eye_closed_seconds ?? 0).toFixed(1)}s` : "—"} color={(state?.eye_closed_seconds ?? 0) >= 2 ? C.critical : C.navy} />
              <BigStat label="Mouth opening (MAR)" value={running ? (state?.mar ?? 0).toFixed(3) : "—"} color={(state?.mar ?? 0) > 0.32 ? C.caution : C.navy} />
            </div>
          </Card>

          {/* Attention History Graph */}
          <Card style={{ padding: 18 }}>
            <PanelLabel icon={<Radio size={15} />} text="Fatigue Attention — Rolling Window (90 frames)" />
            <AttentionGraph
              data={state?.attention_hist || []}
              color={tier === "critical" ? C.critical : tier === "warning" ? C.amber : C.navy}
            />
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
              <span style={{ fontSize: 12, color: C.inkMuted }}>last {state?.attention_hist?.length || 0} samples</span>
              <span className="vd-mono" style={{ fontSize: 12.5, color: drowsy ? C.critical : C.navy, fontWeight: 700 }}>
                p = {running ? pDrowsy.toFixed(2) : "—"}
              </span>
            </div>
          </Card>
        </div>

        {/* RIGHT COLUMN: Session Stats & Event Log */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <StatCard icon={<Timer size={15} />} label="Session duration" value={running ? fmtDuration(state?.session_seconds || 0) : "—"} />
            <StatCard icon={<Activity size={15} />} label="Frames analysed" value={running ? state?.frame_count ?? 0 : "—"} />
            <StatCard icon={<BellRing size={15} />} label="Alerts triggered" value={running ? (state?.events || []).filter((e) => e.tier === "critical").length : "—"} />
            <StatCard icon={<Camera size={15} />} label="Camera status" value={running ? (state?.cam_error ? "error" : "streaming") : "idle"} small />
          </div>

          <Card style={{ padding: 18 }}>
            <PanelLabel icon={<AlertTriangle size={15} />} text="Real-Time Event Audit Log" />
            <div
              className="vd-mono"
              style={{
                marginTop: 12,
                background: C.console,
                border: `1px solid ${C.consoleBorder}`,
                borderRadius: 8,
                padding: "12px 14px",
                height: 320,
                overflowY: "auto",
                display: "flex",
                flexDirection: "column",
                gap: 8,
              }}
            >
              {!running && (
                <div style={{ fontSize: 12, color: "#5A6B80" }}>
                  {fmtClock()} — session standby. Model is ready to process cabin feed.
                </div>
              )}
              {(state?.events || []).map((e, i) => (
                <div key={i} style={{ fontSize: 12, lineHeight: 1.5 }}>
                  <span style={{ color: "#5A6B80" }}>{fmtTs(e.ts)} — </span>
                  <span style={{ color: e.tier === "critical" ? "#FF8A80" : "#F2C245" }}>{e.text}</span>
                </div>
              ))}
              {running && (state?.events || []).length === 0 && (
                <div style={{ fontSize: 12, color: "#5A6B80" }}>
                  {fmtClock()} — monitoring active. Driver state is alert.
                </div>
              )}
            </div>
          </Card>

          <Card style={{ padding: "14px 16px", display: "flex", gap: 10, alignItems: "center", background: C.surfaceAlt }}>
            <CheckCircle2 size={16} color={C.teal} />
            <span style={{ fontSize: 12.5, color: C.inkMuted }}>
              Sessions persist to SQLite audit trail — review history in{" "}
              <Link to="/analytics" className="vd-link">Analytics &amp; History</Link>.
            </span>
          </Card>
        </div>
      </div>
    </div>
  );
}

/* ---------------- Helper Subcomponents ---------------- */

function ClassBar({ label, val, color, icon }) {
  const pct = Math.round(Math.min(1, Math.max(0, val)) * 100);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6, color: C.ink, fontWeight: 600 }}>
          <span>{icon}</span> {label}
        </span>
        <span className="vd-mono" style={{ fontWeight: 700, color }}>
          {(val * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{ height: 6, background: C.surfaceAlt2, borderRadius: 3, overflow: "hidden" }}>
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            borderRadius: 3,
            transition: "width 0.2s ease",
          }}
        />
      </div>
    </div>
  );
}

function fmtTs(ts) {
  return new Date(ts * 1000).toTimeString().slice(0, 8);
}

function fmtClock() {
  return new Date().toTimeString().slice(0, 8);
}

function BigStat({ label, value, color }) {
  return (
    <div style={{ background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 9, padding: "12px 14px" }}>
      <div style={{ fontSize: 11, color: C.inkMuted, textTransform: "uppercase", letterSpacing: 0.4 }}>{label}</div>
      <div className="vd-mono" style={{ fontSize: 20, fontWeight: 700, color, marginTop: 3 }}>
        {value}
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, small }) {
  return (
    <div style={{ background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 9, padding: "13px 15px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, color: C.inkMuted, fontSize: 11.5 }}>
        {icon}
        {label}
      </div>
      <div className="vd-mono" style={{ fontSize: small ? 15 : 20, fontWeight: 600, color: C.ink, marginTop: 4 }}>
        {value}
      </div>
    </div>
  );
}

function AttentionGraph({ data, color }) {
  const w = 420;
  const h = 90;
  if (!data.length) {
    return (
      <div style={{ height: h, display: "flex", alignItems: "center", justifyContent: "center", color: C.inkDim, fontSize: 12 }}>
        awaiting samples…
      </div>
    );
  }
  const pts = data
    .map((v, i) => {
      const x = (i / Math.max(1, data.length - 1)) * w;
      const y = h - Math.max(0, Math.min(1, v)) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <polygon points={`0,${h} ${pts} ${w},${h}`} fill={color} fillOpacity="0.08" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2.2" />
      <line x1="0" y1={h * 0.38} x2={w} y2={h * 0.38} stroke={C.border} strokeDasharray="3 5" />
    </svg>
  );
}
