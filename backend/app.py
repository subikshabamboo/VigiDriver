"""VigiDriver FastAPI application.

Endpoints
---------
GET  /api/health              — service + model status
POST /api/monitor/start       — start monitoring (camera or simulation)
POST /api/monitor/stop        — end session
GET  /api/monitor/state       — current analysis snapshot
GET  /api/monitor/cameras     — enumerate cameras
WS   /api/ws/monitor          — WebSocket state stream
POST /api/analyze             — one-shot image analysis
GET  /api/cam                 — MJPEG annotated stream
POST /api/contact             — store contact enquiry
GET  /api/contact             — admin contact listing
GET  /api/sessions            — session history
GET  /api/sessions/{id}       — session detail with alerts
GET  /api/stats               — aggregate statistics
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

ROOT_DIR = os.path.dirname(BACKEND_DIR)
DIST_DIR = os.path.join(ROOT_DIR, "dist")
INDEX_HTML = os.path.join(DIST_DIR, "index.html")
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

import db
import kmeans as kmeans_mod
import noise as noise_mod
import temporal as temporal_mod
import xai as xai_mod
from models import get_service
from vision import (
    CascadeState,
    EarClosure,
    EyeBehaviorTracker,
    FaceTracker,
    MAR_YAWN_LANDMARK,
    YawnTracker,
    detect_face,
    detect_yawn,
    enhance_low_light,
    extract_eye_mouth_regions,
    shutdown_landmarker,
)

logger = logging.getLogger("vigidriver")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- state

@dataclass
class MonitorState:
    """Shared live-monitor state."""

    session_id: int | None = None
    running: bool = False
    driver_label: str = "Driver_01"
    threshold: float = 0.50
    camera_index: int = 0
    use_camera: bool = True
    use_browser_cam: bool = False

    frame_count: int = 0
    fps: float = 0.0
    p_drowsy: float = 0.0
    p_closed: float = 0.0
    p_open: float = 0.0
    p_no_yawn: float = 1.0
    p_yawn: float = 0.0
    top_class: str = "no_yawn"
    tier: str = "normal"
    face_box: tuple[int, int, int, int] | None = None
    mouth_box: tuple[int, int, int, int] | None = None
    mar: float | None = None
    ear: float | None = None
    landmarks: bool = False
    ear_state: str | None = None  # "calibrating" | "closed" | "open" | None
    blink_count: int = 0
    yawn_count: int = 0
    eye_closed_seconds: float = 0.0
    perclos: float = 0.0
    is_blink: bool = False
    session_seconds: float = 0.0

    # --- Model-1 temporal (Bi-LSTM) + K-Means + XAI surfaces ---
    lstm_p: float | None = None
    lstm_alpha: float | None = None      # self-gating weight applied to the LSTM
    lstm_reliable: bool = False          # alpha cleared the confidence threshold
    lstm_diag: dict[str, Any] = field(default_factory=dict)
    temporal_window: int = 0
    temporal_ready: bool = False
    cluster_id: int | None = None
    cluster_label: str | None = None
    cluster_probs: list[float] = field(default_factory=list)
    is_outlier: bool = False
    gradcam_b64: str = ""
    gradcam_meta: dict[str, Any] = field(default_factory=dict)
    attention_hist: list[float] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    annotated_jpeg: bytes | None = None
    cam_error: str | None = None
    face_detected: bool = False
    brightness: float = 0.0
    started_at: float = 0.0
    last_viewer_ts: float = 0.0
    last_eye_crop: np.ndarray | None = None  # for debugging


state = MonitorState()
state_lock = asyncio.Lock()
cascade = CascadeState()
eye_tracker = EyeBehaviorTracker()
yawn_tracker = YawnTracker()
ear_closure = EarClosure()
feature_extractor = temporal_mod.VigilanceFeatureExtractor()
temporal_buffer = temporal_mod.TemporalBuffer()
kmeans_gate = kmeans_mod.KMeansGate()
tracker = FaceTracker(detect_every=3)
capture: cv2.VideoCapture | None = None
simulated_frame_count: int = 0
_last_eye_ts: float | None = None   # wall-clock of the last eye-tracker
                                    # update; drives the 2 s micro-sleep rule

VIEWER_TIMEOUT = 30.0

# --------------------------------------------------------------------------- lifespan

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    db.init_db()
    stale = await asyncio.to_thread(db.close_stale_sessions)
    if stale:
        logger.info("Finalised %d stale session(s)", stale)
    svc = get_service()
    logger.info("VigiDriver ready — model: %s", svc._model_path)
    # Start the monitor loop as a persistent background task
    monitor_task = asyncio.create_task(monitor_loop())
    yield
    state.running = False
    await asyncio.sleep(0.3)
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    stop_camera()
    stop_simulation()
    shutdown_landmarker()


app = FastAPI(title="VigiDriver API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- camera

def start_camera(index: int = 0) -> None:
    global capture
    if capture is not None and capture.isOpened():
        return
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Keep the driver buffer tiny so we always analyse the newest frame instead
    # of replaying a backlog (a classic cause of "stuck" webcam feeds).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=503, detail=f"Camera {index} unavailable")
    # Warm the camera up: many webcams emit dark/empty frames for the first
    # second while auto-exposure settles. Discard them now so the first
    # analyzed frame is a real image.
    for _ in range(12):
        ok, _ = cap.read()
        if not ok:
            break
        time.sleep(0.03)
    capture = cap

def stop_camera() -> None:
    global capture
    if capture is not None:
        capture.release()
        capture = None

def read_frame() -> np.ndarray | None:
    if capture is not None and capture.isOpened():
        ok, frame = capture.read()
        if ok:
            return frame
    return None

# --------------------------------------------------------------------------- simulation

def _eyelid_openness(frame_count: int) -> float:
    """Simulate eyelid openness. 1.0 = wide open, 0.0 = fully closed.

    Frames 0-20: alert driver with occasional blinks.
    Frames 20-60: progressive fatigue, eyes gradually close.
    Frames 60+: eyes mostly closed (drowsy).
    Occasional yawns around frames 40-50.
    """
    t = frame_count

    # Full blinks in the alert phase: openness dips to 0 for ~3 frames.
    # A real blink fully occludes the eye, so the CNN must see a full closure.
    blink = 0.0
    if t < 20:
        blink = (
            1.00 * np.exp(-((t - 4.0) ** 2) / 1.6)
            + 1.00 * np.exp(-((t - 10.0) ** 2) / 1.6)
            + 0.95 * np.exp(-((t - 16.0) ** 2) / 1.6)
        )

    # Fatigue progression
    fatigue_t = max(0.0, (t - 20) / 45.0)  # ramps over 45 frames after frame 20
    fatigue_close = fatigue_t ** 1.3  # nonlinear closing
    fatigue_close = min(fatigue_close, 0.94)

    openness = float(np.clip(1.0 - blink - fatigue_close, 0.03, 1.0))
    return openness


def _mouth_openness(frame_count: int) -> float:
    """Simulate mouth openness for MAR. 0.0 = closed, 1.0 = wide open (yawn).

    Normal talking: occasional small openings.
    Yawn: large vertical opening around frames 40-50.
    """
    t = frame_count

    # Normal small mouth movements
    normal = 0.06 * np.sin(t * 0.3) + 0.04 * np.sin(t * 0.7 + 1.0)

    # Yawn #1: sustained wide opening for ~1.5 s around frame 45
    yawn_center = 45
    yawn_width = 11
    yawn = 0.0
    if abs(t - yawn_center) < yawn_width * 2:
        yawn = 0.95 * np.exp(-((t - yawn_center) ** 2) / (yawn_width ** 2))
        # Second yawn (~1.2 s) near frame 68
        yawn += 0.95 * np.exp(-((t - (yawn_center + 23)) ** 2) / (9 ** 2))

    openness = float(np.clip(normal + yawn, 0.0, 1.0))
    return openness


def _simulate_frame(frame_count: int) -> np.ndarray:
    """Render a synthetic cabin-cam frame with simulated driver state."""
    H, W = 480, 640
    frame = np.full((H, W, 3), 28, dtype=np.uint8)

    # Warm glow from dashboard
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - W * 0.78) ** 2 + (yy - 50) ** 2)) / 90.0
    glow = np.clip(1.0 - d, 0.0, 1.0) ** 2 * 50
    frame[:, :, 0] = np.clip(frame[:, :, 0] + glow, 0, 255).astype(np.uint8)

    # Face region
    fx, fy, fw, fh = 180, 80, 250, 310
    x0, y0 = max(0, fx), max(0, fy)
    x1b, y1b = min(W, fx + fw), min(H, fy + fh)
    wb, hb = x1b - x0, y1b - y0

    face = frame[y0:y1b, x0:x1b].copy()
    skin = np.array([158, 120, 96], dtype=np.float32)
    for c in range(3):
        face[:, :, c] = np.clip(skin[c] + np.random.normal(0, 5, face.shape[:2]), 0, 255).astype(np.uint8)

    # Hair/head outline
    cv2.ellipse(face, (wb // 2, -8), (wb // 2, hb // 2 + 10), 0, 0, 360, (28, 20, 16), -1)
    cv2.ellipse(face, (wb // 2, 2), (wb // 2 + 10, hb // 10), 0, 0, 360, (35, 28, 24), -1)

    # Eye state from simulation
    eye_open = _eyelid_openness(frame_count)
    mouth_open = _mouth_openness(frame_count)

    # --- Draw eyes ---
    eye_y = int(hb * 0.37)
    eye_w = int(wb * 0.16)
    eye_h = int(hb * 0.08)

    for ex_frac in (0.30, 0.70):
        ex = int(wb * ex_frac)

        # Eye white (sclera) — visible height proportional to openness
        sclera_h = max(2, int(eye_h * eye_open))
        sclera_y0 = eye_y - eye_h // 2 + (eye_h - sclera_h) // 2
        sclera_y1 = sclera_y0 + sclera_h

        if sclera_h > 1:
            cv2.ellipse(face, (ex, eye_y), (int(eye_w * 0.6), int(sclera_h * 0.6)),
                       0, 0, 360, (210, 205, 195), -1)

        # Iris/pupil — visible when eye is open
        if eye_open > 0.08:
            iris_r = max(2, int(eye_w * 0.28 * eye_open))
            # Iris center shifts down as eyelid closes from top
            iris_cy = eye_y + int((eye_h - sclera_h) * 0.3)
            if iris_cy - iris_r >= sclera_y0 and iris_cy + iris_r <= sclera_y1:
                cv2.circle(face, (ex, iris_cy), iris_r, (60, 70, 90), -1)
                cv2.circle(face, (ex, iris_cy), max(1, iris_r // 2), (25, 30, 50), -1)

        # Upper eyelid
        upper_cover = int((1.0 - eye_open) * eye_h * 0.85)
        if upper_cover > 0:
            cv2.ellipse(face, (ex, eye_y - eye_h // 2 + upper_cover // 2),
                       (int(eye_w * 0.7), int(upper_cover * 0.6)),
                       0, 0, 360, (150, 110, 85), -1)

        # Lower eyelid (minor)
        lower_cover = int((1.0 - eye_open) * eye_h * 0.3)
        if lower_cover > 0:
            cv2.ellipse(face, (ex, eye_y + eye_h // 2 - lower_cover // 2),
                       (int(eye_w * 0.6), int(lower_cover * 0.5)),
                       0, 0, 360, (150, 110, 85), -1)

        # Eyelashes when open
        if eye_open > 0.4:
            for lx in range(ex - int(eye_w * 0.35), ex + int(eye_w * 0.35), 3):
                if sclera_y0 > 0:
                    cv2.line(face, (lx, sclera_y0), (lx + 1, sclera_y0 - 3), (60, 50, 40), 1)

    # --- Draw mouth ---
    mouth_y = int(hb * 0.74)
    mouth_w = int(wb * 0.14)
    mouth_h = int(hb * 0.04 + mouth_open * hb * 0.06)

    # Lip line
    cv2.ellipse(face, (int(wb * 0.50), mouth_y), (mouth_w, max(2, mouth_h)),
               0, 0, 360, (130, 90, 75), 2)

    # Open mouth when yawning
    if mouth_open > 0.15:
        inner_h = max(2, int(mouth_h * 0.6 * mouth_open))
        cv2.ellipse(face, (int(wb * 0.50), mouth_y), (int(mouth_w * 0.7), inner_h),
                   0, 0, 360, (50, 35, 28), -1)

    frame[y0:y1b, x0:x1b] = face

    # Time stamp
    stamp = f"{int(frame_count / 15.0) % 60:02d}:{int(frame_count / (15.0*60)) % 60:02d}"
    cv2.putText(frame, stamp, (W - 70, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, stamp, (W - 71, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return frame


class SimulatedCapture:
    """VideoCapture-like that yields synthetic frames."""

    def __init__(self) -> None:
        self._frame_count = 0
        self._open = True

    def isOpened(self) -> bool:
        return self._open

    def read(self) -> tuple[bool, np.ndarray]:
        if not self._open:
            return (False, np.empty((0, 0, 3), dtype=np.uint8))
        frame = _simulate_frame(self._frame_count)
        self._frame_count += 1
        return (True, frame)

    def release(self) -> None:
        self._open = False


def start_simulation() -> None:
    global capture, simulated_frame_count
    if capture is not None and capture.isOpened():
        return
    capture = SimulatedCapture()
    simulated_frame_count = 0

def is_simulation() -> bool:
    return isinstance(capture, SimulatedCapture)

def stop_simulation() -> None:
    global capture
    if isinstance(capture, SimulatedCapture):
        capture.release()
        capture = None

# --------------------------------------------------------------------------- analysis

def analyze_frame(frame_bgr: np.ndarray, temporal: bool = True) -> dict[str, Any]:
    """Run face detection, eye CNN, MAR yawn detection, and cascade on one frame.

    temporal=True feeds the per-frame probabilities into the behaviour trackers
    (blinks, sustained closure, yawn events). One-shot photo analysis passes
    temporal=False so it doesn't pollute the live session's counters.
    """
    global cascade, tracker, simulated_frame_count, _last_eye_ts

    svc = get_service()

    # Low-light recovery: night-time cabin cameras deliver a very dim image
    # that blanks out both Haar and MediaPipe. Lift the frame toward a usable
    # brightness (no-op when already lit) before any detection runs.
    raw_brightness = float(frame_bgr.mean())
    frame_bgr = enhance_low_light(frame_bgr)
    if not temporal:
        raw_brightness = raw_brightness  # one-shot: do not touch live state below

    # Face tracking — for simulated feeds, use the known face box.
    # One-shot (temporal=False) uses throwaway trackers so a dashboard photo
    # can never pollute the live session's counters or the MJPEG overlay
    # (the one-shot path previously incremented frame_count per call).
    one_shot = not temporal
    if is_simulation():
        box = (180, 80, 250, 310)  # Known synthetic face box
        if not one_shot:
            state._sim_box = box
    elif one_shot:
        local_tracker = FaceTracker(detect_every=3, lost_after_misses=tracker.lost_after_misses)
        box = local_tracker.update(frame_bgr)
    else:
        box = tracker.update(frame_bgr)

    if not one_shot:
        state.face_box = box
        state.face_detected = box is not None
        state.brightness = round(raw_brightness, 1)

    if box is None:
        return _empty_result(svc, temporal)

    # Extract eye and mouth regions
    regions = extract_eye_mouth_regions(frame_bgr, box)

    # Run the eye CNN on BOTH eyes in one batched forward pass — roughly half
    # the per-frame inference cost of two separate predict_eye() calls.
    left_result, right_result = svc.predict_eyes(
        [regions["left_eye_crop"], regions["right_eye_crop"]]
    )

    # Average eye probabilities
    if left_result and right_result:
        p_open = (left_result["p_open"] + right_result["p_open"]) / 2.0
        p_closed = (left_result["p_closed"] + right_result["p_closed"]) / 2.0
    elif left_result:
        p_open = left_result["p_open"]
        p_closed = left_result["p_closed"]
    elif right_result:
        p_open = right_result["p_open"]
        p_closed = right_result["p_closed"]
    else:
        p_open = 0.5
        p_closed = 0.5

    # --- Adaptive EAR closure (landmark mode) ---
    # MediaPipe landmarks give a reliable EAR regardless of camera geometry;
    # the user's eye CNN stays authoritative, EAR acts as a geometry-independent
    # cross-check so non-ideal crops can't fool the CNN.
    ear_state: bool | str | None = None
    if regions.get("landmarks") and regions.get("ear") is not None:
        state.mar = regions.get("mar")
        ear_state = ear_closure.update(regions["ear"])
        state.ear = regions["ear"]
        state.landmarks = True
        state.ear_state = (
            "calibrating" if ear_state is None
            else "closed" if ear_state else "open"
        )

    # Combine CNN + EAR into one closure decision.
    #
    # Either signal may raise closure (OR), because each one fails in a
    # different situation and neither can veto the other:
    #   * the eye CNN is only as good as its crop — on loose/unusual crops it
    #     has been seen to hit p_closed=1.00 while the eye was plainly open;
    #   * EAR is lighting-proof but geometrically weak here: MediaPipe's
    #     eyelid landmarks only collapse partially (open ≈0.31 → closed
    #     ≈0.23) and the previously crossed landmark pairs measured mostly
    #     the horizontal offset, so on live frames a genuinely shut eye read
    #     0.42–0.54 and never crossed the threshold.
    # Measured on pixel-labelled live frames the CNN read p_open=0.000 on
    # every truly closed eye, while EAR stayed at 0.42–0.54 (threshold 0.38)
    # — the old "EAR is authoritative" rule vetoed 100% of real closures,
    # which is why a driver asleep for seconds still showed as OPEN.
    # A real CNN prediction sitting exactly on the 0.50 boundary counts as
    # closed, matching models.py ("p_open <= 0.50 => eye closed"). Two
    # pixel-labelled shut frames in the captured set landed on exactly
    # 0.500 and were previously discarded by the strict comparison. The
    # "no eye crop at all" placeholder (p_open = 0.5) must NOT count, so the
    # decision is gated on a prediction actually existing.
    cnn_closed = (left_result is not None or right_result is not None) \
        and p_closed >= 0.5
    ear_closed = ear_state is True
    combined_closed = cnn_closed or ear_closed
    p_closed_eff = max(p_closed, 0.90) if combined_closed else min(p_closed, 0.10)

    # Effective open probability for the temporal blink/sustained tracker.
    # When the eye is open we pass the CNN's own value through instead of
    # forcing it up to 0.90 whenever EAR agreed — that clamp zeroed PERCLOS
    # and hid long-run drowsiness even when the CNN was reporting squinting.
    p_open_eff = 0.05 if combined_closed else p_open

    # Yawn detection from MAR — smoothed into sustained yawn events
    mar = regions.get("mar")
    yawn_threshold = MAR_YAWN_LANDMARK if regions.get("landmarks") else None
    if temporal:
        yawn_res = yawn_tracker.update(mar, threshold=yawn_threshold)
        p_yawn = yawn_res["p_yawn"]
        p_no_yawn = yawn_res["p_no_yawn"]
        is_yawning = yawn_res["is_yawning"]
        yawn_count = yawn_res["yawn_count"]
        yawn_event = yawn_res["yawn_event"]
    else:
        mar_result = detect_yawn(mar)
        if regions.get("landmarks") and mar is not None and mar > MAR_YAWN_LANDMARK:
            mar_result.update(p_yawn=0.9, p_no_yawn=0.1, is_yawning=True)
        p_yawn = mar_result["p_yawn"]
        p_no_yawn = mar_result["p_no_yawn"]
        is_yawning = mar_result["is_yawning"]
        yawn_count = 0
        yawn_event = False

    # Behaviour tracking: blinks vs sustained (micro-sleep) closure, PERCLOS
    if temporal:
        # The live loop's rate is machine-dependent (~3–15 fps), so the
        # tracker is fed real elapsed time; otherwise the 2 s micro-sleep
        # rule counts frames and stretches to several real seconds.
        now_t = time.time()
        dt = None
        if _last_eye_ts is not None:
            dt = min(max(now_t - _last_eye_ts, 0.0), 1.0)
        _last_eye_ts = now_t
        beh = eye_tracker.update(p_open_eff, dt=dt)
    else:
        beh = {
            "is_closed": combined_closed, "is_blink": False,
            "blink_protect": False,
            "sustained_closure": False, "sustained_start": False,
            "closed_seconds": 0.0, "blink_count": eye_tracker.blink_count,
            "perclos": 0.0,
        }

    # --- Model-1 temporal vigilance stream (Bi-LSTM micro-sleep track) ---
    # Per-frame 4-d vigilance vector -> 30-frame standardized window ->
    # best_vigidriver_lstm.keras sigmoid P(drowsy). Inference runs at the
    # window stride (every 5 frames ≈ 3 Hz) to keep CPU headroom.
    feat_vec: np.ndarray | None = None
    p_lstm: float | None = None
    cluster_res: dict[str, Any] = {}
    if temporal:
        le = regions.get("left_eye_center")
        re = regions.get("right_eye_center")
        eye_mid = (
            ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0) if (le and re) else None
        )
        feat_vec = feature_extractor.compute_features({
            "ear": regions.get("ear"),
            "p_open_eff": p_open_eff,
            "mar": regions.get("mar"),
            "face_box": box,
            "eye_mid": eye_mid,
            "eye_mid_y": eye_mid[1] if eye_mid else None,
        })
        temporal_buffer.push(feat_vec)
        lstm_svc = temporal_mod.get_lstm_service()
        state.temporal_window = len(temporal_buffer)
        state.temporal_ready = bool(
            temporal_buffer.ready and lstm_svc.available and feature_extractor.calibrated
        )
        if state.temporal_ready and state.frame_count % temporal_mod.STRIDE == 0:
            try:
                # Self-gated inference: the LSTM verdict is measured for
                # reliability (calibration x manifold support x agreement with
                # the frame's eye signal) in the same pass that produces it.
                rel = temporal_mod.gate_for_fusion(
                    svc=lstm_svc,
                    feature_extractor=feature_extractor,
                    temporal_buffer=temporal_buffer,
                    p_cnn=p_closed_eff,
                )
                p_lstm = rel.raw_p_drowsy
                state.lstm_p = p_lstm
                state.lstm_alpha = rel.alpha_lstm
                state.lstm_reliable = rel.separable
                state.lstm_diag = {
                    "alpha": round(rel.alpha_lstm, 4),
                    "calibration": rel.calibration,
                    "support": round(rel.support, 4),
                    "agreement": round(rel.agreement, 4),
                    "manifold_margin": round(rel.manifold_margin, 4),
                    "gradient_norm": round(rel.gradient_norm, 6),
                    "gated_p_drowsy": round(rel.gated_p_drowsy, 4),
                    "raw_p_drowsy": round(rel.raw_p_drowsy, 4),
                }
            except Exception as exc:  # defensive: never crash the live loop
                logger.warning("LSTM window inference failed: %s", exc)

        # --- K-Means latent clustering engine (k=4) + outlier gate ---
        if feat_vec is not None:
            latent = kmeans_mod.KMeansGate.build_latent(feat_vec, beh, p_lstm)
            gate_res = kmeans_gate.observe(latent)
            if gate_res:
                cluster_res = gate_res
                prev_outlier = state.is_outlier
                state.cluster_id = gate_res["cluster_id"]
                state.cluster_label = gate_res["cluster"]
                state.cluster_probs = gate_res["cluster_probs"]
                state.is_outlier = gate_res["is_outlier"]

    # Combined drowsiness score: eye closure + yawning + PERCLOS trend.
    # Blinks are excluded from the drowsiness signal — a closure still inside
    # the blink-duration envelope (<= ~0.9 s, not yet sustained) contributes
    # nothing; PERCLOS handles long-run drowsiness and the 2 s sustained rule
    # escalates true micro-sleep. Yawn evidence still passes through so a
    # blink + yawn combo is not silently dropped.
    if beh.get("blink_protect") and not beh["sustained_closure"]:
        eye_term = 0.0
        perclos_term = 0.0
    else:
        eye_term = p_closed_eff * 0.8
        perclos_term = beh["perclos"] * 0.2
    p_drowsy = float(np.clip(eye_term + perclos_term + p_yawn * 0.7, 0.0, 1.0))
    # Sustained multi-second closure IS micro-sleep — force the score up
    if beh["sustained_closure"]:
        p_drowsy = max(p_drowsy, 0.95)

    # Multi-model fusion: geometric/physiological score is modulated by the
    # Bi-LSTM temporal micro-sleep model (70/30). The LSTM captures closure
    # velocity over the 2 s window, reducing single-frame flicker. Its last
    # windowed probability is cached and applied to EVERY live frame, not
    # just the stride frames that trigger a fresh tensor inference.
    if temporal and state.temporal_ready:
        p_lstm_eff = p_lstm if p_lstm is not None else state.lstm_p
        if p_lstm_eff is not None:
            # Self-gating: pull the LSTM toward the geometric score in
            # proportion to its measured reliability. alpha = 0 exactly
            # neutralises the temporal stream; alpha = 1 restores the full
            # 70/30 fusion. This is what stops an over-confident temporal
            # verdict on an uncalibrated / out-of-support window from
            # injecting a false fatigue alert.
            alpha = state.lstm_alpha if state.lstm_alpha is not None else 1.0
            p_lstm_gated = alpha * p_lstm_eff + (1.0 - alpha) * p_drowsy
            p_drowsy = float(np.clip(0.70 * p_drowsy + 0.30 * p_lstm_gated, 0.0, 1.0))

    # Top class
    classes_prob = {
        "Closed": p_closed,
        "Open": p_open,
        "no_yawn": p_no_yawn,
        "yawn": p_yawn,
    }
    top_class = max(classes_prob, key=classes_prob.get)

    # Cascade update — one-shot uses a throwaway cascade so bursts of
    # dashboard photo analyses can never bump the live tier ladder.
    if temporal:
        tier = cascade.update(p_drowsy)
    else:
        tier = CascadeState().update(p_drowsy)
    if beh["sustained_closure"]:
        tier = "critical"  # multi-second closure always escalates

    # Store state (live path only — one-shot returns the score without
    # touching the streaming dashboard)
    if temporal:
        state.p_drowsy = p_drowsy
        state.p_closed = p_closed_eff
        state.p_open = 1.0 - p_closed_eff
        state.p_no_yawn = p_no_yawn
        state.p_yawn = p_yawn
        state.top_class = top_class
        state.tier = tier
        state.mar = regions.get("mar")
        state.mouth_box = regions.get("mouth_box")
        state.blink_count = beh["blink_count"]
        state.yawn_count = yawn_count
        state.eye_closed_seconds = beh["closed_seconds"]
        state.perclos = beh["perclos"]
        state.is_blink = beh["is_blink"]
        if p_lstm is None and not state.temporal_ready:
            state.lstm_p = None

        # Update frame count
        state.frame_count += 1

        if state.started_at:
            state.session_seconds = time.time() - state.started_at

        # Attention history
        state.attention_hist.append(p_drowsy)
        if len(state.attention_hist) > 90:
            state.attention_hist = state.attention_hist[-90:]

        # XAI: Grad-CAM saliency refresh. The attribution pass costs ~100 ms,
        # so it is throttled to keep the live loop responsive.
        if state.frame_count % 20 == 0:
            g_crop = regions.get("left_eye_crop")
            if g_crop is None:
                g_crop = regions.get("right_eye_crop")
            if g_crop is not None:
                state.gradcam_b64, state.gradcam_meta = xai_mod.eye_gradcam_png(svc, g_crop)

    # Events
    if temporal and yawn_event and box:
        state.events.append({
            "ts": time.time(), "tier": "warning", "p_drowsy": p_drowsy,
            "text": f"🥱 Yawn #{yawn_count} detected — sustained mouth opening (MAR={mar:.2f})",
        })
    if temporal and beh["sustained_start"] and box:
        state.events.append({
            "ts": time.time(), "tier": "critical", "p_drowsy": p_drowsy,
            "text": f"⚠ SUSTAINED EYE CLOSURE {beh['closed_seconds']:.1f}s — micro-sleep risk!",
        })
    if (temporal and state.is_outlier and not getattr(state, "_last_outlier", False)
            and cluster_res):
        state.events.append({
            "ts": time.time(), "tier": "normal", "p_drowsy": p_drowsy,
            "text": ("◯ K-Means outlier gate: latent vector outside cluster "
                     f"boundaries (d={cluster_res['distance_to_centroid']}) — "
                     "alert escalation held pending corroboration"),
        })
    state._last_outlier = state.is_outlier
    if temporal and tier in ("warning", "critical") and tier != getattr(cascade, "_last_logged_tier", None):
        cascade._last_logged_tier = tier
        state.events.append({
            "ts": time.time(),
            "tier": tier,
            "p_drowsy": p_drowsy,
            "text": (
                f"{'CRITICAL' if tier == 'critical' else 'Warning'}: "
                f"drowsiness {p_drowsy:.2f} "
                f"(Eye: {'closed' if p_closed > 0.5 else 'open'} "
                f"P(closed)={p_closed:.2f}, "
                f"Yawn: {'yes' if is_yawning else 'no'} "
                f"P(yawn)={p_yawn:.2f}) "
                f"above threshold {state.threshold:.2f}"
            ),
        })
    elif tier == "normal":
        cascade._last_logged_tier = None
    state.events = state.events[-40:]

    # Database persistence
    if state.session_id is not None and state.frame_count % 4 == 0 and box:
        db.add_attention(state.session_id, p_drowsy, beh["perclos"], 0.0, tier)

    if temporal and state.session_id is not None and box:
        if yawn_event:
            db.add_alert(state.session_id, "warning", p_drowsy,
                        note=f"yawn #{yawn_count} MAR={mar:.2f}")
        if beh["sustained_start"]:
            db.add_alert(state.session_id, "critical", p_drowsy,
                        note=f"sustained eye closure {beh['closed_seconds']:.1f}s")

    if state.session_id is not None and tier in ("warning", "critical") and state.frame_count % 4 == 0 and box:
        db.add_alert(state.session_id, tier, p_drowsy,
                    note=f"cascade frame {state.frame_count} ({top_class})")

    # Build annotated frame for streaming (live path only)
    if temporal:
        annotated = _draw_overlay(frame_bgr, regions)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
        state.annotated_jpeg = buf.tobytes() if ok else None

    return {
        "p_drowsy": round(p_drowsy, 4),
        "p_closed": round(p_closed, 4),
        "p_open": round(p_open, 4),
        "p_no_yawn": round(p_no_yawn, 4),
        "p_yawn": round(p_yawn, 4),
        "top_class": top_class,
        "tier": tier,
        "face_box": list(box),
        "mar": round(regions.get("mar") or 0, 4),
        "is_eye_closed": combined_closed,
        "is_yawning": is_yawning,
        "is_blink": beh["is_blink"],
        "blink_count": beh["blink_count"],
        "yawn_count": yawn_count,
        "eye_closed_seconds": round(beh["closed_seconds"], 2),
        "perclos": round(beh["perclos"], 4),
        "left_eye": left_result,
        "right_eye": right_result,
    }


def _empty_result(svc, temporal: bool = True) -> dict[str, Any]:
    """Return empty result when no face detected.

    Only the live path (temporal=True) mutates shared state; one-shot calls
    return the empty payload without touching session counters.
    """
    if temporal:
        state.p_drowsy = 0.0
        state.p_closed = 0.0
        state.p_open = 1.0
        state.p_no_yawn = 1.0
        state.p_yawn = 0.0
        state.top_class = "no_yawn"
        state.mar = None
        state.mouth_box = None
        cascade.reset()
        state.face_box = None
        state.annotated_jpeg = None
        state.lstm_p = None
        state.lstm_alpha = None
        state.lstm_reliable = False
        state.lstm_diag = {}
        state.cluster_id = None
        state.cluster_label = None
        state.cluster_probs = []
        state.is_outlier = False
        state.face_detected = False
        state.frame_count += 1
        if state.started_at:
            state.session_seconds = time.time() - state.started_at
        state.attention_hist.append(0.0)
        if len(state.attention_hist) > 90:
            state.attention_hist = state.attention_hist[-90:]
    return {
        "p_drowsy": 0.0,
        "p_closed": 0.0,
        "p_open": 1.0,
        "p_no_yawn": 1.0,
        "p_yawn": 0.0,
        "top_class": "no_yawn",
        "tier": "normal",
        "face_box": None,
        "mar": None,
        "is_eye_closed": False,
        "is_yawning": False,
        "left_eye": None,
        "right_eye": None,
    }


def _draw_overlay(frame_bgr: np.ndarray, regions: dict[str, Any]) -> np.ndarray:
    """Annotate frame with face box, eye/mouth indicators, and status."""
    out = frame_bgr.copy()
    H, W = out.shape[:2]

    if state.face_box is not None:
        fx, fy, fw, fh = state.face_box

        # Face box with tier color
        color = (80, 200, 120) if state.tier == "normal" else (
            (0, 165, 255) if state.tier == "warning" else (60, 60, 230)
        )
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), color, 2)

        # Eye state label
        eye_status = "EYE CLOSED" if state.p_closed > 0.5 else "EYE OPEN"
        eye_color = (60, 60, 230) if state.p_closed > 0.5 else (80, 200, 120)

        # Draw eye indicators at actual eye positions
        for (ecx, ecy) in [regions.get("left_eye_center"), regions.get("right_eye_center")]:
            if ecx is not None and ecy is not None:
                cv2.circle(out, (ecx, ecy), 6, eye_color, -1)

        # Mouth indicator
        if regions.get("mouth_box"):
            mx1, my1, mx2, my2 = regions["mouth_box"]
            mouth_color = (230, 60, 60) if state.p_yawn > 0.3 else (80, 120, 200)
            cv2.rectangle(out, (mx1, my1), (mx2, my2), mouth_color, 1)

        # Status text
        status = "DROWSY" if state.p_drowsy >= state.threshold else "ALERT"
        status_color = (60, 60, 230) if state.p_drowsy >= state.threshold else (80, 200, 120)
        label = f"{status} P(drowsy)={state.p_drowsy:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (fx, fy - th - 8), (fx + tw + 8, fy), color, -1)
        cv2.putText(out, label, (fx + 4, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Eye & yawn detail
        detail = f"Eye:{state.p_closed:.2f} Yawn:{state.p_yawn:.2f}"
        cv2.putText(out, detail, (fx, fy + fh + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Timestamp
    stamp = time.strftime("%H:%M:%S")
    cv2.putText(out, stamp, (W - 92, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(out, stamp, (W - 93, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return out


def snapshot() -> dict[str, Any]:
    """Serializable view of live state for REST/WS clients."""
    return {
        "running": state.running,
        "session_id": state.session_id,
        "driver_label": state.driver_label,
        "use_browser_cam": state.use_browser_cam,
        "frame_count": state.frame_count,
        "fps": round(state.fps, 1),
        "p_drowsy": round(state.p_drowsy, 4),
        "p_core": round(state.p_drowsy, 4),
        "p_closed": round(state.p_closed, 4),
        "p_open": round(state.p_open, 4),
        "p_no_yawn": round(state.p_no_yawn, 4),
        "p_yawn": round(state.p_yawn, 4),
        "top_class": state.top_class,
        "classes": {
            "Closed": round(state.p_closed, 4),
            "Open": round(state.p_open, 4),
            "no_yawn": round(state.p_no_yawn, 4),
            "yawn": round(state.p_yawn, 4),
        },
        "tier": state.tier,
        "threshold": state.threshold,
        "classification": (
            "drowsy" if state.p_drowsy >= state.threshold and state.face_box else "alert"
        ),
        "face_found": state.face_box is not None,
        "face_box": list(state.face_box) if state.face_box else None,
        "mar": round(state.mar or 0, 4),
        "ear": round(state.ear, 4) if state.ear is not None else None,
        "landmarks": state.landmarks,
        "ear_state": state.ear_state,
        "is_blink": state.is_blink,
        "blink_count": state.blink_count,
        "yawn_count": state.yawn_count,
        "eye_closed_seconds": round(state.eye_closed_seconds, 2),
        "perclos": round(state.perclos, 4),
        "gradcam_png": state.gradcam_b64,
        "gradcam_meta": state.gradcam_meta,
        "temporal": {
            "model": "best_vigidriver_lstm.keras",
            "window": state.temporal_window,
            "window_size": temporal_mod.WINDOW,
            "ready": state.temporal_ready,
            "calibrated": feature_extractor.calibrated,
            "p_drowsy_lstm": round(state.lstm_p, 4) if state.lstm_p is not None else None,
            "gating": {
                "alpha": round(state.lstm_alpha, 4) if state.lstm_alpha is not None else None,
                "reliable": state.lstm_reliable,
                **(state.lstm_diag or {}),
            },
        },
        "kmeans": {
            "fitted": kmeans_gate.ready,
            "cluster_id": state.cluster_id,
            "cluster": state.cluster_label,
            "cluster_probs": state.cluster_probs,
            "is_outlier": state.is_outlier,
        },
        "session_seconds": round(state.session_seconds, 1),
        "attention_hist": [round(p, 3) for p in state.attention_hist],
        "events": state.events[-8:],
        "cam_error": state.cam_error,
        "face_detected": state.face_detected,
        "brightness": state.brightness,
        "jpeg": base64.b64encode(state.annotated_jpeg).decode("ascii") if state.annotated_jpeg else "",
    }


# --------------------------------------------------------------------------- monitor loop

async def monitor_loop() -> None:
    """Grab frames, run inference, update state. Runs continuously."""
    global simulated_frame_count
    logger.info("MONITOR LOOP STARTED")

    while True:
        if state.running and state.session_id is not None:
            # Check viewer timeout — but never during the initial 8s grace
            # window while the camera/face is still settling.
            if (time.time() - state.started_at > 8.0
                    and time.time() - state.last_viewer_ts > VIEWER_TIMEOUT):
                state.running = False
                state.events.append({
                    "ts": time.time(), "tier": "normal", "p_drowsy": state.p_drowsy,
                    "text": "Session auto-ended: no viewer for 30s"
                })
                if state.session_id:
                    db.end_session(state.session_id)
                stop_camera()
                stop_simulation()
                state.session_id = None
                continue

            if state.use_browser_cam:
                # Browser client streams frames directly into /api/monitor/frame
                await asyncio.sleep(0.05)
                continue

            t_frame = time.perf_counter()
            frame = await asyncio.to_thread(read_frame)
            if frame is not None:
                state.cam_error = None
                await asyncio.to_thread(analyze_frame, frame)
            else:
                state.cam_error = "frame read failed"

            elapsed = max(time.perf_counter() - t_frame, 1e-3)
            cur_fps = 1.0 / elapsed
            state.fps = cur_fps if state.fps == 0.0 else (0.8 * state.fps + 0.2 * cur_fps)

            # Dynamic sleep targeting 20 FPS (0.05s total period)
            dt_sleep = max(0.005, (1.0 / 20.0) - (time.perf_counter() - t_frame))
            await asyncio.sleep(dt_sleep)
        else:
            await asyncio.sleep(0.05)


# --------------------------------------------------------------------------- schemas

class StartRequest(BaseModel):
    driver_label: str = "Driver_01"
    threshold: float = Field(0.50, ge=0.05, le=0.95)
    camera_index: int = 0
    use_camera: bool = True
    use_simulation: bool = False
    use_browser_cam: bool = False

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

class ContactRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=5, max_length=160)
    subject: str = Field(default="General enquiry", max_length=160)
    message: str = Field(min_length=5, max_length=4000)

    @field_validator("email")
    @classmethod
    def check_email(cls, v: str) -> str:
        if not EMAIL_RE.match(v):
            raise ValueError("invalid email address")
        return v.lower().strip()


# --------------------------------------------------------------------------- routes

@app.get("/")
def root() -> Any:
    if os.path.isfile(INDEX_HTML):
        from fastapi.responses import FileResponse
        return FileResponse(INDEX_HTML)
    return {
        "name": "VigiDriver API",
        "version": "1.0.0",
        "status": "online",
        "health": "/api/health",
        "docs": "/docs",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    svc = get_service()
    info = svc.model_info()
    return {
        "status": "ok",
        "device": info["device"],
        "primary_model": info["primary_model"],
        "secondary_models_loaded": info["secondary_models_loaded"],
        "models": {
            "eye_cnn": "loaded from models/eye_cnn.keras",
            "yawn_detector": "MAR-based geometric detection",
            "lstm_temporal": (
                "loaded from models/best_vigidriver_lstm.keras (30-frame vigilance Bi-LSTM)"
                if temporal_mod.get_lstm_service().available else "not available"
            ),
            "kmeans_gate": ("fitted k=4 online" if kmeans_gate.ready else "warming up"),
        },
        "db": db.DB_PATH,
        "camera_open": capture is not None and capture.isOpened(),
    }


@app.post("/api/monitor/start")
async def monitor_start(req: StartRequest) -> dict[str, Any]:
    if state.running:
        raise HTTPException(status_code=409, detail="Session already running")

    global _last_eye_ts
    _last_eye_ts = None

    if req.use_simulation:
        stop_camera()
        start_simulation()
        simulated_frame_count = 0
        state.use_browser_cam = False
        state.use_camera = False
    elif req.use_browser_cam:
        stop_camera()
        stop_simulation()
        state.use_browser_cam = True
        state.use_camera = False
    elif req.use_camera:
        state.use_browser_cam = False
        state.use_camera = True
        await asyncio.to_thread(start_camera, req.camera_index)
    else:
        state.use_browser_cam = False
        state.use_camera = False

    state.session_id = db.create_session(req.driver_label, req.threshold)
    state.running = True
    state.driver_label = req.driver_label
    state.threshold = req.threshold
    state.camera_index = req.camera_index
    state.frame_count = 0
    state.fps = 0.0
    state.p_drowsy = 0.0
    state.p_closed = 0.0
    state.p_open = 1.0
    state.p_no_yawn = 1.0
    state.p_yawn = 0.0
    state.top_class = "no_yawn"
    state.tier = "normal"
    state.face_box = None
    state.mouth_box = None
    state.mar = None
    state.attention_hist = []
    state.events = []
    state.cam_error = None
    state.started_at = time.time()
    state.last_viewer_ts = time.time()
    state.session_seconds = 0.0
    state.annotated_jpeg = None
    state.blink_count = 0
    state.yawn_count = 0
    state.eye_closed_seconds = 0.0
    state.perclos = 0.0
    state.is_blink = False
    cascade.reset()
    eye_tracker.reset()
    yawn_tracker.reset()
    ear_closure.reset()
    feature_extractor.reset()
    temporal_buffer.reset()
    kmeans_gate.reset()
    state.ear = None
    state.landmarks = False
    state.ear_state = None
    state.lstm_p = None
    state.lstm_alpha = None
    state.lstm_reliable = False
    state.lstm_diag = {}
    state.temporal_window = 0
    state.temporal_ready = False
    state.cluster_id = None
    state.cluster_label = None
    state.cluster_probs = []
    state.is_outlier = False
    state.gradcam_b64 = ""
    state.gradcam_meta = {}

    return {"session_id": state.session_id, "started": True}


@app.post("/api/monitor/stop")
async def monitor_stop() -> dict[str, Any]:
    if not state.running:
        raise HTTPException(status_code=409, detail="No session running")
    state.running = False
    state.use_browser_cam = False
    await asyncio.sleep(0.3)
    if isinstance(capture, SimulatedCapture):
        stop_simulation()
    else:
        stop_camera()
    session = db.end_session(state.session_id)
    return {"stopped": True, "session": session}


@app.post("/api/monitor/frame")
async def monitor_frame(file: UploadFile = File(...)) -> dict[str, Any]:
    """Ingest and analyze a live video frame pushed from the client browser webcam."""
    if not state.running or state.session_id is None:
        raise HTTPException(status_code=400, detail="No active monitoring session")

    state.last_viewer_ts = time.time()
    t_frame = time.perf_counter()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty frame")

    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image encoding")

    state.cam_error = None
    scores = await asyncio.to_thread(analyze_frame, frame, True)

    elapsed = max(time.perf_counter() - t_frame, 1e-3)
    cur_fps = 1.0 / elapsed
    state.fps = cur_fps if state.fps == 0.0 else (0.8 * state.fps + 0.2 * cur_fps)

    snap = snapshot()
    snap["analysis"] = scores
    return snap



@app.get("/api/monitor/state")
def monitor_state() -> dict[str, Any]:
    state.last_viewer_ts = time.time()
    return snapshot()


@app.get("/api/monitor/cameras")
def list_cameras() -> dict[str, Any]:
    found = []
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(idx)
    return {"cameras": found}


@app.get("/api/cam")
def cam_stream() -> StreamingResponse:
    def gen() -> Iterator[bytes]:
        boundary = b"--frame\r\n"
        while state.running:
            if state.annotated_jpeg is None:
                time.sleep(0.05)
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + state.annotated_jpeg + b"\r\n"
            time.sleep(1 / 15.0)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Not a valid image")

    svc = get_service()

    # Detect face and extract regions (dark synthetic renders fall back to
    # the known simulation face box, mirroring vision.py's heuristic)
    box = detect_face(img)
    if box is None and cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() < 60.0:
        box = (180, 80, 250, 310)
    if box is None:
        return {"face_found": False, "p_drowsy": None}

    regions = extract_eye_mouth_regions(img, box)
    gradcam_png = ""
    gradcam_meta: dict[str, Any] = {}
    g_crop = regions.get("left_eye_crop")
    if g_crop is None:
        g_crop = regions.get("right_eye_crop")
    if g_crop is not None:
        gradcam_png, gradcam_meta = xai_mod.eye_gradcam_png(svc, g_crop)
    scores = analyze_frame(img, temporal=False)  # one-shot: don't pollute live counters

    return {
        "face_found": True,
        "face_box": list(box),
        "p_drowsy": scores["p_drowsy"],
        "p_closed": scores["p_closed"],
        "p_open": scores["p_open"],
        "p_no_yawn": scores["p_no_yawn"],
        "p_yawn": scores["p_yawn"],
        "top_class": scores["top_class"],
        "classes": {
            "Closed": scores["p_closed"],
            "Open": scores["p_open"],
            "no_yawn": scores["p_no_yawn"],
            "yawn": scores["p_yawn"],
        },
        "mar": scores["mar"],
        "is_eye_closed": scores["is_eye_closed"],
        "is_yawning": scores["is_yawning"],
        "gradcam_png": gradcam_png,
        "gradcam_meta": gradcam_meta,
    }


@app.get("/api/model/temporal")
def model_temporal() -> dict[str, Any]:
    """Bi-LSTM temporal micro-sleep model + K-Means gating engine registry."""
    lstm = temporal_mod.get_lstm_service()
    info = lstm.model_info()
    info["kmeans_gate"] = kmeans_gate.describe()
    info["calibration"] = {
        "calibrated": feature_extractor.calibrated,
        "offset": [round(float(v), 5) for v in feature_extractor.baseline_offset.ravel()],
        "mode": "per-driver alert-baseline alignment",
    }
    return info


@app.post("/api/xai/gradcam")
async def xai_gradcam(file: UploadFile = File(...)) -> dict[str, Any]:
    """Grad-CAM + LRP attribution sandbox for an uploaded driver photo."""
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Not a valid image")

    svc = get_service()
    box = detect_face(img)
    if box is None and cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() < 60.0:
        # Very dark synthetic cabin render — fallback known face box (same
        # heuristic as vision.extract_eye_mouth_regions).
        box = (180, 80, 250, 310)
    if box is None:
        raise HTTPException(status_code=422, detail="No face detected in image")

    regions = extract_eye_mouth_regions(img, box)
    eye = regions.get("left_eye_crop")
    if eye is None:
        eye = regions.get("right_eye_crop")
    if eye is None:
        raise HTTPException(status_code=422, detail="Eye region could not be cropped")

    b64, meta = xai_mod.eye_gradcam_png(svc, eye)
    if not b64:
        raise HTTPException(status_code=500, detail="Grad-CAM computation failed")

    lrp = None
    try:
        with xai_mod._xai_lock:
            lrp = xai_mod.compute_lrp(svc.model, eye)
    except Exception:
        lrp = None

    strokes = xai_mod.heat_to_png(
        np.asarray(lrp["lrp_heat"], dtype=np.float32), eye)

    eye_pred = None
    if eye is not None:
        eye_pred = svc.predict_eye(eye)

    return {
        "gradcam_png": b64,
        "gradcam_meta": meta,
        "lrp_heat": (lrp or {}).get("lrp_heat"),
        "lrp_overlay_png": base64.b64encode(strokes).decode("ascii") if strokes else "",
        "eye_pred": eye_pred,
        "localization": {
            "target": "ocular region",
            "conv_layer": meta.get("conv_layer"),
        },
    }


@app.post("/api/robustness/selftest")
async def robustness_selftest(file: UploadFile | None = File(None)) -> dict[str, Any]:
    """Noise robustness characterization across the paper's SNR regimes.

    Runs the uploaded cabin frame (or a synthesized one) through the four
    physics noise regimes + EnlightenGAN-style low-light enhancement and
    re-runs the full pipeline per regime, reporting drowsiness drift.
    """
    if file is not None and file.filename:
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Not a valid image")
    else:
        img = _simulate_frame(10)

    svc = get_service()

    def probe(frame: np.ndarray) -> dict[str, Any]:
        local = FaceTracker(detect_every=1, lost_after_misses=1)
        box = local.update(frame)
        if box is None:
            return {"face_found": False}
        regions = extract_eye_mouth_regions(frame, box)
        eye = regions.get("left_eye_crop")
        if eye is None:
            eye = regions.get("right_eye_crop")
        p_open = 0.5
        if eye is not None:
            p_open = svc.predict_eye(eye)["p_open"]
        return {
            "face_found": True,
            "p_open": round(p_open, 4),
            "p_closed": round(1.0 - p_open, 4),
            "mar": regions.get("mar"),
            "ear": regions.get("ear"),
        }

    rows = []
    for name in noise_mod.REGIMES:
        variant = noise_mod.apply_regime(img, name)
        r = probe(variant)
        brightness = float(variant.mean())
        rows.append({
            "regime": name,
            "label": noise_mod.REGIME_LABELS[name],
            "brightness": round(brightness, 2),
            **r,
        })

    enhanced = noise_mod.enhance_lowlight(noise_mod.apply_regime(img, "awgn_10db"))
    rows.append({
        "regime": "lowlight_gan",
        "label": "Low-light + EnlightenGAN-style harmonization",
        "brightness": round(float(enhanced.mean()), 2),
        **probe(enhanced),
    })

    base = next((r.get("p_open") for r in rows if r["regime"] == "pristine"), None)
    max_drift = 0.0
    if base is not None:
        for r in rows:
            if r.get("p_open") is not None:
                r["drift"] = round(abs(r["p_open"] - base), 4)
                max_drift = max(max_drift, r["drift"])

    return {
        "pristine_p_open": base,
        "max_drift": max_drift,
        "stable": bool(max_drift <= 0.15),
        "regimes": rows,
    }


@app.post("/api/contact")
async def post_contact(req: ContactRequest) -> dict[str, Any]:
    mid = db.add_contact_message(req.name, req.email, req.subject, req.message)
    return {"received": True, "id": mid}


@app.get("/api/contact")
def get_contact() -> dict[str, Any]:
    return {"messages": db.list_contact_messages()}


@app.get("/api/sessions")
def get_sessions(limit: int = 100) -> dict[str, Any]:
    return {"sessions": db.list_sessions(limit)}


@app.get("/api/sessions/{session_id}")
def get_session_detail(session_id: int) -> dict[str, Any]:
    s = db.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": s, "alerts": db.alerts_for_session(session_id)}


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    stats = db.session_stats()
    stats["attention_distribution"] = db.attention_distribution()
    return stats


# --------------------------------------------------------------------------- websocket

@app.websocket("/api/ws/monitor")
async def ws_monitor(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            state.last_viewer_ts = time.time()
            payload = snapshot()
            payload["type"] = "state"
            await ws.send_json(payload)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return


# --------------------------------------------------------------------------- static frontend
if os.path.isdir(DIST_DIR):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def catch_all_spa(full_path: str):
        if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json"):
            raise HTTPException(status_code=404, detail="Not Found")
        target = os.path.join(DIST_DIR, full_path)
        if os.path.isfile(target):
            return FileResponse(target)
        if os.path.isfile(INDEX_HTML):
            return FileResponse(INDEX_HTML)
        raise HTTPException(status_code=404, detail="Not Found")

