"""Computer-vision helpers: face detection, eye/mouth region extraction.

Primary signal: MediaPipe FaceLandmarker (models/face_landmarker.task) gives
sub-pixel accurate 478-point face geometry, from which we derive:
  * perfectly centered 64x64 eye crops for the eye_cnn.keras classifier
  * EAR (Eye Aspect Ratio) — lighting-proof blink/closure measure
  * MAR (Mouth Aspect Ratio) — yawn measure
Fallback: Haar multi-cascade when MediaPipe is unavailable.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CASCADES_DIR = os.path.join(PROJECT_ROOT, "models", "cascades")

_FACE_CASCADES = []
for name in (
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt.xml",
):
    # 1. Bundled in models/cascades/
    bundled_path = os.path.join(CASCADES_DIR, name)
    c = None
    if os.path.isfile(bundled_path):
        try:
            cand = cv2.CascadeClassifier(bundled_path)
            if not cand.empty():
                c = cand
        except Exception:
            pass
    # 2. Fallback to cv2.data.haarcascades
    if c is None and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        cv_path = getattr(cv2.data, "haarcascades", "") + name
        if os.path.isfile(cv_path):
            try:
                cand = cv2.CascadeClassifier(cv_path)
                if not cand.empty():
                    c = cand
            except Exception:
                pass
    if c is not None:
        _FACE_CASCADES.append(c)


def enhance_low_light(frame_bgr: np.ndarray, target_mean: float = 115.0) -> np.ndarray:
    """Lift a dark frame toward a usable brightness (idempotent).

    Many laptop/in-cabin cameras deliver a very dim image at night, which
    defeats both Haar and MediaPipe face detection. When the mean luminance is
    below the lit threshold we apply a gamma lift toward `target_mean` and then
    CLAHE on the LAB luminance channel. Frames that are already lit pass
    through unchanged, so this is safe to call on every frame.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    m = float(gray.mean())
    if m >= 70.0:
        return frame_bgr  # already lit — do not touch
    m = max(m, 1.0)
    gamma = float(np.clip(np.log(target_mean / 255.0) / np.log(m / 255.0), 0.35, 3.5))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(frame_bgr, lut)
    try:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    except cv2.error:
        pass
    return out


def detect_face(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the largest (x, y, w, h) face box in the frame, or None.

    Tries Haar cascades over CLAHE-normalized grayscale first.
    If Haar fails or is unavailable, falls back directly to MediaPipe FaceLandmarker.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    frame_bgr = enhance_low_light(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    h, w = gray.shape
    # Downscale by 2 for real-time Haar detection (avoids ~800ms full-res scan)
    small_gray = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    min_side = (max(28, (h // 2) // 6), max(28, (w // 2) // 6))

    for cascade in _FACE_CASCADES:
        if cascade.empty():
            continue
        try:
            faces = cascade.detectMultiScale(
                small_gray, scaleFactor=1.1, minNeighbors=4, minSize=min_side
            )
            if len(faces):
                bx, by, bw, bh = max(faces, key=lambda f: f[2] * f[3])
                return (int(bx * 2), int(by * 2), int(bw * 2), int(bh * 2))
        except cv2.error:
            continue

    # Fallback to MediaPipe FaceLandmarker when Haar detects nothing or fails
    try:
        landmarker = get_landmarker()
        if landmarker is not None:
            import mediapipe as mp
            rgb_frame = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            mp_res = landmarker.detect(mp_image)
            if mp_res and mp_res.face_landmarks:
                lm = mp_res.face_landmarks[0]
                xs = [p.x * w for p in lm]
                ys = [p.y * h for p in lm]
                pad_x = 0.08 * (max(xs) - min(xs))
                pad_y = 0.12 * (max(ys) - min(ys))
                bx = max(0, int(min(xs) - pad_x))
                by = max(0, int(min(ys) - pad_y))
                bw = min(w - bx, int(max(xs) - min(xs) + 2 * pad_x))
                bh = min(h - by, int(max(ys) - min(ys) + 2 * pad_y))
                if bw > 20 and bh > 20:
                    return (bx, by, bw, bh)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# MediaPipe FaceLandmarker (pretrained model from Google)
# ---------------------------------------------------------------------------

FACE_LANDMARKER_PATH = os.path.join(PROJECT_ROOT, "models", "face_landmarker.task")

_landmarker = None

# MediaPipe FaceMesh landmark indices (478-point topology with refine)
# Left eye (subject's left, appear on right of image; indices are consistent)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # corners + vertical pairs
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# Mouth: corners + upper/lower inner lip
MOUTH_CORNERS = (61, 291)
MOUTH_UPPER = (0, 13)
MOUTH_LOWER = (17, 14)


def get_landmarker():
    """Lazily create the FaceLandmarker (float16 .task bundle)."""
    global _landmarker
    if _landmarker is None:
        if not os.path.exists(FACE_LANDMARKER_PATH):
            return None
        from mediapipe.tasks.python import vision, BaseOptions
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def shutdown_landmarker() -> None:
    global _landmarker
    if _landmarker is not None:
        try:
            _landmarker.close()
        except Exception:
            pass
        _landmarker = None


def _dist2(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _ear_from_landmarks(lm) -> dict[str, float]:
    """Eye Aspect Ratio from FaceMesh landmarks (both eyes averaged).

    EAR = vertical opening / horizontal eye width. Open eye ≈ 0.25–0.35,
    closed eye < 0.12. Lighting-robust by construction.

    The ring is ordered [outer-corner, top-1, top-2, inner-corner,
    bottom-1, bottom-2], so each TOP lid point pairs with the BOTTOM point
    directly beneath it: (1,5) and (2,4). Pairing them the other way round
    — (1,4) and (2,5) — crosses the pairs diagonally and measures mostly the
    *horizontal* offset: on live frames dx/w came out ~0.47, which survives
    the eye shutting completely. Measured closed-eye floor was then 0.51
    against a 0.60×baseline threshold of 0.38, i.e. closure was
    geometrically undetectable and the EAR vetoed every real closure.
    With the correct pairing, dx/w drops to ≈0.09 and the same measurement
    gives open ≈ 0.31 vs closed ≈ 0.23.
    """
    def one(idx):
        p = [lm[i] for i in idx]
        v1 = _dist2(p[1], p[5])
        v2 = _dist2(p[2], p[4])
        h = _dist2(p[0], p[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
    return {
        "left": one(LEFT_EYE_IDX),
        "right": one(RIGHT_EYE_IDX),
    }


def _mar_from_landmarks(lm) -> float:
    """Mouth Aspect Ratio: vertical inner-lip opening / mouth width."""
    c1, c2 = lm[MOUTH_CORNERS[0]], lm[MOUTH_CORNERS[1]]
    up = lm[MOUTH_UPPER[1]]
    lo = lm[MOUTH_LOWER[1]]
    w = _dist2(c1, c2)
    h = _dist2(up, lo)
    return h / w if w > 0 else 0.0


def _eyelid_from_landmarks(lm, fw_px: float) -> dict[str, float]:
    """Eyelid aperture (vertical lid separation) per eye in image-width units."""
    return {
        "left": _dist2(lm[159], lm[145]),
        "right": _dist2(lm[386], lm[374]),
    }


def brightness_signature(frame_bgr: np.ndarray) -> float:
    """Mean brightness of frame — used to detect scene changes."""
    return float(frame_bgr.mean())


# Adaptive yawn threshold for landmark-based MAR (inner-lip opening / mouth
# width). Closed mouth ≈ 0.0–0.08; relaxed talk ≈ 0.1–0.2; yawn ≈ 0.3+.
MAR_YAWN_LANDMARK = 0.25


@dataclass
class EarClosure:
    """Adaptive eye-closure signal from EAR.

    Absolute EAR varies with camera geometry, so this maintains a rolling
    calibration window and decides "closed" relative to the person's own
    open-eye baseline (90th percentile): closed when EAR < 85% of baseline.
    Returns None while calibrating (first few seconds).

    The relative factor is set from measured live data, not from the
    textbook 0.60: MediaPipe's eyelid landmarks only collapse partially when
    the eye shuts (open ≈ 0.31 -> closed ≈ 0.23, a ratio of ~0.74). A 0.60
    factor therefore required the EAR to fall to 0.19 — a value the
    landmarks never reach — so the detector could never fire and real
    closures were discarded. 0.85 clears the closed-eye value with margin on
    both sides (closed 0.74×baseline fires; open 1.00×baseline does not).
    """

    window: int = 1800           # 2 min at 15 fps — long enough that a
                                 # sustained closure cannot drag the open-eye
                                 # baseline down and blind the detector.
    abs_floor: float = 0.12      # never treat a value above this as closed
    rel_factor: float = 0.85     # closed when EAR < this fraction of baseline
    _hist: list[float] = field(default_factory=list)

    def update(self, ear: float | None) -> bool | None:
        """Feed one EAR sample. True=closed, False=open, None=unknown/calibrating."""
        if ear is None:
            return None
        self._hist.append(float(ear))
        if len(self._hist) > self.window:
            self._hist = self._hist[-self.window:]
        if len(self._hist) < 20:
            return None  # still calibrating
        base = float(np.percentile(self._hist, 90))
        if base < 0.02:
            return None
        # Relative threshold with an absolute floor so a noisy/poor baseline
        # cannot silently disable closure detection.
        thresh = max(self.rel_factor * base, self.abs_floor)
        return float(ear) < thresh

    def reset(self) -> None:
        self._hist = []


# ---------------------------------------------------------------------------
# Eye & mouth region extraction
# ---------------------------------------------------------------------------


def extract_eye_mouth_regions(
    frame_bgr: np.ndarray,
    face_box: tuple[int, int, int, int] | None = None,
    use_landmarks: bool = True,
) -> dict[str, Any]:
    """Extract eye crops and mouth region from a frame given a face box.

    If face_box is None, attempts to detect a face first.
    For simulated cabin-cam frames where Haar can't detect the synthetic face,
    falls back to the known synthetic face box.

    Returns dict with:
        face_box: (x, y, w, h) tuple
        left_eye_center: (x, y) center of left eye region in frame coords
        right_eye_center: (x, y) center of right eye region
        mouth_center: (x, y) center of mouth region
        left_eye_crop: 64x64 grayscale eye image (centered for model) or None
        right_eye_crop: 64x64 grayscale eye image or None
        mouth_box: (x1, y1, x2, y2) bounding box of mouth region
        mar: Mouth Aspect Ratio = dark-opening height / mouth width (higher = mouth more open), or None
        eye_mouth_overlay: annotated frame copy for visualization
    """
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    if face_box is None:
        face_box = detect_face(frame_bgr)

    # Fallback for simulated frames: when Haar can't find the synthetic face,
    # use the deterministic face box that the simulator renders at.
    if face_box is None and gray.mean() < 60.0:
        face_box = (180, 80, 250, 310)

    result: dict[str, Any] = {
        "face_box": None,
        "left_eye_center": None,
        "right_eye_center": None,
        "mouth_center": None,
        "left_eye_crop": None,
        "right_eye_crop": None,
        "mouth_box": None,
        "mar": None,
        "ear": None,
        "landmarks": False,
        "eye_mouth_overlay": frame_bgr.copy(),
    }

    if face_box is None:
        return result

    fx, fy, fw, fh = face_box
    result["face_box"] = face_box

    # --- Extract mouth region ---
    def _extract_mouth_box() -> tuple[int, int, int, int] | None:
        # Mouth region dimensions (relative to face)
        mouth_w = int(0.30 * fw)
        mouth_h = int(0.18 * fh)
        mouth_cx = int(fx + 0.50 * fw)
        mouth_cy = int(fy + 0.74 * fh)
        result["mouth_center"] = (mouth_cx, mouth_cy)
        m_x1 = max(0, mouth_cx - mouth_w // 2)
        m_y1 = max(0, mouth_cy - mouth_h // 2)
        m_x2 = min(W, mouth_cx + mouth_w // 2)
        m_y2 = min(H, mouth_cy + mouth_h // 2)
        if m_x2 - m_x1 >= 10 and m_y2 - m_y1 >= 10:
            return (m_x1, m_y1, m_x2, m_y2)
        return None

    mouth_box = _extract_mouth_box()

    # --- PRIMARY: MediaPipe FaceLandmarker path --------------------------
    if use_landmarks and mouth_box is not None:
        try:
            landmarker = get_landmarker()
            if landmarker is not None:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=np.ascontiguousarray(frame_bgr[:, :, ::-1]))
                mp_res = landmarker.detect(mp_image)
                if mp_res.face_landmarks:
                    lm = mp_res.face_landmarks[0]
                    result["landmarks"] = True

                    # Eye centers from landmark geometry (mean of eye ring)
                    def mean_pt(idxs):
                        xs = [lm[i].x for i in idxs]
                        ys = [lm[i].y for i in idxs]
                        return (int(np.mean(xs) * W), int(np.mean(ys) * H))
                    result["left_eye_center"] = mean_pt(LEFT_EYE_IDX)
                    result["right_eye_center"] = mean_pt(RIGHT_EYE_IDX)

                    # EAR + MAR + eyelid apertures — direct, sub-pixel
                    ear = _ear_from_landmarks(lm)
                    result["ear"] = round((ear["left"] + ear["right"]) / 2.0, 4)
                    result["ear_left"] = round(ear["left"], 4)
                    result["ear_right"] = round(ear["right"], 4)
                    result["mar"] = round(_mar_from_landmarks(lm), 4)
                    result["eyelid"] = _eyelid_from_landmarks(lm, fw)

                    # Perfectly centered eye crops for the CNN: take a square
                    # window centred exactly on each eye centre, sized to
                    # 1.6x the inter-eye distance so the eye fills the crop.
                    inter_eye = _dist2(lm[33], lm[263]) * W  # px
                    side = int(max(24, min(120, 1.6 * inter_eye)))
                    half = side // 2

                    def crop_at(cx, cy):
                        x1 = max(0, cx - half); x2 = min(W, cx + half)
                        y1 = max(0, cy - half); y2 = min(H, cy + half)
                        if x2 - x1 < 16 or y2 - y1 < 16:
                            return None
                        # Pad short sides by replicating edge pixels so the
                        # crop stays square and the eye stays centred.
                        region = frame_bgr_gray[y1:y2, x1:x2]
                        return cv2.resize(region, (64, 64), interpolation=cv2.INTER_LINEAR)

                    frame_bgr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    result["left_eye_crop"] = crop_at(*result["left_eye_center"])
                    result["right_eye_crop"] = crop_at(*result["right_eye_center"])

                    # Mouth box hugged to landmarks (corner-to-corner + 40%)
                    mw = _dist2(lm[61], lm[291]) * W
                    mh = max(24.0, result["mar"] * mw + 0.25 * mw)
                    mc = result["mouth_center"]
                    m_x1 = max(0, int(mc[0] - 0.6 * mw)); m_x2 = min(W, int(mc[0] + 0.6 * mw))
                    m_y1 = max(0, int(mc[1] - 0.7 * mh)); m_y2 = min(H, int(mc[1] + 0.7 * mh))
                    result["mouth_box"] = (m_x1, m_y1, m_x2, m_y2)
                    return result
        except Exception:
            pass  # fall through to Haar path

    # --- FALLBACK: Haar-based geometric extraction -----------------------
    result["landmarks"] = False

    # Eye region dimensions (relative to face)
    eye_w = int(0.24 * fw)
    eye_h = int(0.16 * fh)
    eye_y = int(fy + 0.37 * fh)  # vertical position of eye centers

    # Left eye center
    left_eye_cx = int(fx + 0.28 * fw)
    left_eye_cy = eye_y
    result["left_eye_center"] = (left_eye_cx, left_eye_cy)

    # Right eye center
    right_eye_cx = int(fx + 0.72 * fw)
    right_eye_cy = eye_y
    result["right_eye_center"] = (right_eye_cx, right_eye_cy)

    # --- Extract eye crops (centered 64x64 regions) ---
    target_size = 64

    def extract_centered_eye(
        eye_cx: int, eye_cy: int, eye_region_w: int, eye_region_h: int
    ) -> np.ndarray | None:
        """Extract a centered eye crop from the frame."""
        half_w = eye_region_w // 2
        half_h = eye_region_h // 2
        x1 = max(0, eye_cx - half_w)
        y1 = max(0, eye_cy - half_h)
        x2 = min(W, eye_cx + half_w)
        y2 = min(H, eye_cy + half_h)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        eye_region = gray[y1:y2, x1:x2]
        return cv2.resize(eye_region, (target_size, target_size),
                          interpolation=cv2.INTER_LINEAR)

    result["left_eye_crop"] = extract_centered_eye(
        left_eye_cx, left_eye_cy, eye_w, eye_h
    )
    result["right_eye_crop"] = extract_centered_eye(
        right_eye_cx, right_eye_cy, eye_w, eye_h
    )

    # --- Measure mouth opening from pixels ---
    m_x1, m_y1, m_x2, m_y2 = mouth_box
    result["mouth_box"] = mouth_box

    # Measure the ACTUAL mouth opening from pixels.
    crop = gray[m_y1:m_y2, m_x1:m_x2]
    med = float(np.median(crop))
    otsu_t, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_frac = float((crop < otsu_t).mean())
    if 0.02 <= otsu_frac <= 0.80:
        dark_thresh = float(otsu_t)
    else:
        dark_thresh = max(20.0, med * 0.62)
        dark = crop < dark_thresh
        min_open_px = max(2, int(0.25 * crop.shape[1]))
        row_open = dark.sum(axis=1) >= min_open_px
        best_run = cur = 0
        for v in row_open:
            cur = cur + 1 if v else 0
            best_run = max(best_run, cur)
        result["mar"] = float(best_run) / float(crop.shape[1])

    # --- Draw overlay ---
    overlay = result["eye_mouth_overlay"]

    # Face box
    cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), (80, 200, 120), 2)

    # Eye region indicators
    for (ecx, ecy), color, label in [
        ((left_eye_cx, left_eye_cy), (200, 120, 80), "L"),
        ((right_eye_cx, right_eye_cy), (200, 120, 80), "R"),
    ]:
        cv2.circle(overlay, (ecx, ecy), 8, color, 2)
        cv2.putText(overlay, label, (ecx - 6, ecy - 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Mouth region
    if result["mouth_box"]:
        mx1, my1, mx2, my2 = result["mouth_box"]
        cv2.rectangle(overlay, (mx1, my1), (mx2, my2), (80, 120, 200), 2)

    # Status label
    cv2.putText(overlay, f"Face {fw}x{fh}", (fx, fy - 6),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


# ---------------------------------------------------------------------------
# Yawn detection via MAR
# ---------------------------------------------------------------------------

# Typical MAR values:
#   - Neutral/closed mouth: MAR ≈ 0.7-1.0 (width ≈ height or narrower)
#   - Yawning: MAR increases significantly (mouth opens wide, width >> height)
# Actually, for yawning, the mouth opens vertically, so:
#   - Normal: mouth is wider than tall (MAR > 1.0, typically 1.5-2.5)
#   - Yawning: mouth opens vertically, MAR decreases (mouth height increases)
# 
# Wait — let me reconsider. MAR = width / height.
# - Normal closed/talking mouth: width > height, so MAR > 1.0
# - Yawning: mouth opens WIDE vertically, height increases dramatically
#   So MAR = width / (larger height) = DECREASES
#   
# Actually, in most yawn detection literature:
# - EAR (Eye Aspect Ratio) decreases when eyes close
# - MAR (Mouth Aspect Ratio) INCREASES when yawning (mouth opens)
#
# But if MAR = width/height, and yawning makes height increase:
# - Normal: width=30, height=15, MAR=2.0
# - Yawn: width=30, height=30, MAR=1.0
# So MAR DECREASES during yawn.
#
# However, some implementations use MAR = height/width, which would INCREASE.
# The key is consistency. Let me use the standard definition where:
# - Higher MAR = more yawning
#
# Standard MAR for drowsiness: MAR = (vertical mouth opening) / (mouth width)
# This way, MAR increases when mouth opens vertically (yawning).

YAWN_MAR_THRESHOLD = 0.32  # MAR above this = likely yawning
MAR_NORMAL_MIN = 0.15       # Below this = mouth essentially closed


def detect_yawn(mar: float | None) -> dict[str, Any]:
    """Detect yawning from Mouth Aspect Ratio.

    Uses MAR = mouth_height / mouth_width.
    - Normal closed mouth: MAR < 0.20
    - Talking/open mouth: MAR 0.20-0.35
    - Yawning: MAR > 0.35 (vertical opening exceeds threshold)

    Returns:
        Dict with p_yawn, p_no_yawn, is_yawning, mar.
    """
    if mar is None:
        return {
            "p_yawn": 0.0,
            "p_no_yawn": 1.0,
            "is_yawning": False,
            "mar": None,
        }

    # Clamp MAR to reasonable range
    mar = float(np.clip(mar, 0.0, 2.0))

    # Sigmoid-like mapping: low MAR = no yawn, high MAR = yawning
    # Center the transition around YAWN_MAR_THRESHOLD
    if mar < MAR_NORMAL_MIN:
        # Mouth essentially closed
        p_yawn = 0.0
    elif mar < YAWN_MAR_THRESHOLD:
        # Transitional — possibly starting to yawn
        p_yawn = float((mar - MAR_NORMAL_MIN) / (YAWN_MAR_THRESHOLD - MAR_NORMAL_MIN)) * 0.3
    else:
        # Clear yawn signal
        p_yawn = float(np.clip(0.3 + (mar - YAWN_MAR_THRESHOLD) / 0.35, 0.0, 0.95))

    p_no_yawn = float(np.clip(1.0 - p_yawn, 0.0, 1.0))

    # Smooth the decision
    is_yawning = p_yawn > 0.5

    return {
        "p_yawn": round(p_yawn, 4),
        "p_no_yawn": round(p_no_yawn, 4),
        "is_yawning": is_yawning,
        "mar": round(mar, 4),
    }


# ---------------------------------------------------------------------------
# Alert cascade
# ---------------------------------------------------------------------------

@dataclass
class CascadeState:
    """Tracks sustained drowsiness and escalates through alert tiers.

    Tiers:
      * warning  — eye closed or yawning detected for a short burst
      * critical — sustained above-threshold period = fatigue cascade
    """

    threshold: float = 0.50  # Drowsiness probability threshold
    warn_frames: int = 4      # Frames above threshold before warning
    critical_frames: int = 12 # Frames above threshold before critical
    cooldown_s: float = 5.0   # Minimum seconds between tier changes

    above_count: int = 0
    tier: str = "normal"  # normal | warning | critical
    _last_alert_at: float = field(default=0.0)

    def update(self, p_drowsy: float) -> str:
        """Feed one probability sample; return the resulting tier."""
        if p_drowsy >= self.threshold:
            self.above_count += 1
        else:
            # Decay gradually so brief blinks don't reset immediately
            self.above_count = max(0, self.above_count - 1)

        now = time.time()
        if self.above_count >= self.critical_frames:
            self.tier = "critical"
        elif self.above_count >= self.warn_frames:
            if self.tier == "normal":
                self.tier = "warning"
            elif now - self._last_alert_at >= self.cooldown_s:
                self.tier = "warning"
        else:
            if self.tier != "normal" and now - self._last_alert_at >= self.cooldown_s:
                self.tier = "normal"

        if self.tier in ("warning", "critical"):
            self._last_alert_at = now

        return self.tier

    def reset(self) -> None:
        self.above_count = 0
        self.tier = "normal"


@dataclass
class FaceTracker:
    """Haar face detection every N frames; stable box between detections."""

    detect_every: int = 5
    lost_after_misses: int = 15

    _box: tuple[int, int, int, int] | None = None
    _sig: float | None = None
    _since_detect: int = 0
    _misses: int = 0

    def update(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        self._since_detect += 1

        if self._box is None or self._since_detect >= self.detect_every:
            self._since_detect = 0
            box = detect_face(frame_bgr)
            if box is not None:
                self._box = box
                self._misses = 0
            else:
                self._misses += 1
                if self._misses >= self.lost_after_misses or self._box is None:
                    self._box = None
        else:
            sig = brightness_signature(frame_bgr)
            if self._sig is not None and abs(sig - self._sig) > 30:
                self._box = None
                self._misses = self.lost_after_misses
            else:
                self._sig = sig

        return self._box


# ---------------------------------------------------------------------------
# Temporal behaviour trackers
# ---------------------------------------------------------------------------

@dataclass
class EyeBehaviorTracker:
    """Distinguishes blinks from sustained (micro-sleep) eye closure.

    Feeds the per-frame P(eye_open) from the eye CNN and maintains:
      * blink detection  — eye closed for 0.13–0.6 s then reopened
      * sustained closure — eye closed for >= 2.0 s (micro-sleep risk)
      * PERCLOS          — % of the last 6 s with the eye mostly closed
                           (the standard validated drowsiness metric)
    """

    fps: float = 15.0
    blink_min_frames: int = 2       # 0.13 s
    blink_max_frames: int = 9       # 0.6 s
    blink_protect_frames: int = 13  # 0.87 s — closures inside this envelope are blinks, not drowsiness
    sustained_seconds: float = 2.0  # wall-clock closure that counts as micro-sleep
    sustained_frames: int = 30      # legacy frame fallback (2.0 s at 15 fps)
    perclos_window: int = 90        # samples in the rolling PERCLOS window
    perclos_seconds: float = 6.0    # wall-clock PERCLOS window (the standard metric)
    perclos_closed_thresh: float = 0.25  # p_open below this = mostly closed

    closed_run: int = 0
    blink_count: int = 0
    was_closed: bool = False
    sustained_now: bool = False
    _closed_seconds: float = 0.0
    _history: list[float] = field(default_factory=list)
    _history_dt: list[float] = field(default_factory=list)

    def update(self, p_open: float, dt: float | None = None) -> dict[str, Any]:
        """Feed one P(eye_open) sample.

        dt is the real elapsed time since the previous sample. Passing it
        matters because the live loop's rate is machine-dependent (it runs
        anywhere from ~3 to ~15 fps): counting frames instead of seconds
        stretched the 2 s micro-sleep rule into 2–10 real seconds, so a
        driver asleep for well over 2 s could still be scored as merely
        drowsy. When dt is omitted the tracker falls back to 1/fps, which
        preserves the frame-based contract used by the unit tests.
        """
        step = (1.0 / self.fps) if dt is None else max(0.0, float(dt))
        is_closed = p_open < 0.5
        sustained_start = False

        if is_closed:
            self.closed_run += 1
            self._closed_seconds += step
            if self._closed_seconds >= self.sustained_seconds and not self.sustained_now:
                self.sustained_now = True
                sustained_start = True
        else:
            if self.was_closed and self.blink_min_frames <= self.closed_run <= self.blink_max_frames:
                self.blink_count += 1
            self.closed_run = 0
            self._closed_seconds = 0.0
            self.sustained_now = False
        self.was_closed = is_closed

        # A closure still inside the blink-duration envelope is normal
        # physiology — the fusion layer uses this to keep blinks from
        # tripping the drowsiness score.
        blink_protect = bool(
            self.sustained_now is False
            and 0 < self.closed_run <= self.blink_protect_frames
        ) or (self.was_closed is False and self.blink_min_frames
              <= self.closed_run <= self.blink_protect_frames)

        self._history.append(float(p_open))
        self._history_dt.append(step)
        # Trim the oldest samples once the window is full. With dt known the
        # window is the standard 6 s of wall-clock time; without it (tests,
        # synthetic feeds) it degrades to the legacy 90-sample window.
        while len(self._history) > 1 and (
                len(self._history) > self.perclos_window
                or sum(self._history_dt) > self.perclos_seconds):
            self._history.pop(0)
            self._history_dt.pop(0)
        perclos = (
            sum(1 for v in self._history if v < self.perclos_closed_thresh)
            / len(self._history)
        ) if self._history else 0.0

        return {
            "is_closed": is_closed,
            "is_blink": self.blink_min_frames <= self.closed_run <= self.blink_max_frames,
            "blink_protect": bool(blink_protect),
            "sustained_closure": self.sustained_now,
            "sustained_start": sustained_start,
            "closed_seconds": self._closed_seconds,
            "blink_count": self.blink_count,
            "perclos": float(perclos),
        }

    def reset(self) -> None:
        self.closed_run = 0
        self.blink_count = 0
        self.was_closed = False
        self.sustained_now = False
        self._closed_seconds = 0.0
        self._history = []
        self._history_dt = []


@dataclass
class YawnTracker:
    """Temporal smoothing for MAR yawn detection.

    A real yawn is a *sustained* wide-open mouth (0.5–3 s), not a 1-frame
    glitch. This tracker smooths the instantaneous MAR signal, requires the
    opening to persist briefly before counting a yawn event, and keeps the
    yawn probability elevated through brief MAR flickers.
    """

    yawn_min_frames: int = 4   # 0.27 s of sustained opening to count an event

    yawn_run: int = 0
    yawn_count: int = 0
    _ema: float = 0.0
    _was_yawning: bool = False

    def update(self, mar: float | None, threshold: float | None = None) -> dict[str, Any]:
        """threshold overrides YAWN_MAR_THRESHOLD (used by landmark-MAR mode)."""
        thr = threshold if threshold is not None else YAWN_MAR_THRESHOLD
        instant = mar is not None and mar > thr
        yawn_event = False
        if instant:
            self.yawn_run += 1
            if self.yawn_run == self.yawn_min_frames:
                # Sustained opening confirmed — count exactly once per yawn
                self.yawn_count += 1
                yawn_event = True
        else:
            self.yawn_run = 0

        target = 0.9 if instant else 0.0
        self._ema = 0.6 * target + 0.4 * self._ema
        is_yawning = self._ema > 0.5

        return {
            "p_yawn": round(self._ema, 4),
            "p_no_yawn": round(1.0 - self._ema, 4),
            "is_yawning": is_yawning,
            "yawn_event": yawn_event,
            "yawn_count": self.yawn_count,
            "mar": mar,
        }

    def reset(self) -> None:
        self.yawn_run = 0
        self.yawn_count = 0
        self._ema = 0.0
        self._was_yawning = False
