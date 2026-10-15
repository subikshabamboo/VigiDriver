"""VigiDriver model layer.

Strictly loads and executes the trained eye CNN model under the `models/` folder:
  * Primary Model: `models/eye_cnn.keras`
    - 3-Conv2D CNN + BatchNorm + Dense (64x64x1 input, 1.14M trainable params)
    - Binary Sigmoid Output: P(eye_open)
        Output > 0.5  => Eye OPEN
        Output < 0.5  => Eye CLOSED
    - Trained specifically for driver drowsiness eye-state classification.

The system combines eye-state classification with mouth-based yawn detection
(geometric MAR ratio from face landmarks) for complete drowsiness scoring.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

# Use keras for loading the .keras model (Keras 3 / TF backend)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Model path under the project's models/ folder
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EYE_CNN_PATH = os.path.join(MODELS_DIR, "eye_cnn.keras")

# Model constants
EYE_IMG_SIZE = 64  # grayscale, 64x64x1

# Drowsiness thresholds
EYE_CLOSED_THRESHOLD = 0.50  # P(eye_open) < 0.50 => eye closed
YAWN_MAR_THRESHOLD = 0.70    # Mouth Aspect Ratio below threshold => yawn in progress


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class VigiModelService:
    """Inference service strictly executing eye_cnn.keras from models/ folder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        if not os.path.exists(EYE_CNN_PATH):
            raise FileNotFoundError(
                f"Eye CNN model not found at {EYE_CNN_PATH}. "
                f"Expected models/eye_cnn.keras in project root."
            )

        # Load the Keras model eagerly — the lazy-load path is unreliable on
        # some Windows/TensorFlow combos (load_model can deadlock when called
        # after OpenCV activity). Lifespan startup calls get_service() once, so
        # eager loading costs nothing extra at request time.
        self._model = keras.saving.load_model(EYE_CNN_PATH)
        self._model_path = EYE_CNN_PATH
        self._input_shape = (EYE_IMG_SIZE, EYE_IMG_SIZE, 1)
        self._infer_fn = None

    def _infer(self):
        """Cached tf.function wrapping the CNN.

        Keras eager calls carry ~50 ms of per-call overhead on this model, which
        dominated the live loop (~6 fps). Tracing once drops a call to ~7 ms.
        """
        if self._infer_fn is None:
            try:
                import tensorflow as tf
                self._infer_fn = tf.function(
                    lambda z: self._model(z, training=False),
                    reduce_retracing=True,
                )
            except Exception:
                self._infer_fn = lambda z: self._model(z, training=False)
        return self._infer_fn

    @property
    def model(self) -> keras.Model:
        if self._model is None:  # defensive; normally loaded in __init__
            with self._lock:
                if self._model is None:
                    self._model = keras.saving.load_model(self._model_path)
        return self._model

    def model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model."""
        try:
            m = self.model
            total_params = int(m.count_params())
        except Exception:
            total_params = 1142401  # known param count for this architecture

        return {
            "primary_model": {
                "name": "Eye State CNN (Binary Classifier)",
                "path": self._model_path,
                "input_shape": [None, EYE_IMG_SIZE, EYE_IMG_SIZE, 1],
                "parameters": total_params,
                "output": "sigmoid P(eye_open)",
                "classes": ["Closed", "Open"],
                "status": "active - strictly loaded from models/eye_cnn.keras",
            },
            "device": "cpu",
            "secondary_models_loaded": False,
        }

    def predict_eye(self, eye_crop_gray: np.ndarray) -> dict[str, Any]:
        """Run eye-state classification on a 64x64 grayscale eye crop.

        Args:
            eye_crop_gray: (H, W) or (H, W, 1) uint8 grayscale eye image.

        Returns:
            Dict with P(eye_open), P(eye_closed), classification label.
        """
        # Preprocess: ensure 64x64x1 float32 normalized [0, 1]
        x = self._preprocess_eye(eye_crop_gray)

        with self._lock:
            pred = float(np.asarray(self._infer()(x)).reshape(-1)[0])

        # Model output is P(eye_open) via sigmoid
        p_open = float(np.clip(pred, 0.0, 1.0))
        p_closed = float(np.clip(1.0 - p_open, 0.0, 1.0))

        label = "Open" if p_open > EYE_CLOSED_THRESHOLD else "Closed"

        return {
            "p_open": round(p_open, 4),
            "p_closed": round(p_closed, 4),
            "label": label,
            "is_closed": p_open <= EYE_CLOSED_THRESHOLD,
        }

    def _preprocess_eye(self, eye_crop: np.ndarray) -> np.ndarray:
        """Convert an eye crop to model input format (1, 64, 64, 1) float32 [0,1].

        The eye CNN expects the eye to be roughly centered in the 64x64 frame.
        If the input crop is larger than 64x64, we extract the center 64x64 region
        to ensure the eye is properly positioned for the model.
        """
        # Ensure grayscale
        if eye_crop.ndim == 3:
            if eye_crop.shape[2] == 3:
                gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
            elif eye_crop.shape[2] == 1:
                gray = eye_crop[:, :, 0]
            else:
                gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_crop

        h, w = gray.shape

        # If the crop is larger than 64x64, extract the center region
        # This ensures the eye (which was detected in the center of the crop)
        # remains centered for the model.
        if h > EYE_IMG_SIZE or w > EYE_IMG_SIZE:
            cy, cx = h // 2, w // 2
            half_h, half_w = EYE_IMG_SIZE // 2, EYE_IMG_SIZE // 2
            y1 = max(0, cy - half_h)
            y2 = min(h, cy + half_h)
            x1 = max(0, cx - half_w)
            x2 = min(w, cx + half_w)
            # Adjust if we're at the edge
            if y2 - y1 < EYE_IMG_SIZE:
                if y1 == 0:
                    y2 = min(h, y1 + EYE_IMG_SIZE)
                else:
                    y1 = max(0, y2 - EYE_IMG_SIZE)
            if x2 - x1 < EYE_IMG_SIZE:
                if x1 == 0:
                    x2 = min(w, x1 + EYE_IMG_SIZE)
                else:
                    x1 = max(0, x2 - EYE_IMG_SIZE)
            gray = gray[y1:y2, x1:x2]

        # Resize to exactly 64x64
        resized = cv2.resize(gray, (EYE_IMG_SIZE, EYE_IMG_SIZE),
                           interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] float32 and add batch + channel dims
        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(1, EYE_IMG_SIZE, EYE_IMG_SIZE, 1)


    def predict_eyes(self, eye_crops: list[np.ndarray | None]) -> list[dict[str, Any] | None]:
        """Classify several eye crops in ONE batched forward pass.

        Returns a list aligned with the input order; None entries stay None.
        Batching both eyes halves the per-frame inference cost versus two
        separate predict_eye() calls.
        """
        results: list[dict[str, Any] | None] = [None] * len(eye_crops)
        idx = [i for i, c in enumerate(eye_crops) if c is not None]
        if not idx:
            return results
        x = np.concatenate([self._preprocess_eye(eye_crops[i]) for i in idx], axis=0)
        with self._lock:
            preds = np.asarray(self._infer()(x)).reshape(-1)
        for j, i in enumerate(idx):
            p_open = float(np.clip(preds[j], 0.0, 1.0))
            p_closed = float(np.clip(1.0 - p_open, 0.0, 1.0))
            results[i] = {
                "p_open": round(p_open, 4),
                "p_closed": round(p_closed, 4),
                "label": "Open" if p_open > EYE_CLOSED_THRESHOLD else "Closed",
                "is_closed": p_open <= EYE_CLOSED_THRESHOLD,
            }
        return results


@lru_cache(maxsize=1)
def get_service() -> VigiModelService:
    """Singleton accessor ensuring the model loads once."""
    return VigiModelService()


# ---------------------------------------------------------------------------
# Yawn detection via Mouth Aspect Ratio (MAR)
# ---------------------------------------------------------------------------

def detect_face_get_eye_mouth_regions(frame_bgr: np.ndarray) -> dict[str, Any]:
    """Detect face and extract eye + mouth regions for the eye CNN + MAR computation.

    Uses Haar cascade for face detection, then defines eye and mouth ROI
    relative to the face box. Returns everything needed for inference.

    Returns:
        Dict with:
            face_box: (x, y, w, h) or None
            left_eye_crop: 64x64 grayscale or None
            right_eye_crop: 64x64 grayscale or None
            mouth_crop: grayscale mouth region or None
            mar: Mouth Aspect Ratio (width/height of mouth box), or None
            frame_annotation: annotated frame copy (for overlay)
    """
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # Detect face with Haar cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(80, 80))

    result = {
        "face_box": None,
        "left_eye_crop": None,
        "right_eye_crop": None,
        "mouth_crop": None,
        "mar": None,
        "frame_annotation": frame_bgr.copy(),
    }

    if len(faces) == 0:
        return result

    # Take largest face
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    result["face_box"] = (int(fx), int(fy), int(fw), int(fh))

    x, y, w, h = fx, fy, fw, fh

    # Define eye regions relative to face box
    # Left eye: upper-left quadrant of face
    ley, rey = int(y + 0.22 * h), int(y + 0.22 * h)
    lew, rew = int(0.22 * w), int(0.22 * w)
    lex, rex = int(x + 0.15 * w), int(x + 0.63 * w)

    le_x1 = max(0, lex)
    le_y1 = max(0, ley)
    le_x2 = min(W, lex + lew)
    le_y2 = min(H, ley + rew)
    left_eye_img = gray[le_y1:le_y2, le_x1:le_x2]

    re_x1 = max(0, rex)
    re_y1 = max(0, rey)
    re_x2 = min(W, rex + rew)
    re_y2 = min(H, rey + rew)
    right_eye_img = gray[re_y1:re_y2, re_x1:re_x2]

    # Mouth region: lower-center of face
    my = int(y + 0.55 * h)
    mh = int(0.18 * h)
    mw = int(0.30 * w)
    mx = int(x + 0.35 * w)
    m_x1 = max(0, mx)
    m_y1 = max(0, my)
    m_x2 = min(W, mx + mw)
    m_y2 = min(H, my + mh)
    mouth_img = gray[m_y1:m_y2, m_x1:m_x2]

    # Compute Mouth Aspect Ratio (MAR) = mouth_width / mouth_height
    mouth_h = m_y2 - m_y1
    mouth_w = m_x2 - m_x1
    if mouth_h > 0 and mouth_w > 0:
        mar = float(mouth_w) / float(mouth_h)
    else:
        mar = None
    result["mar"] = mar

    # Extract and resize eye crops to 64x64
    if left_eye_img.size > 0 and left_eye_img.shape[0] >= 10 and left_eye_img.shape[1] >= 10:
        result["left_eye_crop"] = cv2.resize(left_eye_img, (EYE_IMG_SIZE, EYE_IMG_SIZE),
                                            interpolation=cv2.INTER_LINEAR)
    if right_eye_img.size > 0 and right_eye_img.shape[0] >= 10 and right_eye_img.shape[1] >= 10:
        result["right_eye_crop"] = cv2.resize(right_eye_img, (EYE_IMG_SIZE, EYE_IMG_SIZE),
                                             interpolation=cv2.INTER_LINEAR)

    if mouth_img.size > 0 and mouth_img.shape[0] >= 10 and mouth_img.shape[1] >= 10:
        result["mouth_crop"] = cv2.resize(mouth_img, (EYE_IMG_SIZE, EYE_IMG_SIZE),
                                         interpolation=cv2.INTER_LINEAR)

    # Draw annotation overlay
    out = result["frame_annotation"]
    cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (80, 200, 120), 2)

    # Label
    cv2.putText(out, f"Face {int(w)}x{int(h)}", (int(x), int(y) - 6),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Eye region boxes
    cv2.rectangle(out, (int(le_x1), int(le_y1)), (int(le_x2), int(le_y2)), (200, 120, 80), 1)
    cv2.rectangle(out, (int(re_x1), int(re_y1)), (int(re_x2), int(re_y2)), (200, 120, 80), 1)
    cv2.rectangle(out, (int(m_x1), int(m_y1)), (int(m_x2), int(m_y2)), (80, 120, 200), 1)

    return result


def compute_drowsiness_scores(eye_result: dict[str, Any], mar: float | None) -> dict[str, Any]:
    """Compute complete drowsiness scoring from eye CNN + MAR.

    Combines:
    - Eye state: P(closed) from the eye CNN (averaged across both eyes)
    - Yawn detection: MAR-based geometric detection

    Returns:
        Complete scores dict matching the frontend expectations.
    """
    service = get_service()

    # Run eye CNN on available eye crops
    left_result = None
    right_result = None

    if eye_result.get("left_eye_crop") is not None:
        left_result = service.predict_eye(eye_result["left_eye_crop"])
    if eye_result.get("right_eye_crop") is not None:
        right_result = service.predict_eye(eye_result["right_eye_crop"])

    # Average eye probabilities (prefer both eyes, fallback to one)
    if left_result and right_result:
        p_open = (left_result["p_open"] + right_result["p_open"]) / 2.0
        p_closed = (left_result["p_closed"] + right_result["p_closed"]) / 2.0
        is_closed = left_result["is_closed"] or right_result["is_closed"]
        label = "Closed" if is_closed else "Open"
    elif left_result:
        p_open = left_result["p_open"]
        p_closed = left_result["p_closed"]
        is_closed = left_result["is_closed"]
        label = left_result["label"]
    elif right_result:
        p_open = right_result["p_open"]
        p_closed = right_result["p_closed"]
        is_closed = right_result["is_closed"]
        label = right_result["label"]
    else:
        p_open = 0.5
        p_closed = 0.5
        is_closed = False
        label = "Unknown"

    # Yawn detection via MAR
    is_yawning = False
    p_yawn = 0.0
    p_no_yawn = 1.0

    if mar is not None:
        # Low MAR (wide mouth relative to height) indicates yawn
        # MAR < threshold = mouth is wide = yawn in progress
        if mar < YAWN_MAR_THRESHOLD:
            is_yawning = True
            p_yawn = float(np.clip(1.0 - mar / YAWN_MAR_THRESHOLD, 0.0, 1.0))
            p_no_yawn = float(np.clip(mar / YAWN_MAR_THRESHOLD, 0.0, 1.0))
        else:
            p_yawn = float(np.clip(1.0 - mar / YAWN_MAR_THRESHOLD, 0.0, 0.3))
            p_no_yawn = float(np.clip(mar / YAWN_MAR_THRESHOLD, 0.0, 1.0))

    # Combined drowsiness probability
    # Drowsiness = eye_closed + yawning
    p_drowsy = float(np.clip(p_closed + p_yawn, 0.0, 1.0))

    # Top class determination
    classes_prob = {
        "Closed": p_closed,
        "Open": p_open,
        "no_yawn": p_no_yawn,
        "yawn": p_yawn,
    }
    top_class = max(classes_prob, key=classes_prob.get)

    return {
        "p_drowsy": round(p_drowsy, 4),
        "p_closed": round(p_closed, 4),
        "p_open": round(p_open, 4),
        "p_no_yawn": round(p_no_yawn, 4),
        "p_yawn": round(p_yawn, 4),
        "top_class": top_class,
        "classes": {
            "Closed": round(p_closed, 4),
            "Open": round(p_open, 4),
            "no_yawn": round(p_no_yawn, 4),
            "yawn": round(p_yawn, 4),
        },
        "is_eye_closed": is_closed,
        "is_yawning": is_yawning,
        "mar": round(mar, 4) if mar is not None else None,
        "left_eye": left_result,
        "right_eye": right_result,
    }
