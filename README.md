---
title: VigiDriver - Edge-AI Driver Monitoring System
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# VigiDriver: A Spatiotemporal Edge-AI Framework for Predictive Driver Monitoring

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-4285F4.svg)](https://developers.google.com/mediapipe)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VigiDriver** is an end-to-end, production-grade Driver Monitoring System (DMS) that shifts drowsiness detection from *reactive* alerts (triggered only after micro-sleep occurs) to *predictive* spatiotemporal vigilance analysis.

Designed for edge deployment and resilient against challenging in-cabin environments (night driving, glare, shadow, low illumination), VigiDriver fuses dual-eye spatial CNN inference, 468-point facial landmark biometrics, 30-frame Bi-LSTM temporal sequence memory, unsupervised K-Means outlier gating, and Explainable AI (Grad-CAM) in real time at **15–25 FPS**.

---

## 🏗️ System Architecture & 7-Stage Pipeline

```text
                     CAMERA FRAME (640×480 RGB)
                                 │
                                 ▼  [1. Ingestion & Camera Buffering]
                 • Browser Webcam (HTML5 getUserMedia)
                 • Simulation Stream (Synthetic cabin fatigue ramp)
                 • Local Host Webcam (cv2.VideoCapture, BufferSize=1)
                                 │
                                 ▼  [2. Adaptive Low-Light Recovery]
                 RGB ──► LAB space ──► CLAHE on L-channel (clip 2.0, 8×8)
                                 │
                                 ▼  [3. MediaPipe Face Mesh Extraction]
                 Extracts 468 landmarks:
                 • Left Eye:  [362, 385, 387, 263, 373, 380]
                 • Right Eye: [33, 160, 158, 133, 153, 144]
                 • Mouth:     [13, 14, 61, 291]
                 • Head Pose: 3D rotation vectors (Yaw, Pitch, Roll)
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
  [STREAM A: Spatial CNN]   [STREAM B: Biometrics]  [STREAM C: Sequence Window]
  • Left & Right 64×64      • EAR (Eye Aspect Ratio) • 4-D Feature Vector:
    grayscale ocular crops  • Adaptive baseline       [EAR, EAR_ratio, MAR, Yaw]
  • Batched forward pass:     calibration            • 30-frame FIFO buffer
    Tensor (2, 64, 64, 1)   • MAR (Mouth Aspect)     • 30.6K-parameter Bi-LSTM
  • eye_cnn.keras (~7 ms)   • PERCLOS (% Closure)      unrolls temporal context
  • Outputs P(eye_open)     • Blink vs sustained     • Outputs P(drowsy_lstm)
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                 [4. Self-Gated Reliability Fusion]
                 α = Support(z) × Agreement(LSTM, CNN) × Calibrated
                 • If CNN reports eyes open, α collapses to preserve alert state
                 • Fused Score = (1 - α) × Geometric + α × LSTM
                                 │
                                 ▼
                 [5. Unsupervised K-Means Gating]
                 • Projects latent features into k=4 behavioral clusters
                 • Rejects Out-Of-Distribution (OOD) artifacts & noise
                                 │
                                 ▼
                 [6. Multi-Tier Cascade Alert Engine]
                 • Normal (P < 0.50): Alert monitoring state
                 • Warning (P ≥ 0.50): Fatigue building chime & speech
                 • Critical (Sustained closure ≥ 2.0s): High-urgency siren
                                 │
                                 ▼
                 [7. Explainable AI & Telemetry Dispatch]
                 • Grad-CAM generates ocular saliency attribution heatmaps
                 • Real-time telemetry streamed to React Dashboard
                 • Trip audit logs persisted to SQLite (vigidriver.db)
```

---

## ✨ Key Technical Highlights

| Component | Technical Implementation | Purpose / Advantage |
| :--- | :--- | :--- |
| **Primary Spatial CNN** | `models/eye_cnn.keras`<br>3-Conv2D + BatchNorm + Dense (1.14M params) | Classifies ocular openness $P(\text{eye\_open})$ from batched 64×64 crops in ~7 ms on CPU. |
| **Facial Biometrics** | `models/face_landmarker.task`<br>468-point dense 3D mesh | Tracks EAR, MAR, and dynamic PERCLOS. Calibrates to each individual driver's open-eye baseline. |
| **Temporal Bi-LSTM** | `models/best_vigidriver_lstm.keras`<br>Bi-directional LSTM (30.6K params) | Analyzes 30-frame rolling windows (2-second context) to detect slow eyelid closure velocity. |
| **Self-Gated Reliability** | Dynamic weighting ($\alpha \in [0.015, 0.50]$) | Automatically attenuates the LSTM whenever it disagrees with clear visual evidence. |
| **Latent K-Means Gate** | Online fitted clustering ($k=4$) | Identifies out-of-distribution head movements and suppresses false alarms. |
| **Explainable AI (XAI)** | Grad-CAM activation mapping | Visualizes which ocular regions triggered the drowsiness classification. |
| **Remote Browser Webcam** | HTML5 `getUserMedia` + JPEG streaming | Allows anyone on any smartphone or laptop to test their own face over the web with zero installation. |

---

## 📂 Repository Structure

```text
VigiDriver/
├── backend/
│   ├── app.py              # FastAPI application, REST endpoints, and WebSocket monitor
│   ├── models.py           # Model service loading models/eye_cnn.keras
│   ├── temporal.py         # 30-frame vigilance extractor & Bi-LSTM inference service
│   ├── vision.py           # Spatial downscaled face tracking, EAR, MAR, and alert cascade
│   ├── kmeans.py           # Unsupervised k=4 behavioral clustering & outlier gate
│   ├── xai.py              # Grad-CAM attribution and saliency generation
│   ├── noise.py            # Low-light enhancement and noise robustness testing
│   ├── db.py               # SQLite session, alert, and telemetry persistence
│   └── run.py              # Standalone Uvicorn launcher
├── models/
│   ├── eye_cnn.keras       # Trained binary eye state classifier (1.14M params)
│   ├── best_vigidriver_lstm.keras  # Trained 30-frame vigilance Bi-LSTM (30.6K params)
│   └── face_landmarker.task # MediaPipe 468-point face landmarker model
├── src/                    # React 18 frontend (Vite)
│   ├── components/         # LiveMonitoringDashboard, AlertScreen, Analytics, etc.
│   ├── lib/                # API client, WebAudio alert synthesizers, UI components
│   └── assets/             # Branding and hero visual assets
├── dist/                   # Production-compiled frontend bundle (served by FastAPI)
├── tests/                  # 5-Phase automated verification test suite (54/54 PASS)
├── Dockerfile              # Production Dockerfile (Hugging Face Spaces, Render, Railway)
├── render.yaml             # Render 1-click blueprint configuration
├── DEPLOYMENT.md           # Step-by-step public cloud deployment guide
├── start_vigidriver.bat    # Windows 1-click batch launcher
├── start_vigidriver.ps1    # PowerShell launcher
└── requirements.txt        # Pinned Python production dependencies
```

---

## 🚀 Quickstart Guide

### Option 1: One-Click Desktop Launch (Windows)
Double-click [`start_vigidriver.bat`](file:///g:/VigiDriver_main%20-%20Copy/start_vigidriver.bat) or run the PowerShell script:
```powershell
.\start_vigidriver.ps1
```
The server will start and automatically serve the complete full-stack application at **`http://127.0.0.1:8000`**.

---

### Option 2: Manual CLI Startup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/subikshabamboo/VigiDriver.git
   cd VigiDriver
   ```

2. **Create and activate a virtual environment (Python 3.11 recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install production dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the unified full-stack server:**
   ```bash
   python backend/run.py
   ```
   Open **`http://127.0.0.1:8000`** in your browser. The single FastAPI process serves both the backend API and the React frontend.

---

### Option 3: Frontend Development Mode (Optional)
If you wish to edit React components with hot-module replacement (HMR):
```bash
npm install
npm run dev
```
The Vite development server will run at `http://localhost:5173` and proxy API calls to the backend on port 8000.

---

## 🌐 Public Cloud Deployment

VigiDriver includes full Docker containerization and remote browser webcam streaming, enabling deployment to free cloud tiers so evaluators can test the application from anywhere.

* **Hugging Face Spaces (Recommended — Free 16 GB RAM & 2 vCPUs):**
  Create a new Docker Space at [huggingface.co/new-space](https://huggingface.co/new-space) and push this repository. Hugging Face automatically builds the included `Dockerfile` and provides a permanent public HTTPS URL.
* **Render.com:**
  Connect your GitHub repository; Render detects `render.yaml` and deploys the Docker web service.
* **Instant Cloudflare Tunnel:**
  Share your local running server with remote evaluators in 10 seconds:
  ```powershell
  cloudflared tunnel --url http://127.0.0.1:8000
  ```

*For complete step-by-step cloud setup instructions, refer to [`DEPLOYMENT.md`](file:///g:/VigiDriver_main%20-%20Copy/DEPLOYMENT.md).*

---

## 🧪 Verification & Test Suite

VigiDriver includes comprehensive automated test coverage across five distinct verification phases:

```bash
# Run static model and robustness unit tests
pytest tests/test_phase1_static_models.py -v

# Run live API integration tests
python tests/test_phase2_api.py

# Run temporal Bi-LSTM, Grad-CAM, and K-Means gating tests
python tests/test_phase4_temporal_xai.py
python tests/test_phase5_lstm_gating.py
```

### Verification Results:
- **Phase 1 (Static Models & Invariants):** 5/5 PASSED (20 unit assertions)
- **Phase 2 (Live API & Session Lifecycle):** 20/20 PASSED
- **Phase 4 (Temporal Vigilance & Grad-CAM):** 10/10 PASSED
- **Phase 5 (Self-Gated Reliability Fusion):** 7/7 PASSED
- **Total:** **54/54 Tests 100% PASS**

---

## 📜 License

This project is open-source and licensed under the **[MIT License](LICENSE)**.
