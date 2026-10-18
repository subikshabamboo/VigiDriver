# VigiDriver — Complete Cloud & Remote Deployment Guide

This guide explains how to deploy **VigiDriver** so that evaluators, review committee members, peers, and external users can access and interact with the full-stack system from any browser or smartphone in the world.

---

## 🚀 Key Features for Remote Users

1. **💻 Remote Device Camera ("My Device Camera"):**
   - Cloud servers do not possess a physical webcam.
   - VigiDriver features native browser webcam streaming via `navigator.mediaDevices.getUserMedia()`.
   - Any remote user visiting the website can stream their laptop or phone camera directly to the backend neural pipeline.
2. **🚗 Simulation Stream (Synthetic Cabin Video):**
   - For users who cannot grant camera permissions or want an immediate automated demonstration.
   - Evaluates real neural models (`models/eye_cnn.keras`, MAR yawn tracker, Bi-LSTM temporal micro-sleep model, and K-Means gate) on realistic driver fatigue ramps.
3. **🔒 End-to-End HTTPS Compatibility:**
   - Modern browsers strictly require HTTPS to access webcams. All cloud platforms below provide free, automated SSL certificates.

---

## 🌟 Method 1: Hugging Face Spaces (Recommended — 100% Free, 16 GB RAM)

**Why this is the best choice for VigiDriver:**
- **16 GB RAM + 2 vCPU** (Free Tier): Prevents TensorFlow/Keras out-of-memory crashes that occur on smaller 512 MB cloud tiers.
- Permanent public HTTPS URL (e.g., `https://huggingface.co/spaces/<your-username>/vigidriver`).
- Native Docker container execution with automatic builds on push.

### Step-by-Step Instructions:

1. **Push your latest changes to GitHub:**
   ```bash
   git add .
   git commit -m "Configure production Docker container and browser camera support"
   git push origin main
   ```

2. **Create a Hugging Face Space:**
   - Log in or create a free account at [Hugging Face](https://huggingface.co/).
   - Click on your profile picture at the top right and click **New Space** (or visit [huggingface.co/new-space](https://huggingface.co/new-space)).
   - **Space Name:** `vigidriver` (or your preferred name).
   - **License:** `mit` or `apache-2.0`.
   - **Space SDK:** Select **Docker** -> **Blank**.
   - **Space Hardware:** Select **Free (2 vCPU, 16 GB RAM)**.
   - Click **Create Space**.

3. **Deploy the Code to your Space:**
   - **Option A (Via Git Remote):**
     Hugging Face provides a Git remote URL on your new Space page:
     ```bash
     git remote add space https://huggingface.co/spaces/<your-username>/vigidriver
     git push --force space main
     ```
   - **Option B (Direct GitHub Sync):**
     In your Space Settings -> **GitHub Actions / Repository Sync**, connect your GitHub repository `subikshabamboo/VigiDriver`. Every push to GitHub will automatically build and deploy.

4. **Verify Deployment:**
   - Hugging Face will automatically run `docker build` using the included `Dockerfile`.
   - Once the build shows `Running`, your app is live and usable worldwide at:
     `https://<your-username>-vigidriver.hf.space`

---

## 🛠️ Method 2: Render.com (1-Click Blueprint / Docker)

1. Sign up or log in at [Render.com](https://render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository `subikshabamboo/VigiDriver`.
4. Render automatically detects `render.yaml` and `Dockerfile`.
   - **Environment:** `Docker`
   - **Plan:** `Free`
   - **Health Check Path:** `/api/health`
5. Click **Create Web Service**.
6. Render builds the Docker image and provides a public HTTPS URL:
   `https://vigidriver.onrender.com`

---

## ⚡ Method 3: Instant 10-Second Live URL via Cloudflare Tunnel

If you want an evaluator or friend to test your running local application immediately without waiting for a cloud build:

1. Make sure your local VigiDriver server is running on `http://127.0.0.1:8000`:
   ```bash
   python backend/run.py
   ```
2. Open another terminal and run Cloudflare's quick tunnel (free, no sign-up needed):
   ```bash
   winget install --id Cloudflare.cloudflared
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
3. Cloudflare outputs a public HTTPS link (e.g. `https://random-words.trycloudflare.com`).
4. Share that link with anyone on their laptop or phone — they can open their camera and test the full AI pipeline instantly!

---

## 💻 Method 4: Local Single-Click Launchers

For running on your local machine:
- **Windows Batch Launcher:** Double-click `start_vigidriver.bat`
- **PowerShell Launcher:** Right-click `start_vigidriver.ps1` -> *Run with PowerShell*
- **Manual CLI:**
  ```bash
  python backend/run.py
  ```
  Then open `http://127.0.0.1:8000` in any web browser.

---

## 🧪 Verification Matrix

| Endpoint / Feature | Verification Method | Status |
| :--- | :--- | :--- |
| **GET /** | SPA catch-all serves pre-built React frontend bundle | ✅ Verified |
| **GET /api/health** | Verifies CNN, MAR, Bi-LSTM, K-Means models loaded | ✅ Verified |
| **POST /api/monitor/start** | Supports `use_browser_cam`, `use_simulation`, local camera | ✅ Verified |
| **POST /api/monitor/frame** | Ingests multipart JPEG frames from client webcams | ✅ Verified |
| **POST /api/monitor/stop** | Gracefully cleans up capture resources and DB state | ✅ Verified |
| **CORS Middleware** | Regex pattern matches all remote origins and domains | ✅ Verified |
| **Unit & Integration Tests**| 54/54 automated test assertions across all phases | ✅ 100% PASS |
