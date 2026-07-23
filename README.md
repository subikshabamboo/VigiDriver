# VigiDriver: A Spatiotemporal Edge-AI Framework for Predictive Driver Monitoring Systems (DMS)

**VigiDriver** is a production-grade, end-to-end Driver Monitoring System (DMS) designed to transition drowsiness detection from *reactive* alerts (triggered post-sleep) to *predictive* spatiotemporal analysis. Built with a heavy emphasis on edge-readiness and extreme environment resilience, VigiDriver addresses the "reality gap" between clean lab datasets and chaotic real-world vehicle cabins (e.g., night driving, harsh dashboard reflections, and low-light sensor grain).

---

## 🏗️ System Architecture & Workflow

```text
[ Raw UTA-RLDD Videos ] 
        │
        ▼
[ Subject-Level Isolation ] ──► (Guarantees Zero Data Leakage Across Splits)
        │
        ▼
[ On-the-Fly Sliding Window Engine ] (30-frame temporal clips, stride 15)
        │
        ▼
[ Physics-Based Augmentation Engine ] 
    ├── YUV-CLAHE (Adaptive Shadow Enhancement)
    ├── Non-Uniform Dashboard Glow (Instrument Cluster Light Physics)
    ├── Spatiotemporal Sweeping Glare (Oncoming Headlights / Streetlamps)
    └── Poisson-Gaussian Sensor Noise (Real CMOS Hardware Grain)
        │
        ▼
[ Hybrid Spatial-Temporal Model ] (CNN Backbone + LSTM Temporal Memory + Attention)
        │
        ▼
[ Edge-Ready Inference / ONNX Export ]

```

---

## ✨ Key Technical Highlights

1. **Physically-Grounded Domain Randomization (`augmentation.py`):**
* **YUV-CLAHE:** Enhances facial features hidden in deep shadows by applying contrast adjustment strictly to the luminance ($Y$) channel, preserving natural human skin tones.
* **Dashboard Glow:** Simulates a vertical upward-facing light gradient mimicking an instrument cluster reflecting onto the driver's lower face.
* **Sweeping Glare:** Projects a moving circular light source across frames dynamically based on video progress to replicate passing streetlamps or oncoming traffic.
* **Poisson-Gaussian Noise:** Models heteroscedastic CMOS sensor physics where photon shot noise scales with signal brightness alongside thermal read noise.


2. **Identity-Safe Data Governance (`dataset.py`):**
* Enforces strict **Subject-Level Cross-Validation**. Subjects in the training split never cross over into validation or testing sets, ensuring the model learns the physiological universalities of fatigue rather than memorizing individual driver faces.


3. **Lazy-Loading Stream Architecture:**
* Eliminates RAM bottlenecks and storage inflation by using an on-the-fly indexing map and OpenCV stream seeking (`cv2.CAP_PROP_POS_FRAMES`) to extract temporal clips dynamically during training batches.



---

## 📂 Repository Structure

```text
vigi_driver/
│
├── augmentation.py      # The physical lighting & noise novelty engine
├── dataset.py           # Subject-safe splitting & sliding window generator
├── test_pipeline.py     # Pipeline validation & Tensor shape verification script
└── README.md            # Project documentation

```

---

## 🚀 Getting Started & Installation

### Prerequisites

Ensure you have Python 3.10+ and PyTorch installed. If you are running inside a **Kaggle Notebook**, you can utilize GPU acceleration (T4 x2) for optimal performance.

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib

```

### Running the Pipeline Verification

To verify your data pipeline, indexation, and physical augmentation modules, run the test runner script:

```bash
python test_pipeline.py

```

**Expected Console Output:**

```text
Initializing Data Pipeline Test on Kaggle...
Found subjects: ['05', '06', '02', '04', '03', '01']
Assigned Train Subjects: ['05', '06', '02', '04']
Dataset initialized! Total sliding windows found: 7045
Fetching a sample batch through the pipeline...

--- Kaggle Pipeline Verification Successful ---
Batch Window Tensor Shape : torch.Size([4, 30, 3, 224, 224])
Batch Labels Shape        : torch.Size([4])
Data Type                 : torch.float32
Label Values              : tensor([0, 1, 1, 0])

```

---

## 🔬 Visualizing the Novelty Pipeline

To inspect the physical transformations applied by the augmentation engine, run the visualization utility:

```bash
python visualize_novelty.py

```

This generates a side-by-side comparison (`novelty_comparison.png`) demonstrating the shift from raw standard video frames to the robust, noise-injected, shadow-enhanced training frames.

---

## 📈 Roadmap & Future Extensions

* [x] Physics-based augmentation and domain randomization engine.
* [x] Subject-isolated streaming sliding-window dataset framework.
* [ ] Integration of MobileNetV2 spatial backbone + LSTM temporal memory.
* [ ] Temporal Attention mechanism for explainable driver state logging.
* [ ] ONNX export pipeline for low-latency edge deployment.

---

## 📜 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
