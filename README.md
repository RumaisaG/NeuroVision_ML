# NeuroVision — ML Service

FastAPI microservice for Alzheimer's disease classification from brain MRI scans. A fine-tuned DenseNet-169 model trained on the ADNI dataset classifies scans into four clinical stages and returns Grad-CAM++ heatmaps for explainability. The training pipeline is fully documented in the included Jupyter notebook.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Dataset](#dataset)
- [Evaluation Results](#evaluation-results)
- [API Reference](#api-reference)
- [Getting Started — Local](#getting-started--local)
- [Deployment on Modal](#deployment-on-modal)
- [Checkpoint Format](#checkpoint-format)
- [Grad-CAM++ Explainability](#grad-cam-explainability)
- [Important Notes](#important-notes)

---

## Overview

The ML service is a standalone stateless FastAPI application. Each request:

1. Fetches the MRI image from its Supabase public URL
2. Applies a brain mask to suppress background noise
3. Normalises with ImageNet statistics
4. Runs inference through DenseNet-169
5. Generates a Grad-CAM++ heatmap over the relevant brain regions
6. Returns predicted class, confidence, four-class probabilities, and the heatmap as base64 PNG

The service is called by the Express backend's `mlService.js` after a scan is uploaded and analysis is triggered.

---

## Repository Structure

```
ml/
├── NeuroVision_ModelTraining.ipynb   # Complete training pipeline (Google Colab)
├── predict.py                        # FastAPI inference service
├── requirements.txt                  # Python dependencies
├── models/                          
│   ├── densenet169_fastapi.pt        # PyTorch checkpoint (produced by notebook)
│ 
└── README.md
```

---

## Model Architecture

```
Input
  RGB image → resize to 224×224

Backbone
  DenseNet-169 (pretrained on ImageNet)
  Classifier head replaced with nn.Identity()
  Output: 1664-dimensional feature vector

Custom Classification Head
  Dropout(0.4)
  Linear(1664 → 512) → ReLU → BatchNorm1D(512)
  Dropout(0.3)
  Linear(512 → 256) → ReLU
  Dropout(0.2)
  Linear(256 → 4)

Output
  Logits (4 classes) → Softmax → Probabilities
```

| Hyperparameter | Value |
|---|---|
| Backbone | DenseNet-169 (ImageNet pretrained) |
| Dropout | 0.4 (head), 0.3, 0.2 (deeper layers) |
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimiser | Adam (weight decay 1e-4) |
| Loss function | CrossEntropyLoss (label smoothing 0.1) |
| Scheduler | CosineAnnealingLR (T_max=40, eta_min=1e-6) |
| Max epochs | 40 |
| Early stopping | Patience = 5 |
| Seed | 42 |

---

## Training Pipeline

The complete training pipeline is in `NeuroVision_ModelTraining.ipynb`, designed to run on Google Colab with a T4 GPU. It is self-contained — mount your Google Drive, point `DATA_DIR` to your dataset, and run all cells.

### Two-phase training strategy

**Phase 1 — Warmup (5 epochs)**
The DenseNet-169 backbone is frozen. Only the custom classification head is trained. This prevents the randomly initialised head from corrupting the pretrained backbone weights early in training. Learning rate: `1e-4`.

**Phase 2 — Selective fine-tuning (up to 40 epochs)**
The backbone is selectively unfrozen — early low-level layers (`denseblock1`, `denseblock2`, `conv0`) remain frozen to preserve generic feature detectors, while deeper layers are updated. Learning rate is reduced to `2e-5` with cosine annealing. Early stopping monitors validation loss with patience of 5.

### Data augmentation (training only)

| Transform | Parameters |
|---|---|
| Resize | 240 × 240 (then centre crop to 224) |
| Random horizontal flip | p = 0.5 |
| Random affine | degrees ±5, translate 3%, scale 97–103% |
| Normalise | ImageNet mean/std |

Validation and test transforms: resize to 224 × 224 + normalise only.

### Notebook cells overview

| Cell | Purpose |
|---|---|
| 0 | Mount Google Drive |
| 1 | Install dependencies, import libraries |
| 2 | GPU detection and device setup |
| 3 | Reproducibility seed (`SEED=42`) |
| 4 | `Config` class — all hyperparameters in one place |
| 5 | Dataset exploration and class distribution plots |
| 6 | Sample MRI visualisation grid |
| 7 | `AlzheimerDataset` — custom PyTorch Dataset |
| 8 | `get_transforms()` — augmentation pipelines |
| 9 | `build_splits()` — stratified 70/15/15 split |
| 10 | `AlzheimerClassifier` — model definition |
| 11 | `EarlyStopping`, `train_epoch`, `validate_epoch` |
| 12 | `train_model()` — full two-phase training loop |
| 13 | Training curves plot (loss, accuracy, LR) |
| 14 | Load best checkpoint |
| 15 | `load_best_model()` — checkpoint restoration |
| 16 | Classification report + CSV export |
| 17 | Confusion matrix (raw + normalised) |
| 18 | ROC curves (per class + macro AUC) |
| 19 | Precision-recall curves |
| 20 | Per-class metrics bar chart |
| 21 | Confidence distribution (correct vs incorrect) |
| 22 | `print_summary_metrics()` — full evaluation table |
| 23 | `get_target_layer()` — Grad-CAM target layer |
| 24 | `export_for_fastapi()` — save inference checkpoint |
| 25 | Output summary — all saved files listed |

### Running the notebook

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `NeuroVision_ModelTraining.ipynb`
3. Runtime → Change runtime type → **T4 GPU**
4. Mount your Google Drive (Cell 0)
5. Place your dataset at:
   ```
   MyDrive/Colab Notebooks/Alzheimer/Dataset/
   ├── NonDemented/
   ├── VeryMildDemented/
   ├── MildDemented/
   └── ModerateDemented/
   ```
6. Run all cells (Runtime → Run all)

Training takes approximately **45–90 minutes** on a T4 GPU.

---

## Dataset

| Property | Value |
|---|---|
| Source | Link: `https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented` |
| Classes | 4 |
| Samples per class | 4,000 (balanced by random sampling) |
| Total training pool | 16,000 images |
| Train split | 70% (11,200 images) |
| Validation split | 15% (2,400 images) |
| Test split | 15% (2,400 images) |
| Image format | PNG / JPEG |

**Class labels:**

| Class | Clinical stage |
|---|---|
| `NonDemented` | No significant Alzheimer's indicators |
| `VeryMildDemented` | Early-stage neurodegeneration markers |
| `MildDemented` | Noticeable cognitive impairment |
| `ModerateDemented` | Significant cognitive decline |

The dataset is balanced at 4,000 samples per class using random sampling to prevent class imbalance bias during training.

---

## Evaluation Results

All metrics computed on the held-out test set (2,400 images, never seen during training or validation).

| Metric | Score |
|---|---|
| Test accuracy | reported in notebook output |
| Macro precision | reported in notebook output |
| Macro recall | reported in notebook output |
| Macro F1-score | reported in notebook output |
| Weighted F1-score | reported in notebook output |
| Macro AUC (OvR) | reported in notebook output |

> Run Cell 22 (`print_summary_metrics`) in the notebook to see the exact figures from your training run. Results vary slightly depending on random seed and GPU.

### Evaluation artefacts produced

All saved automatically to `cfg.OUTPUT_DIR`:

| File | Description |
|---|---|
| `class_distribution.png` | Dataset class balance (available vs sampled) |
| `sample_mri_scans.png` | 5 sample images per class |
| `training_curves.png` | Loss, accuracy, and LR across epochs |
| `confusion_matrix.png` | Raw counts + normalised confusion matrix |
| `roc_curves.png` | ROC curve per class + macro AUC |
| `pr_curves.png` | Precision-recall curves per class |
| `per_class_metrics.png` | Precision / recall / F1 bar chart |
| `confidence_distribution.png` | Confidence histogram (correct vs incorrect) |
| `gradcam_*.png` | Grad-CAM visualisation examples |
| `classification_report.csv` | Full metrics table (importable into report) |
| `densenet169_best.pt` | Best validation checkpoint |
| `densenet169_fastapi.pt` | FastAPI inference export |

---

## API Reference

Base URL (local): `http://localhost:8000`

### `POST /predict`

**Request body:**
```json
{
  "image_url":       "https://xxx.supabase.co/storage/v1/object/public/mri-scans/scans/brain.jpg",
  "scan_id":         "SC-2024-001",
  "gradcam_enabled": true
}
```

**Response:**
```json
{
  "predicted_class": "MildDemented",
  "confidence":      0.9174,
  "probabilities": {
    "NonDemented":      0.0312,
    "VeryMildDemented": 0.0421,
    "MildDemented":     0.9174,
    "ModerateDemented": 0.0093
  },
  "gradcam_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### `GET /health`

```json
{
  "status": "ok",
  "model":  "densenet169",
  "device": "cuda",
  "file":   "/model/densenet169_fastapi.pt",
  "exists": true
}
```

---

## Getting Started — Local

### Prerequisites

- Python **3.11** — PyTorch does not support Python 3.12+ yet
- Your trained `densenet169_fastapi.pt` checkpoint (produced by the notebook)

### Installation

```bash
cd ml

# Create venv with Python 3.11
C:\Python311\python.exe -m venv venv      # Windows
python3.11 -m venv venv                   # Linux / macOS

# Activate
venv\Scripts\activate                     # Windows
source venv/bin/activate                  # Linux / macOS

# Install PyTorch CPU (lighter for local dev, no GPU needed)
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install fastapi==0.111.1 uvicorn[standard]==0.30.1 \
  pillow==10.3.0 numpy==1.26.4 \
  opencv-python-headless==4.9.0.80 \
  pytorch-grad-cam==1.5.5 httpx==0.27.0
```

### Place your model file

```
ml/models/densenet169_fastapi.pt    ← produced by notebook Cell 24
```

### Run

```bash
uvicorn predict:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs: `http://localhost:8000/docs`

Test with curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://your-supabase-url/...", "scan_id":"test-001"}'
```

