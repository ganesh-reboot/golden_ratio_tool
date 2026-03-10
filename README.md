# InsightFace Batch Exporter 🧠✨
*GoldenFace-style scoring, robust face landmarks, head pose & quality metrics — at scale.*

---

> **What is this?**  
> A single-file CLI that scans a folder of people/images, runs InsightFace (with optional MediaPipe + YOLO fallbacks), computes **GoldenFace-style geometry scores**, head pose, and quality metrics, and writes:
>
> - Per-image CSV (`if_master.csv`)  
> - **Consolidated per-person CSV** (`if_summary_by_person.csv`)  
> - Optional annotated overlays (JPG)  
> - Optional landmark JSON dumps
>
> Built for **non-frontal, in-the-wild** photos — and resilient enough for stylized/cartoon faces (optional anime YOLO fallback).

---

## 🗂️ Folder Layout

```
root/
  Person A/
    a1.jpg
    a2.png
  Person B/
    b1.jpg
# (or: images directly under root)
```

---

## 🔥 Highlights

- **Smart per-image detection** — try multiple detector input sizes and pick the **first** valid or **best** result (`--det-policy first-hit|best-hit`).
- **Landmarks, tiered** — InsightFace **3D-68** (preferred) → **2D-106** fallback → **MediaPipe 468** fallback → (optional) **YOLO anime** to crop cartoons then MediaPipe.
- **GoldenFace-style metrics** — `gf_tgsm_deflection`, `gf_vfm_deflection`, `gf_tzm_deflection`, `gf_tsm_deflection`, `gf_lc_deflection`, plus **overall** `gf_geometric_ratio_score` (0–100).
- **Head pose** — yaw / pitch / roll via PnP (coarse but useful) + optional **frontality filter**.
- **Quality & geometry** — face area %, Laplacian sharpness, brightness; length/width, thirds, φ-closeness, canthal tilt, jaw angle, symmetry error, shape guess.
- **Consolidation** — multi-image → **one record per person** (medians/means), with auto-chosen **best image**.

---

## 📦 Installation

**Python:** 3.8–3.12

Base (GPU):
```bash
pip install insightface onnxruntime-gpu opencv-python tqdm numpy
```

CPU (no CUDA/cuDNN):
```bash
pip install insightface onnxruntime opencv-python tqdm numpy
```

**Optional fallbacks**
```bash
# MediaPipe Face Mesh (468 landmarks)
pip install mediapipe==0.10.14

# Ultralytics YOLO (for anime/cartoon crop)
pip install ultralytics
```

**CUDA note (Windows):** when configured correctly you should see  
`Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']`  
Run with `--ctx-id 0` (GPU0). Use `--ctx-id -1` to force CPU.

---

## 🚀 Quick Start

GPU (PowerShell / CMD on Windows):
```powershell
python .\insightface_master.py .\root `
  --outdir .\bald_out `
  --det-sizes 480 `
  --det-policy best-hit `
  --ctx-id 0 `
  --save-overlays --save-landmarks --print-found
```

CPU:
```powershell
python .\insightface_master.py .\root `
  --outdir .\bald_out `
  --det-sizes 480,640 `
  --ctx-id -1 `
  --save-overlays
```

Auto sizes, faster pass:
```powershell
python .\insightface_master.py .\root `
  --outdir .\bald_out `
  --det-sizes auto `
  --det-policy first-hit `
  --ctx-id 0 `
  --save-overlays --save-landmarks --print-found
```

Prefer only 3D-68 landmarks:
```powershell
python .\insightface_master.py .\root `
  --outdir .\bald_out `
  --landmarks 68 `
  --det-sizes 480 `
  --ctx-id 0
```

Frontality filter (keep rejections in CSV):
```powershell
python .\insightface_master.py .\root `
  --outdir .\bald_out `
  --det-sizes 480,640 `
  --frontality-filter --yaw-max 40 --pitch-max 30 --roll-max 30 `
  --keep-rejected `
  --ctx-id 0
```

Anime/cartoon fallback (YOLO → MediaPipe):
```powershell
python .\insightface_master.py .\root `
  --outdir .\bald_out `
  --det-sizes 384,480,640 `
  --anime-weights C:\path\to\anime_face.pt `
  --ctx-id 0 `
  --save-overlays
```

---

## ⚙️ CLI Reference

```
python insightface_master.py ROOT
  [--out IF_MASTER.csv]                   # per-image CSV (in --outdir)
  [--summary IF_SUMMARY_BY_PERSON.csv]    # per-person CSV (in --outdir)
  [--outdir OUTDIR]                       # output dir (artifacts here)
  [--landmarks {68,106,auto}]             # 68 preferred; 106 fallback (auto=default)
  [--det-sizes {auto|comma_list}]         # e.g. 384,480,640,800 or auto
  [--det-policy {first-hit|best-hit}]     # try sizes per image
  [--ctx-id INT]                          # 0=GPU0, -1=CPU
  [--save-overlays] [--save-landmarks]    # write JPG overlays + JSON landmarks
  [--print-found]                         # list people/images found
  [--frontality-filter]                   # drop by yaw/pitch/roll
  [--yaw-max DEG --pitch-max DEG --roll-max DEG]
  [--min-det-score FLOAT]                 # default 0.45
  [--keep-rejected]                       # keep filtered rows with reason
  [--anime-weights PATH_TO_YOLO_PT]       # enable anime/cartoon fallback
```

**Detector sizes**
- `--det-sizes auto` → derives sizes from each image  
- `--det-sizes 384,480,640,800` → explicit ladder (great for stylized faces)  
- `--det-policy best-hit` (default) tries all sizes and picks the best  
- `--det-policy first-hit` stops at the first valid result (faster)

---

## 📑 Outputs

### A) Per-image CSV → `OUTDIR/if_master.csv`
- Identity: `person`, `image`
- Detection: `det_score`, `bbox`, `det_size_used`
- Pose: `pose_yaw_deg`, `pose_pitch_deg`, `pose_roll_deg`, `is_frontal`
- Ratios (flattened as `ratios.*`): `length_to_width`, thirds, φ-closeness, IPD norms, etc.
- Geometry: `jaw_angle_deg`, `canthal_tilt_deg`, `symmetry_error_px`
- **GoldenFace-style**:  
  `gf_tgsm_deflection`, `gf_vfm_deflection`, `gf_tzm_deflection`,  
  `gf_tsm_deflection`, `gf_lc_deflection`,  
  `gf_geometric_ratio_score` (**0–100**, higher = closer to “golden”)
- Quality: `quality_face_area_pct`, `quality_sharpness_lap`, `quality_brightness_mean`
- Artifacts: `overlay_path`, `landmarks_json`, plus `*_error` columns
- Filtering: `filter_reason`, `error`

### B) Per-person CSV → `OUTDIR/if_summary_by_person.csv`
- Counts: `n_images`, `n_used`, `n_errors`, `n_rejected`, `frontal_ratio`
- **Best image**: `best_image`, `best_overlay`, `best_det_size_used`
- Medians/means: pose, face area %, sharpness, brightness, jaw angle, canthal tilt, symmetry error
- Mode: `shape_guess_mode`
- **Golden**: `gf_geometric_ratio_score_0_100` (median across person images)
- Ratio medians: `median_ratios.*`

### C) Artifacts
- Overlays → `OUTDIR/overlays/*_overlay.jpg`  
- Landmarks JSON → `OUTDIR/landmarks/*_landmarks.json`

---

## 🧮 GoldenFace-Style Score (How it’s computed)

We follow the spirit of the original “deflection” metrics:

- **TGSM** — Trichion–Glabella–Subnasale–Menton thirds evenness  
- **VFM** — Five vertical bands across face width  
- **TZM** — Trichion→Menton vs zygoma width ≈ φ  
- **TSM** — (Trichion→Subnasale) vs (Subnasale→Menton) ≈ φ  
- **LC** — Lateral canthus span vs mouth width ≈ 2.30  

Each produces a **deflection %** (lower = better). Overall:

```
gf_geometric_ratio_score = 100 - average(deflections)
```

These are geometry-only heuristics; they do **not** claim to measure attractiveness.

---

## 🧪 Tips

- For stylized/oblique shots, **smaller det-size (e.g., 480)** often detects more faces.
- Use `--det-policy best-hit` for maximum robustness per image.
- Leave the **frontality filter OFF** for “in-the-wild” content; sort by yaw/pitch/roll later.
- For cartoons/anime, enable `--anime-weights` so we crop face-like regions before MediaPipe fallback.

---

## 🛠️ Troubleshooting

- **CPU only shows up** — You’ll see `['CPUExecutionProvider']`. Install matching CUDA + cuDNN and use `onnxruntime-gpu`, then run with `--ctx-id 0`.
- **cublas / cudnn DLL missing (Windows)** — Ensure CUDA Toolkit and cuDNN match your onnxruntime-gpu build. Add CUDA `bin` directories to `PATH` and restart your shell.
- **MediaPipe not found** — Install `mediapipe==0.10.14` (Python 3.8–3.12). If still failing, omit it—InsightFace alone will run, just with fewer fallbacks.
- **YOLO weights** — Supply a valid `.pt` file via `--anime-weights` (Ultralytics v8). Without it, anime/cartoon fallback is disabled.

---

## 🔗 References (plain URLs)

InsightFace: https://github.com/deepinsight/insightface  
ONNX Runtime: https://onnxruntime.ai/  
MediaPipe Face Mesh: https://developers.google.com/mediapipe/solutions/vision/face_mesh  
Ultralytics YOLO: https://github.com/ultralytics/ultralytics  
GoldenFace (original idea): https://github.com/Aksoylu/GoldenFace
