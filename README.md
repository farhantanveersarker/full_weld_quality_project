# Full Weld Quality Inspection System

**Final-Year B.Tech Project | Artificial Intelligence & Machine Learning**

An automated Weld Quality Inspection system that uses Deep Learning and Computer Vision to detect, classify, and localize welding defects in industrial weld images.

The project uses YOLOv8 object detection models and a custom annotated dataset to identify nine categories of weld conditions, including cracks, porosity, undercut, spatter, incomplete fusion, and other welding imperfections.

**Project Domain:** Artificial Intelligence, Machine Learning, Deep Learning, and Computer Vision.

**Primary Objective:** To develop an intelligent inspection system that supports automated welding defect identification and improves the efficiency and consistency of weld quality assessment.

## Defect Classes

| ID | Class |
|----|-------|
| 0 | Burn_through |
| 1 | Crack |
| 2 | Excess_Reinforcement |
| 3 | Good_Weld |
| 4 | Incomplete_Fusion |
| 5 | Overlap |
| 6 | Porosity |
| 7 | Spatter |
| 8 | Undercut |

## Project Structure

```
.
├── data.yaml                  # YOLO dataset config (paths relative to repo root)
├── dataset/
│   ├── weld-train/            # Training split  (images/ + labels/)
│   ├── weld-validation/       # Validation split (images/ + labels/)
│   ├── weld-test/             # Test split      (images/ + labels/)
│   ├── weld-*_smoke/          # Small subsets used for quick smoke-test runs
│   └── archives/              # Original .zip exports of each split
├── models/
│   ├── best_model.pt          # Best trained weld-defect detection model
│   └── yolov8n.pt             # Pretrained YOLOv8n base weights
├── notebooks/
│   ├── train.ipynb            # Training & evaluation pipeline (local / smoke test)
│   └── train_gpu.ipynb        # Full training run on GPU
└── runs/                      # Training outputs (metrics, plots, weights)
```

## Dataset

The dataset is included in the [`dataset/`](dataset/) folder. It is also available on Google Drive: [Weld Quality Dataset](https://drive.google.com/drive/folders/1Cvhg9frmN2msbZ_YIT9IlfGD-XbSoYki?usp=sharing)

## Getting Started

```bash
pip install ultralytics torch torchvision opencv-python-headless matplotlib pandas seaborn pyyaml pillow
```

Open a notebook from the `notebooks/` folder and set `RUN_MODE` in the first config cell:

- `"smoke_test"` — quick pipeline check on a small sample
- `"full"` — full training on the entire dataset

Inference with the trained model:

```python
from ultralytics import YOLO

model = YOLO("models/best_model.pt")
results = model.predict("dataset/weld-test/images", save=True)
```

## Author

[farhantanveersarker](https://github.com/farhantanveersarker)
