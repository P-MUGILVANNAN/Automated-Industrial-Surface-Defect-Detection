# MVTec Anomaly Detection (MVTec AD) Dataset Guide

This project is built to work seamlessly with the official **MVTec Anomaly Detection Dataset**, the standard industrial computer vision benchmark for surface inspection.

Official Website & Download: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## 1. Supported Categories

MVTec AD comprises 15 industrial categories (5 textures, 10 objects):

### Texture Categories
- `carpet`
- `grid`
- `leather`
- `tile`
- `wood`

### Object Categories
- `bottle`
- `cable`
- `capsule`
- `hazelnut`
- `metal_nut`
- `pill`
- `screw`
- `toothbrush`
- `transistor`
- `zipper`

---

## 2. Directory Structure

Place or extract categories into `data/mvtec_anomaly_detection/` as follows:

```
data/mvtec_anomaly_detection/
├── bottle/
│   ├── train/
│   │   └── good/
│   │       ├── 000.png
│   │       └── 001.png
│   ├── test/
│   │   ├── good/
│   │   │   └── 000.png
│   │   ├── broken_large/
│   │   │   └── 000.png
│   │   ├── broken_small/
│   │   │   └── 000.png
│   │   └── contamination/
│   │       └── 000.png
│   └── ground_truth/
│       ├── broken_large/
│       │   └── 000_mask.png
│       ├── broken_small/
│       │   └── 000_mask.png
│       └── contamination/
│           └── 000_mask.png
└── ... (other categories)
```

---

## 3. Synthetic Demo Dataset Generation

If you have not downloaded the full multi-gigabyte MVTec AD archive yet, you can instantly generate a synthetically textured industrial dataset conforming strictly to the MVTec format:

```bash
python scripts/prepare_dataset.py --generate-demo --category bottle
```

This generates:
- High-fidelity normal industrial samples for training
- Defective test samples with ground-truth masks (e.g. scratches, stains, fractures)
- Normal test samples for false-alarm calibration and evaluation
