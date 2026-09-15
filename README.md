# Hierarchical Factor Graph Optimization Calibration Tool

A camera extrinsic calibration system based on Hierarchical Factor Graph
Optimization (FGO) with a PySide6 GUI.  
Optimization runs in three stages — environment markers, wheel markers, then
vehicle cameras — and each stage weights its factors with MARSCOT uncertainty.

---

## 📁 Project Structure

```
hierarchical_fgo/
│
├── calibration_tool.py              # Main GUI entry point (PySide6)
├── fgo_system.py                    # Core factor graph optimization logic
├── pose_covariance_calculator.py    # MARSCOT pose uncertainty
├── Harris_matrix_extractor.py       # Harris matrix extraction
├── main.py                          # CLI entry point
├── requirements.txt                 # Python package dependencies
├── ARCHITECTURE.md                  # System architecture notes
├── GUIDE.md                         # Detailed usage guide
│
└── src/
    ├── config.py                    # All configuration (CalibrationConfig)
    └── calibration.py               # Calibration runner (CalibrationSystem)
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Installing GTSAM (conda recommended):**
> ```bash
> conda install -c conda-forge gtsam
> ```

### 2. Run the GUI

```bash
cd hierarchical_fgo
python calibration_tool.py
```

### 3. Run from the CLI

```bash
cd hierarchical_fgo
python main.py
```

---

## 🔄 Three-Stage Optimization Pipeline

```
[Stage 1] Environment Marker Optimization
    ↓  AprilTag images captured by the environment camera
    ↓  Marker detection → MARSCOT uncertainty → factor graph construction
    ↓  min_pair_obs filter drops rarely co-observed marker pairs, then LM
    ↓  Result: absolute poses of the marker nodes (saved to optimization_results.pkl)
    ↓  Auto popup: per-marker 6DoF pose table (relative to the reference marker)

[Stage 2] Wheel Marker Optimization
    ↓  Vehicle marker images captured by the wheel camera
    ↓  Stage 1 result plus the wheel marker observation factors
    ↓  min_pair_obs filter, then LM
    ↓  Result: combined environment + wheel marker poses
    ↓  Auto popup: per-marker 6DoF pose table (relative to the reference marker)

[Stage 3] Vehicle Camera Optimization
    ↓  Images captured by the vehicle cameras (fisheye / pinhole)
    ↓  Vehicle camera nodes added on top of the frozen Stage 2 result
    ↓  geo_skip consistency filter drops outlier observations, then LM
    ↓  Result: camera extrinsics in the vehicle frame (6DoF)
    ↓  Export: [x, y, z, roll, pitch, yaw] plus the 4x4 transformation matrix
```

---

## 📋 GUI Workflow

| Step | Item | Description |
|------|------|-------------|
| 1 | **Environment Image Directory** | Select the folder of environment marker (AprilTag) images |
| 2 | **Wheel Image Directory** | Select the folder of vehicle wheel marker images |
| 3 | **Camera Parameters** | Enter fx, fy, cx, cy, k1, k2, k3, p1, p2 |
| 4 | **🎯 Marker Settings** | Marker size, reference marker ID, vehicle marker board IDs, excluded board IDs, tag family, board grid (rows x cols, gap) — in the Marker & AprilTag Configuration dialog |
| 5 | **📊 Uncertainty Settings** | See [Configuration Reference](#-configuration-reference) below |
| 6 | **🌍 Environment** | Run Stage 1; the marker pose window pops up when it finishes |
| 7 | **🚗 Extrinsic** | Load Stage 1, then run Stage 2 and Stage 3 |
| 8 | **Export Poses** | Save the vehicle camera poses (.txt / .json / .yaml — falls back to JSON when PyYAML is absent) |

---

## ⚙ Configuration Reference

Every observation's 6x6 pose covariance is computed with MARSCOT (unscented
transform through PnP) and handed to GTSAM unchanged. There is no alternative
covariance model to select, so the **📊 Uncertainty Settings** dialog exposes
only the observation filters below.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `geo_skip_rot_deg` | Stage 3 rotation threshold for vehicle observation consistency [deg] | `4.0` |
| `geo_skip_pos_m` | Stage 3 position threshold for vehicle observation consistency [m] | `0.1` |
| `min_pair_obs` | Minimum co-observation count for a marker pair — pairs below it are removed, `1` disables the filter (Stages 1 and 2) | `2` |

---

## 📤 Output

### Marker pose auto popup (after Stage 1 / Stage 2)
- Shows the 6DoF poses relative to the reference marker
- Drag-select the numeric area and press Ctrl+C to copy
- A bulk copy button produces Excel-compatible tab-separated text

### Export Poses (vehicle cameras)
Choose the output format: `.txt` / `.json` / `.yaml`

```
[x, y, z, roll, pitch, yaw]: [0.5231, 0.0124, -0.2345, 0.376, -1.044, 0.071]  (m, deg)
4x4 Transformation Matrix:
  [ 0.999832,  0.001234, -0.018234,  0.523100]
  [-0.001452,  0.999978, -0.006543,  0.012400]
  ...
```

---

## 🐛 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ImportError` | Check that you launched from inside the `hierarchical_fgo/` folder |
| `gtsam` install failure | `conda install -c conda-forge gtsam` |
| No markers detected | Check the image paths, marker size, and marker ID settings |
| Stage 1 popup does not appear | Wait a few seconds after the optimization finishes (it runs on a worker thread) |
| Vehicle camera parameter error | Set the intrinsics for every camera in the Camera Parameters dialog |

---

## 📝 Notes

1. **Working directory**: always launch from inside the `hierarchical_fgo/` folder.
2. **Stage order**: run Stage 1 first, then Stage 2, then Stage 3. The order cannot be changed.
3. **Vehicle camera parameters**: before running Stage 3, set the intrinsics of every camera in the Camera Parameters dialog.
4. **Saved results**: `optimization_results.pkl` is written automatically when Stage 1 finishes, and Stages 2 and 3 build on it.
