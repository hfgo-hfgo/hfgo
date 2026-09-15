# Hierarchical FGO System Architecture

Internal structure of the camera extrinsic calibration system built on
hierarchical factor graph optimization (FGO). For usage see
[README.md](README.md); for headless (programmatic) use and the configuration
reference see [GUIDE.md](GUIDE.md).

## 📁 Directory Structure

```
Hierarchical_fgo/
├── calibration_tool.py              # PySide6 GUI (main entry point)
├── main.py                          # CLI entry point (launches the GUI)
├── fgo_system.py                    # FGO engine: detection, factor graph, visualization
├── pose_covariance_calculator.py    # MARSCOT pose uncertainty (unscented transform)
├── Harris_matrix_extractor.py       # Corner covariance (structure tensor)
│
├── src/
│   ├── config.py                    # CalibrationConfig (every setting)
│   └── calibration.py               # CalibrationSystem (three-stage runner)
│
├── visualization_results/           # Where visualizations are written (optional)
└── optimization_results.pkl         # Stage 1 result (generated automatically)
```

Image paths are not stored in the repository. They are supplied through
`CalibrationConfig` as `image_directory` / `wheel_directory` and
`vehicle_camera_configs[i]['images']`.

## 🎯 ID / Symbol Scheme

GTSAM symbols are of two kinds: markers `M` and cameras `C`.

| Range | Kind | Assigned in |
|-------|------|-------------|
| `C100 +` | Environment cameras (one image = one camera) | `fgo_system.py` (`100 + idx`) |
| `C10000 +` | Wheel cameras | Stage 2 incremental optimization |
| `C20000 +` | Vehicle cameras (as many as are configured) | Stage 3 (`20000 + cam_idx`) |
| `M<board_id>` | Marker boards | `board_id = min(tag_id on the board)` |

`board_id` is the **representative ID of a board**, not an individual AprilTag
ID. Environment boards compute it as
`env_first_tag_id + group_index x (rows x cols)`, while wheel boards use the
values listed in `vehicle_markers` as they are.

## 📊 Execution Flow (three stages)

```
Stage 1 - Environment marker optimization    CalibrationSystem.run_optimization()
  |-- Detect env images -> merge per board -> MARSCOT uncertainty
  |-- Collect marker pairs (M2M) -> min_pair_obs filter -> BFS initialization
  |-- GTSAM Levenberg-Marquardt optimization
  |-- Save optimization_results.pkl   <- only the Stage 1 result is saved
  +-- Queue the marker 6DoF pose table for the GUI

Stage 2 - Wheel marker optimization          run_wheel_optimization()
  |-- Add wheel image observations on top of the frozen Stage 1 result
  |-- Apply the min_pair_obs filter to the wheel marker pairs
  +-- Combined environment + wheel marker result

Stage 3 - Vehicle camera optimization        run_vehicle_camera_optimization()
  |-- Initialize each camera pose from one image (lowest geometric residual candidate)
  |-- geo_skip filter: drop observations inconsistent with the frozen marker map
  |-- Carry the Stage 1 and 2 camera poses over into the result after optimization
  +-- Extrinsics in the vehicle frame (6DoF plus 4x4 matrix)
```

Stage 2 and Stage 3 results are not written to disk. Use the GUI's Export to
keep them.

## 🔌 Module Dependencies

```
calibration_tool.py (GUI)  /  main.py (CLI -> GUI)
        |
        +-- src/calibration.py  CalibrationSystem
                  |-- src/config.py            CalibrationConfig
                  +-- fgo_system.py            UncertaintyFactorGraph
                            +-- pose_covariance_calculator.py
                                      +-- Harris_matrix_extractor.py
```

The GUI runs the computation on a `CalibrationThread` (QThread) and creates
windows only on the main thread. Artifacts produced by the worker, such as the
pose tables, are handed to the main thread through a queue
(`pending_pose_tables`).

## 📝 Design Principles

1. **Hierarchical freezing** — each stage treats the previous stage's result as fixed.
2. **Configuration driven** — every parameter lives in `CalibrationConfig` and is
   passed explicitly, with no global variable fallbacks.
3. **Independent stage failure** — if Stage 2 or 3 fails, the earlier results
   survive and the failure is reported as a warning in the completion dialog.
4. **Observation-level filtering** — outliers are removed one observation at a
   time rather than by discarding a whole image (`min_pair_obs`, `geo_skip`).
5. **One covariance model** — every observation's 6x6 covariance comes from
   MARSCOT and reaches GTSAM unchanged, with no alternative to configure.
