# Hierarchical FGO Developer Guide

How to run the pipeline directly from code without the GUI, plus the
configuration reference. For the GUI workflow see [README.md](README.md); for
the internal structure see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Running

### GUI (recommended)

```bash
python main.py            # or: python calibration_tool.py
```

### Directly from code (headless)

`CalibrationSystem` exposes the three stages as they are. The return values are
shown below.

```python
from src.config import CalibrationConfig
from src.calibration import CalibrationSystem

cfg = CalibrationConfig()
cfg.image_directory = '/path/to/env/'
cfg.wheel_directory = '/path/to/wheel/'
cfg.camera_params   = [fx, fy, cx, cy]          # environment camera
cfg.vehicle_camera_configs[0]['images'] = ['/path/to/front.png']

calib = CalibrationSystem(cfg)

# Stage 1 - environment markers (writes optimization_results.pkl)
system, graph, initial, result, detections, uncertainty_info = \
    calib.run_optimization(use_saved=False)

# Stage 2 - wheel markers
wheel_graph, wheel_result, wheel_detections, wheel_camera_ids = \
    calib.run_wheel_optimization()

# Stage 3 - vehicle cameras
v_graph, v_result, v_detections, v_camera_ids = \
    calib.run_vehicle_camera_optimization()
```

On failure a stage returns `None` in place of its result, so the caller has to
check. `run_optimization(use_saved=True)` reads the Stage 1 result from
`save_file` instead of recomputing it.

Query the results through GTSAM symbols.

```python
import gtsam
markers = {gtsam.Symbol(k).index(): result.atPose3(k)
           for k in result.keys() if chr(gtsam.Symbol(k).chr()) == 'M'}
cameras = {gtsam.Symbol(k).index(): result.atPose3(k)
           for k in result.keys() if chr(gtsam.Symbol(k).chr()) == 'C'}
```

---

## Configuration Reference (`src/config.py`)

The values below are the current defaults. The GUI's Save Config regenerates
this file, so hand edits are overwritten with the GUI's current values whenever
Save Config runs.

### Paths and cameras

```python
cfg.image_directory   # environment image folder
cfg.wheel_directory   # wheel image folder
cfg.camera_params      = [fx, fy, cx, cy]        # environment camera
cfg.dist_coeffs        # OpenCV order [k1, k2, p1, p2, k3]
cfg.wheel_camera_params, cfg.wheel_dist_coeffs
cfg.vehicle_camera_configs   # per-camera dict: intrinsic / camera_model / dist_coeffs / images
                             # camera_model: 'pinhole' or 'fisheye' (Kannala-Brandt, 4 coefficients)
```

### Markers and board layout

```python
cfg.marker_size        = 0.098765432   # half the side of an environment tag [m]
cfg.marker_size2       = 0.061728395   # half the side of a wheel tag [m]
cfg.apriltag_family    = 'tagStandard41h12'
cfg.env_grid_rows      = 2             # tag array printed on one board
cfg.env_grid_cols      = 2
cfg.env_tag_gap        = 0.246914      # gap between the sides of adjacent tags [m]
cfg.env_first_tag_id   = 10            # * basis for board_id grouping
cfg.wheel_grid_rows    = 2
cfg.wheel_grid_cols    = 2
cfg.wheel_tag_gap      = 0.154321
cfg.first_marker_id    = 10            # board_id used as the world origin
cfg.vehicle_markers    = [74, 78, 86, 82]   # wheel board_ids [FL, RL, RR, FR]
```

`board_id = min(tag_id on the board)`. For environment boards it works out to
`env_first_tag_id + group_index x (rows x cols)` (for example, first tag 10 with
a 2x2 grid gives 10, 14, 18, and so on).

### Detector tuning

```python
cfg.quad_decimate     = 2.0   # only the quad search is downscaled; corners are refined at full resolution
cfg.quad_sigma        = 0.0
cfg.refine_edges      = 1
cfg.decode_sharpening = 0.25
cfg.nthreads          = 0     # 0 means one thread per CPU core
```

### Observation covariance

The covariance model is fixed: every observation's 6x6 pose covariance comes
from MARSCOT (unscented transform through PnP) and is passed to GTSAM as it is,
off-diagonal correlation terms included. There is nothing to configure here, and
no clamping is applied.

Corner covariances feeding that transform come from the structure tensor in
`Harris_matrix_extractor.py`; see [Extension Points](#extension-points) to
replace either step.

### Outlier observation filters

```python
cfg.min_pair_obs = 2          # drop marker pairs co-observed fewer times than this (1 disables the filter)
                              # applies to Stages 1 and 2 only
cfg.geo_skip_rot_deg = 4.0    # Stage 3: exclude vehicle observations that disagree with the
cfg.geo_skip_pos_m   = 0.1    #          frozen marker map by more than this
cfg.excluded_apriltag_ids = []  # env/wheel images in which one of these board_ids is detected are skipped entirely
                                # (not applied to Stage 3 vehicle images)
```

`min_pair_obs` decides pair by pair, and on a pair that trips the filter it
removes **only the observations of the less-observed marker**. If that leaves a
marker with no pairs at all, the marker drops out of the optimization.

### Result files

```python
cfg.use_saved_results = False              # default for run_optimization(use_saved=None)
cfg.save_file = "optimization_results.pkl" # Stage 1 result (relative to the working directory)
```

Only the Stage 1 result is written to a file. Stage 2 and Stage 3 results live
in memory, so use the GUI's Export Poses / Export Images to keep them.

---

## Extension Points

| What you want to do | Where to look |
|---|---|
| Add a new uncertainty computation | `_calculate_covariance_*` in `pose_covariance_calculator.py` |
| Swap the corner covariance model | `Harris_matrix_extractor.py` (including the `SIGMA_N` constant) |
| Change how the factor graph is built | `create_uncertainty_factor_graph` and `create_combined_factor_graph_with_base_result` in `fgo_system.py` |
| Add a visualization | `visualize_*` in `fgo_system.py`, then add an entry to the GUI combo box in `calibration_tool.py` |
| Add a configuration item | Add the attribute to `src/config.py` and add it identically to the Save Config template in `calibration_tool.py` |

`SIGMA_N = 1/6` is not a measured sensor noise but a constant that preserves the
existing scale. The factor graph only uses ratios between observations, so it
does not affect the optimization result; it matters only when the reported sigma
has to be read in physical units.

---

## Troubleshooting

| Symptom | What to check |
|---|---|
| `ImportError` | Whether you launched from the repository root (the folder holding `fgo_system.py`) |
| `gtsam` install failure | `conda install -c conda-forge gtsam` |
| No markers detected | `apriltag_family`, the tag size, and `env_first_tag_id` against the actual printed IDs |
| Boards grouped incorrectly | `env_grid_rows/cols` and `env_first_tag_id` (the basis of the board_id computation) |
| Wheel markers leaking into Stage 1 | Whether `min_pair_obs` was saved as 1, which disables the filter |
| Camera pose looks wrong in Stage 3 | Whether at least two boards are visible in a single vehicle image (one board cannot be validated) |
| Vehicle camera parameter error | Intrinsics have to be entered per camera in the Camera Parameters dialog |
| `Open3D View` disabled | Optional dependency — `pip install open3d` |
