"""
Configuration module for Hierarchical Factor Graph Optimization
Manages all configuration values of the calibration system.
"""

# Matplotlib backend setup (prevents Qt GUI conflicts)
import matplotlib
matplotlib.use('Agg')  # non-GUI backend (file saving only)

import numpy as np
import os


class CalibrationConfig:
    """calibration configuration class"""

    def __init__(self):
        # ============================================================================
        # Basic directory settings
        # ============================================================================

        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file)  # hierarchical_fgo/src/
        hierarchical_fgo_dir = os.path.dirname(src_dir)  # hierarchical_fgo/
        self.project_root = os.path.dirname(hierarchical_fgo_dir)  # factor_graph_calibration/

        self.image_directory = 'data/env/'   # must be set from GUI before running
        self.wheel_directory = 'data/wheel/'   # must be set from GUI before running
        self.vehicle_directory = ''   # unused: images managed via vehicle_camera_configs
        self.image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']

        # ============================================================================
        # Marker settings
        # ============================================================================
        self.marker_size = 0.098765432  # half of marker size (for environment markers)
        self.marker_size2 = 0.061728395  # half of marker size (for vehicle wheel markers)

        # ============================================================================
        # AprilTag / AprilGrid settings (AprilTag 3 detector)
        # ============================================================================
        # Family: tagStandard41h12, tagStandard52h13, tag36h11, tag25h9, tag16h5,
        #         tagCircle21h7, tagCircle49h12, tagCustom48h12
        self.apriltag_family = 'tagStandard41h12'

        # AprilTag 3 detector tuning. quad_decimate downsamples only the quad
        # search; corners are refined at full resolution afterwards.
        self.quad_decimate = 2.0
        self.quad_sigma = 0.0
        self.refine_edges = 1
        self.decode_sharpening = 0.25
        self.nthreads = 0  # 0 = one thread per CPU core

        # Board layout: n×m AprilTags are printed on one physical marker board.
        # board_id = min(tag_ids_on_board) = first_tag_id + group_index * (rows*cols)
        # tag_gap is the printed gap between neighbouring tag edges, in meters.
        self.env_grid_rows    = 2
        self.env_grid_cols    = 2
        self.env_tag_gap      = 0.246914
        self.env_first_tag_id = 10   # min tag_id of first env board

        self.wheel_grid_rows    = 2
        self.wheel_grid_cols    = 2
        self.wheel_tag_gap      = 0.154321
        # NOTE: wheel board_id is looked up directly from vehicle_markers below
        # (no separate wheel_first_tag_id anchor).

        # marker size mapping by board ID
        self.marker_size_mapping = {}

        # World-origin board ID (GTSAM M1). board_id = min(tag_ids_on_board)
        self.first_marker_id = 10

        # Vehicle wheel board IDs [fl, rl, rr, fr]. board_id = min(tag_ids_on_board)
        self.vehicle_markers = [74, 78, 86, 82]

        # ============================================================================
        # Camera parameter settings
        # ============================================================================
        # Environment camera parameters [fx, fy, cx, cy]
        # Initial values shown in GUI — overwritten when user saves Camera Parameters dialog
        self.camera_params = [1455.736873, 1455.736873, 1008.0, 756.0]

        # Distortion coefficients [k1, k2, p1, p2, k3] (OpenCV order)
        self.dist_coeffs = np.array([np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)])

        # Wheel camera parameters
        # Initial values shown in GUI — overwritten when user saves Camera Parameters dialog
        self.wheel_camera_params = [1455.736873, 1455.736873, 1008.0, 756.0]
        self.wheel_dist_coeffs = np.array([np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)])

        # ============================================================================
        # Vehicle camera settings
        # ============================================================================
        # number of vehicle cameras (adjustable dynamically)
        self.num_vehicle_cameras = 7

        # vehicle camera settings (index-ordered)
        self.vehicle_camera_configs = [
            {
                'intrinsic': [924.3000013, 924.3000013, 960.0, 540.0],
                'camera_model': 'fisheye',
                'dist_coeffs': np.array([np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)]),
                'images': [
                    'data/vehicle_cam/front_wide_120fov.png'
                ]
            },
            {
                'intrinsic': [3688.700014, 3688.700014, 960.0, 540.0],
                'camera_model': 'pinhole',
                'dist_coeffs': np.array([np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)]),
                'images': [
                    'data/vehicle_cam/front_tele_30fov.png'
                ]
            },
            {
                'intrinsic': [924.2000005, 924.2000005, 960.0, 540.0],
                'camera_model': 'fisheye',
                'dist_coeffs': np.array([np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)]),
                'images': [
                    'data/vehicle_cam/cross_left_120fov.png'
                ]
            },
            {
                'intrinsic': [924.9999989, 924.9999989, 960.0, 540.0],
                'camera_model': 'fisheye',
                'dist_coeffs': np.array([np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)]),
                'images': [
                    'data/vehicle_cam/cross_right_120fov.png'
                ]
            },
            {
                'intrinsic': [1557.599996, 1557.599996, 960.0, 540.0],
                'camera_model': 'pinhole',
                'dist_coeffs': np.array([np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)]),
                'images': [
                    'data/vehicle_cam/rear_left_70fov.png'
                ]
            },
            {
                'intrinsic': [1555.700005, 1555.700005, 960.0, 540.0],
                'camera_model': 'pinhole',
                'dist_coeffs': np.array([np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)]),
                'images': [
                    'data/vehicle_cam/rear_right_70fov.png'
                ]
            },
            {
                'intrinsic': [3689.599973, 3689.599973, 960.0, 540.0],
                'camera_model': 'pinhole',
                'dist_coeffs': np.array([np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)]),
                'images': [
                    'data/vehicle_cam/rear_tele_30fov.png'
                ]
            }
        ]

        # ============================================================================
        # Observation filters
        # ============================================================================
        # Every observation's 6x6 pose covariance comes from MARSCOT (Unscented
        # Transform through PnP); there is no alternative model to select.

        # Geometric consistency thresholds (Stage 3: vehicle camera observations
        # inconsistent with the frozen marker map beyond these limits are
        # excluded from the factor graph)
        self.geo_skip_rot_deg  = 4.0
        self.geo_skip_pos_m    = 0.1

        # M2M pair minimum observation filter
        self.min_pair_obs = 0

        # AprilTag board IDs excluded from detection: any env/wheel image in
        # which one of these IDs is detected is skipped entirely (empty list =
        # no exclusion). Stage 3 vehicle images are NOT filtered by this.
        self.excluded_apriltag_ids = []

        # ============================================================================
        # Environment optimization settings
        # ============================================================================
        # Default for run_optimization(use_saved=None) — library/CLI use only;
        # the GUI always passes an explicit use_saved value.
        self.use_saved_results = False  # True: Load saved results, False: Run new
        # Stage 1 result file (relative to the working directory)
        self.save_file = 'optimization_results.pkl'

    def update_num_vehicle_cameras(self, num_cameras):
        """update number of vehicle cameras"""
        # allow 0 cameras (no images selected)
        if num_cameras < 0:
            num_cameras = 0

        current_num = len(self.vehicle_camera_configs)
        self.num_vehicle_cameras = num_cameras

        if num_cameras > current_num:
            for i in range(current_num, num_cameras):
                self.vehicle_camera_configs.append({
                    'intrinsic': [1.0, 1.0, 0.0, 0.0],
                    'camera_model': 'pinhole',
                    'dist_coeffs': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                    'images': []
                })
        elif num_cameras < current_num:
            # remove camera
            self.vehicle_camera_configs = self.vehicle_camera_configs[:num_cameras]

    def update_camera_params(self, fx, fy, cx, cy, k1=0, k2=0, k3=0, p1=0, p2=0):
        """update camera parameters (environment camera only)"""
        self.camera_params = [fx, fy, cx, cy]
        # OpenCV 5-element order: [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])
        # vehicle_camera_configs are managed independently per-camera via the dialog

    def update_image_directory(self, path):
        """update image directory path"""
        self.image_directory = path

    def update_wheel_directory(self, path):
        """update wheel image directory path"""
        self.wheel_directory = path
