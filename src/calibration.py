"""
Calibration Module for Hierarchical Factor Graph Optimization
Module responsible for Factor Graph Optimization calibration logic

fgo_system.py is located in the hierarchical_fgo/ folder
"""

import sys
import os

# Add current directory (hierarchical_fgo) to sys.path
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)  # hierarchical_fgo/src/
hierarchical_fgo_dir = os.path.dirname(src_dir)  # hierarchical_fgo/
sys.path.insert(0, hierarchical_fgo_dir)

# Import all required functions and classes from FGO system
from fgo_system import (
    # Core functions
    create_uncertainty_system,
    run_uncertainty_factor_graph_optimization,
    build_apriltag_config,

    # Utility functions
    save_optimization_results,
    load_optimization_results,
)
import fgo_system as fgc  # For g_current_dir path constant



class CalibrationSystem:
    """
    Main class for managing Hierarchical Factor Graph Optimization system.
    Receives Config object and executes FGO optimization.
    All configuration values are passed explicitly to FGO functions - no global variable usage.
    """

    def __init__(self, config=None):
        """
        Args:
            config: CalibrationConfig object
        """
        self.config = config
        self.system = None
        self.graph = None
        self.initial_estimate = None
        self.result = None
        self.detections = None
        self.uncertainty_info = None

        # Store detection data and camera IDs by stage (explicit type distinction)
        self.environment_camera_ids = []  # Stage 1: Environment cameras (100+)
        self.wheel_camera_ids = []        # Stage 2: Wheel cameras (10000+)
        self.vehicle_camera_ids = []      # Stage 3: Vehicle cameras (20000+)
        self.vehicle_detections = None    # Stage 3 detection data

        # Every stage's graph: self.graph only keeps the latest one, but the
        # pose-scene views draw each stage's camera→marker edges.
        self.observation_graphs = []

    def run_optimization(self, use_saved=None):
        """
        Execute Hierarchical Factor Graph Optimization (Stage 1: Environment).

        Args:
            use_saved: Whether to use saved results (use config setting if None)

        Returns:
            tuple: (system, graph, initial_estimate, result, detections, uncertainty_info)
        """
        if use_saved is None:
            use_saved = self.config.use_saved_results if self.config else False

        print("\n" + "=" * 60)
        print("[config] Hierarchical FGO - observation covariance: MARSCOT (Unscented Transform)")
        print("=" * 60 + "\n")

        cfg = self.config

        # Validate environment camera parameters (fx must be > 1 — default placeholder is 1.0)
        if cfg.camera_params is None or cfg.camera_params[0] <= 1.0:
            print("❌ Environment camera parameters not set!")
            print("  Please open Camera Parameters dialog and enter fx, fy, cx, cy and distortion values.")
            return None, None, None, None, None, None

        if use_saved:
            print("\n" + "=" * 60)
            print("[stage1] Loading saved environment optimization results...")
            print("=" * 60)

            self.graph, self.initial_estimate, self.result, self.detections, self.uncertainty_info = \
                load_optimization_results(getattr(cfg, 'save_file', 'optimization_results.pkl'))

            if self.result is None:
                print("\n❌ Failed to load saved results!")
                print("  Please run 'Run Environment Calibration' first.")
                return None, None, None, None, None, None

            self.system = create_uncertainty_system(
                camera_params=cfg.camera_params,
                apriltag_config=build_apriltag_config(cfg),
                marker_size=cfg.marker_size,
                marker_size2=cfg.marker_size2,
                image_directory=cfg.image_directory,
                wheel_directory=cfg.wheel_directory,
                dist_coeffs=cfg.dist_coeffs,
                wheel_camera_params=cfg.wheel_camera_params,
                wheel_dist_coeffs=cfg.wheel_dist_coeffs,
                marker_size_mapping=cfg.marker_size_mapping,
                first_marker_id=cfg.first_marker_id,
                vehicle_markers=cfg.vehicle_markers,
                image_extensions=cfg.image_extensions,
                vehicle_camera_configs=cfg.vehicle_camera_configs,
                min_pair_obs=getattr(cfg, 'min_pair_obs', 2),
                excluded_apriltag_ids=getattr(cfg, 'excluded_apriltag_ids', None),
                enable_visualization=getattr(cfg, 'enable_visualization', True),
            )
        else:
            print("\n" + "=" * 60)
            print("[stage1] Running new optimization...")
            print("=" * 60)

            self.system, self.graph, self.initial_estimate, self.result, self.detections, self.uncertainty_info = \
                run_uncertainty_factor_graph_optimization(
                    camera_params=cfg.camera_params,
                    apriltag_config=build_apriltag_config(cfg),
                    marker_size=cfg.marker_size,
                    marker_size2=cfg.marker_size2,
                    image_directory=cfg.image_directory,
                    wheel_directory=cfg.wheel_directory,
                    dist_coeffs=cfg.dist_coeffs,
                    wheel_camera_params=cfg.wheel_camera_params,
                    wheel_dist_coeffs=cfg.wheel_dist_coeffs,
                    marker_size_mapping=cfg.marker_size_mapping,
                    first_marker_id=cfg.first_marker_id,
                    vehicle_markers=cfg.vehicle_markers,
                    image_extensions=cfg.image_extensions,
                    min_pair_obs=getattr(cfg, 'min_pair_obs', 2),
                    excluded_apriltag_ids=getattr(cfg, 'excluded_apriltag_ids', None),
                    enable_visualization=getattr(cfg, 'enable_visualization', True),
                )

            if self.result is not None:
                save_file = getattr(cfg, 'save_file', 'optimization_results.pkl')
                save_optimization_results(
                    self.graph, self.initial_estimate, self.result,
                    self.detections, self.uncertainty_info, save_file
                )
                print(f"[stage1] Results saved to: {save_file}")

        self.observation_graphs = [self.graph] if self.graph is not None else []
        return self.system, self.graph, self.initial_estimate, self.result, self.detections, self.uncertainty_info

    def run_wheel_optimization(self, visualize=True):
        """
        Process and optimize wheel images (Stage 2).
        Must be executed after run_optimization().

        Args:
            visualize: Render and save the Stage 2 3D plot.

        Returns:
            tuple: (wheel_graph, wheel_result, wheel_detections, wheel_camera_ids)
        """
        if self.result is None:
            print("⚠  run_optimization() must be executed first.")
            return None, None, None, None

        if self.config.wheel_camera_params is None or self.config.wheel_camera_params[0] <= 1.0:
            print("❌ Wheel camera parameters not set!")
            print("  Please open Camera Parameters dialog and enter Wheel camera values.")
            return None, None, None, None

        print("\n" + "=" * 60)
        print("[stage2] Hierarchical FGO - Stage 2: Processing Wheel Images...")
        print("=" * 60)

        # Initialize detection data for M2M visualization
        self.system.all_detections_for_viz = self.detections.copy()

        wheel_graph, wheel_result, wheel_detections, wheel_camera_ids = \
            self.system.process_wheel_images_with_base_result(self.result)

        if wheel_result is not None:
            print("  ✅ Stage 2 optimization completed successfully!")

            self.graph = wheel_graph
            self.result = wheel_result
            self.observation_graphs.append(wheel_graph)

            # Store wheel camera IDs (explicit type distinction)
            self.wheel_camera_ids = wheel_camera_ids

            # Note: Vehicle coordinate system is added after full pipeline completion

            self.detections.update(wheel_detections)

            all_marker_ids = sorted(list(set(
                d['marker_id']
                for detections_list in self.detections.values()
                for d in detections_list
            )))

            print(f"  Total markers after wheel: {len(all_marker_ids)}")

            # Add wheel detection data to M2M visualization
            self.system.all_detections_for_viz.update(wheel_detections)

            if visualize:
                print("\n[viz] Creating Stage 2 visualizations...")

                # visualization_results folder lives in hierarchical_fgo/
                viz_dir = os.path.join(fgc.g_current_dir, "visualization_results")
                os.makedirs(viz_dir, exist_ok=True)

                self.system.visualize_poses_3d_optimized_wheel_only(
                    wheel_result, all_marker_ids,
                    save_path=os.path.join(viz_dir, "stage2_markers_with_wheel_cameras.png")
                )
                print(f"  ✅ Stage 2 visualization saved to: {viz_dir}")
        else:
            print("  ⚠  Stage 2 optimization failed or skipped.")

        return wheel_graph, wheel_result, wheel_detections, wheel_camera_ids

    def run_vehicle_camera_optimization(self, vehicle_image_directory=None):
        """
        Process and optimize vehicle camera images (Stage 3).
        Must be executed after run_optimization().

        Args:
            vehicle_image_directory: Vehicle camera image directory (auto-search if None)

        Returns:
            tuple: (vehicle_graph, vehicle_result, vehicle_detections, vehicle_camera_ids)
        """
        if self.result is None:
            print("⚠  run_optimization() must be executed first.")
            return None, None, None, None

        print("\n" + "=" * 60)
        print("[stage3] Hierarchical FGO - Stage 3: Processing Vehicle Camera Images...")
        print("=" * 60)

        # Skip immediately if 0 cameras
        if len(self.config.vehicle_camera_configs) == 0:
            print("\n  ⚠  No vehicle cameras configured")
            print("\n  ⚠  Please add cameras and select images in Camera Parameters dialog")
            print("\n  Stage 3 skipped.")
            return None, None, None, None

        # Check selected images from Config for each camera
        has_selected_images = False
        total_selected_images = 0

        for i, cam_config in enumerate(self.config.vehicle_camera_configs):
            selected_imgs = cam_config.get('images', [])
            if selected_imgs:
                has_selected_images = True
                total_selected_images += len(selected_imgs)
                print(f"\n  Camera {i}: {len(selected_imgs)} images selected")

        if has_selected_images:
            print(f"\n  Total: {total_selected_images} images from GUI selection")
            print(f"\n  Using {len(self.config.vehicle_camera_configs)} cameras from config")
        else:
            # Skip Stage 3 if no images selected in GUI
            print("\n  ⚠  No images selected in GUI for vehicle cameras")
            print("\n  ⚠  Please select images in Camera Parameters dialog")
            print("\n  Stage 3 skipped.")
            return None, None, None, None

        from fgo_system import run_incremental_optimization_with_multiple_cameras

        vehicle_graph, vehicle_initial, vehicle_result, vehicle_detections, vehicle_camera_ids = \
            run_incremental_optimization_with_multiple_cameras(
                self.result,
                camera_params=self.config.camera_params,
                apriltag_config=build_apriltag_config(self.config),
                marker_size=self.config.marker_size,
                marker_size2=self.config.marker_size2,
                marker_size_mapping=self.config.marker_size_mapping,
                vehicle_markers=self.config.vehicle_markers,
                vehicle_camera_configs=self.config.vehicle_camera_configs,
                geo_skip_rot_deg=self.config.geo_skip_rot_deg,
                geo_skip_pos_m=self.config.geo_skip_pos_m,
            )

        if vehicle_result is not None:
            print("  ✅ Stage 3 optimization completed successfully!")

            self.graph = vehicle_graph
            self.result = vehicle_result
            self.observation_graphs.append(vehicle_graph)

            # Create vehicle coordinate system cache (needed for vehicle-centric visualization)
            print("  Computing vehicle coordinate system cache...")
            self.system.add_vehicle_coordinate_system_post_optimization(self.result)
            print("  ✅ Vehicle coordinate system cache created!")

            self.detections.update(vehicle_detections)

            # Add vehicle camera detections to M2M visualization data
            # (camera_id 20000+ is set per-detection inside run_incremental_optimization_with_multiple_cameras)
            if hasattr(self.system, 'all_detections_for_viz'):
                self.system.all_detections_for_viz.update(vehicle_detections)

            # Store vehicle camera detection data separately (used in reprojection)
            self.vehicle_detections = vehicle_detections

            # Store vehicle_camera_ids (used in visualization)
            self.vehicle_camera_ids = vehicle_camera_ids

            print(f"  Vehicle camera IDs: {sorted(vehicle_camera_ids)}")
            print(f"  Total detections: {len(vehicle_detections)} images")
        else:
            print("  ⚠  Stage 3 optimization failed or skipped.")
            vehicle_camera_ids = []

        return vehicle_graph, vehicle_result, vehicle_detections, vehicle_camera_ids
