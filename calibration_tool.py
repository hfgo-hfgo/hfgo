import sys
import os
import copy
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton,
                               QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFileDialog,
                               QDialog, QLineEdit, QFormLayout,
                               QTextEdit, QComboBox, QScrollArea, QMessageBox,
                               QTabWidget, QGroupBox, QProgressBar, QFrame, QInputDialog,
                               QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)
from PySide6.QtCore import Qt, QObject, Signal, QThread
from PySide6.QtGui import QFont, QTextCursor, QColor

# Matplotlib integration
import matplotlib
matplotlib.use('QtAgg')

# Matplotlib font settings (prevent Qt font warnings)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
# Journals reject Type 3 fonts (matplotlib's PDF/EPS default): 42 embeds the
# TrueType face itself, and svg.fonttype='none' keeps SVG text as text.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 16

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the '3d' projection

import cv2
import numpy as np

# Import config.py with relative path
sys.path.insert(0, os.path.dirname(__file__))
from src.config import CalibrationConfig
from fgo_system import APRILTAG_FAMILY_GEOMETRY


# ── Shared widget styles ──────────────────────────────────────────────────
_TAB_WIDGET_STYLE = """
    QTabWidget::pane {
        border: 1px solid #dee2e6;
        border-radius: 5px;
        background-color: white;
    }
    QTabBar::tab {
        background-color: #e9ecef;
        padding: 10px 20px;
        margin-right: 2px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
    }
    QTabBar::tab:selected {
        background-color: #1b338c;
        color: white;
    }
"""

_SPLITTER_STYLE = """
    QSplitter::handle:horizontal {
        background-color: #e0e0e0;
        border-left: 1px solid #cfcfcf;
        border-right: 1px solid #cfcfcf;
    }
    QSplitter::handle:horizontal:hover {
        background-color: #1b338c;
    }
    QSplitter::handle:horizontal:pressed {
        background-color: #142a70;
    }
"""

# Dragging the splitter past these would leave a panel too narrow to operate.
_SIDEBAR_MIN_WIDTH = 200
_DASHBOARD_MIN_WIDTH = 460

# Natural width of the sidebar's controls. The panel is stretched to the
# viewport while there is room, but never squeezed below this — past it the
# scroll area pans horizontally instead, so a narrowed sidebar keeps its rows
# readable rather than compressing them into unusable slivers.
_SIDEBAR_CONTENT_WIDTH = 340

_OK_BUTTON_STYLE = (
    "QPushButton { background-color: #4CAF50; color: white; border: none;"
    " border-radius: 5px; font-size: 14px; }"
    "QPushButton:hover { background-color: #45a049; }"
)

_CANCEL_BUTTON_STYLE = (
    "QPushButton { background-color: #f44336; color: white; border: none;"
    " border-radius: 5px; font-size: 14px; }"
    "QPushButton:hover { background-color: #da190b; }"
)


def _action_button(text, color, hover_color, padding='8px 15px', font_size='13px'):
    """Create a flat coloured action button for a tab's control row."""
    button = QPushButton(text)
    button.setStyleSheet(f"""
        QPushButton {{
            background-color: {color};
            color: white;
            border: none;
            border-radius: 5px;
            padding: {padding};
            font-size: {font_size};
        }}
        QPushButton:hover {{
            background-color: {hover_color};
        }}
    """)
    return button


def _dialog_button_row(ok_slot, cancel_slot, stretch=0):
    """Build the green-OK / red-Cancel footer row shared by the settings dialogs."""
    ok_button = QPushButton("OK")
    ok_button.setFixedSize(100, 40)
    ok_button.setStyleSheet(_OK_BUTTON_STYLE)
    ok_button.clicked.connect(ok_slot)

    cancel_button = QPushButton("Cancel")
    cancel_button.setFixedSize(100, 40)
    cancel_button.setStyleSheet(_CANCEL_BUTTON_STYLE)
    cancel_button.clicked.connect(cancel_slot)

    row = QHBoxLayout()
    row.addStretch(stretch)
    row.addWidget(ok_button)
    row.addWidget(cancel_button)
    return row


class OutputRedirector(QObject):
    """Class to redirect stdout/stderr to GUI"""
    output_written = Signal(str)

    def __init__(self):
        super().__init__()

    def write(self, text):
        if text.strip():  # Exclude empty lines
            self.output_written.emit(str(text))

    def flush(self):
        pass


class CalibrationThread(QThread):
    """Execute calibration in a separate thread"""
    finished = Signal(bool, str, str)  # success, message, calibration_type
    progress = Signal(str)  # progress message
    calibration_system = Signal(object)  # Pass CalibrationSystem object

    def __init__(self, config, calibration_type='environment'):
        super().__init__()
        self.config = config
        self.calibration_type = calibration_type

    def run(self):
        try:
            from src.calibration import CalibrationSystem

            self.progress.emit(f"Starting {self.calibration_type} calibration...\n")

            # Use config received from GUI (don't create new one!)
            config = self.config
            calib_system = CalibrationSystem(config)

            # Pass CalibrationSystem object to GUI
            self.calibration_system.emit(calib_system)

            if self.calibration_type == 'environment':
                # Stage 1: Environment Marker Optimization (always run new and save)
                system, graph, initial_estimate, result, detections, uncertainty_info = \
                    calib_system.run_optimization(use_saved=False)

                if result is None:
                    raise Exception("Environment optimization failed!")

                self.progress.emit("\n✓ Environment optimization completed and saved to optimization_results.pkl!\n")
                self.finished.emit(True,
                    "Environment optimization completed successfully!\n\n"
                    "Results saved to: optimization_results.pkl\n"
                    "You can now run Extrinsic Calibration to process wheel and vehicle cameras.",
                    'environment')
            else:
                # Full pipeline - extrinsic calibration
                # Stage 1: Load Environment results from saved pkl file
                self.progress.emit("\n[stage1] Loading saved Environment optimization results...\n")
                system, graph, initial_estimate, result, detections, uncertainty_info = \
                    calib_system.run_optimization(use_saved=True)

                if result is None:
                    # No saved optimization_results.pkl (or it failed to load) —
                    # fall back to running Environment optimization fresh instead
                    # of failing outright.
                    self.progress.emit(
                        "\n⚠  No saved Environment results found (or failed to load).\n"
                        "   Running Environment optimization now...\n\n"
                    )
                    system, graph, initial_estimate, result, detections, uncertainty_info = \
                        calib_system.run_optimization(use_saved=False)

                    if result is None:
                        raise Exception(
                            "Environment optimization failed!\n"
                            "Please check camera parameters and image directories, then try again."
                        )
                    self.progress.emit("\n  ✅ Environment optimization completed and saved to optimization_results.pkl!\n\n")
                else:
                    self.progress.emit("  ✅ Stage 1 loaded from saved results!\n\n")

                self.progress.emit("[stage2] Wheel Image Optimization...\n")
                wheel_graph, wheel_result, wheel_detections, wheel_camera_ids = \
                    calib_system.run_wheel_optimization()

                if wheel_result is not None:
                    self.progress.emit("\n  ✅ Stage 2 complete!\n\n")
                else:
                    self.progress.emit("\n  ⚠  Stage 2 skipped or failed\n\n")

                self.progress.emit("[stage3] Vehicle Camera Optimization...\n")
                vehicle_graph, vehicle_result, vehicle_detections, vehicle_camera_ids = \
                    calib_system.run_vehicle_camera_optimization()

                if vehicle_result is not None:
                    self.progress.emit("  ✅ Stage 3 complete!\n\n")
                else:
                    self.progress.emit("  ⚠  Stage 3 skipped or failed\n\n")

                # Stage 2/3 failures must be visible in the completion dialog,
                # not just in the log — otherwise a wheel/vehicle failure is
                # presented as a full success.
                stage_warnings = []
                if wheel_result is None:
                    stage_warnings.append("⚠ Stage 2 (wheel) skipped or failed")
                if vehicle_result is None:
                    stage_warnings.append("⚠ Stage 3 (vehicle) skipped or failed")
                if stage_warnings:
                    msg = ("✓ Extrinsic calibration finished with warnings:\n\n"
                           + "\n".join(stage_warnings)
                           + "\n\nSee the Calibration Log tab for details.")
                else:
                    msg = "✓ Extrinsic calibration completed!"
                self.finished.emit(True, msg, 'extrinsic')

        except Exception as e:
            import traceback
            error_msg = f"✗ Error: {str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg, self.calibration_type)


def vehicle_frame_camera_poses(fgo_system, result, vehicle_camera_ids):
    """{camera_index: [x, y, z, roll, pitch, yaw]} in the vehicle frame.

    Same transform Export Poses writes out: metres and intrinsic-ZYX degrees,
    keyed by the Cam number rather than the internal 20000+ camera id.
    """
    import gtsam
    from scipy.spatial.transform import Rotation

    frame = getattr(fgo_system, 'cached_vehicle_frame', None)
    if frame is None:
        fgo_system.add_vehicle_coordinate_system_post_optimization(result)
        frame = getattr(fgo_system, 'cached_vehicle_frame', None)
    if frame is None:
        return {}

    center = np.asarray(frame['center'], dtype=float)
    R_world_to_vehicle = np.asarray(frame['R_world_to_vehicle'], dtype=float)

    poses = {}
    for camera_id in sorted(vehicle_camera_ids):
        camera_key = gtsam.symbol('C', camera_id)
        if not result.exists(camera_key):
            continue

        T_camera_in_world = result.atPose3(camera_key)
        position = R_world_to_vehicle @ (
            np.asarray(T_camera_in_world.translation(), dtype=float) - center)
        rotation = R_world_to_vehicle @ T_camera_in_world.rotation().matrix()
        yaw, pitch, roll = Rotation.from_matrix(rotation).as_euler('ZYX', degrees=True)

        poses[camera_id - 20000] = [
            float(position[0]), float(position[1]), float(position[2]),
            float(roll), float(pitch), float(yaw),
        ]
    return poses


class PathSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        # Create CalibrationConfig instance
        self.config = CalibrationConfig()

        # Set relative path based on project root
        self.project_root = os.path.dirname(os.path.abspath(__file__))

        # Calibration thread
        self.calibration_thread = None

        # CalibrationSystem object (for memory reference)
        self.calibration = None

        # Visualization figure cache: {viz_type_index: Figure}
        # Invalidated when new calibration completes
        self._viz_cache = {}

        # Setup stdout/stderr redirectors
        self.stdout_redirector = OutputRedirector()
        self.stderr_redirector = OutputRedirector()

        self.init_ui()
        self.update_config_display()

        # Connect output redirect
        self.stdout_redirector.output_written.connect(self.append_log)
        self.stderr_redirector.output_written.connect(self.append_log)

    def init_ui(self):
        self.setWindowTitle("Hierarchical Factor Graph Optimization Calibration Tool")

        # Get screen geometry for responsive sizing
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # Set window size as percentage of screen size
        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.7)

        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.setGeometry(x, y, window_width, window_height)

        # Set minimum size to prevent too small windows
        self.setMinimumSize(1000, 700)

        # Central widget setup
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #f0f0f0;")
        self.setCentralWidget(central_widget)

        # Main horizontal layout (Left:Right = 1:2)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # A splitter rather than fixed stretch factors, so the sidebar can be
        # widened when a long dataset path needs reading and narrowed again to
        # give the dashboard's plots and tables the room they want.
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setStyleSheet(_SPLITTER_STYLE)
        self.main_splitter.setHandleWidth(7)
        # Collapsing a panel to zero is easy to do by accident and hard to
        # notice afterwards, so the minimum widths below are the hard stops.
        self.main_splitter.setChildrenCollapsible(False)

        sidebar = self._build_sidebar()
        sidebar.setMinimumWidth(_SIDEBAR_MIN_WIDTH)
        dashboard = self._build_right_panel()
        dashboard.setMinimumWidth(_DASHBOARD_MIN_WIDTH)

        self.main_splitter.addWidget(sidebar)
        self.main_splitter.addWidget(dashboard)

        # Keep the 1:2 split this window has always opened with, and let a
        # window resize divide the new space the same way.
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([window_width // 3, window_width - window_width // 3])

        main_layout.addWidget(self.main_splitter)

    def _add_section_header(self, layout, text, icon_slot=None, margin_top=15):
        """Add a bold sidebar section header, optionally with a trailing ⋯ button."""
        row = QHBoxLayout()
        label = QLabel(text)
        label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: #333; margin-top: {margin_top}px;")
        row.addWidget(label)
        row.addStretch()
        if icon_slot is not None:
            button = self._create_icon_button("⋯")
            button.clicked.connect(icon_slot)
            row.addWidget(button)
        layout.addLayout(row)

    def _add_path_row(self, layout, text, dir_type):
        """Add a '<name>: <path>' row with a ⋯ directory picker; returns the label."""
        row = QHBoxLayout()
        label = QLabel(text)
        label.setStyleSheet("font-size: 13px; padding: 8px; background-color: #f8f9fa; border-radius: 5px;")
        label.setWordWrap(True)
        row.addWidget(label)

        button = self._create_icon_button("⋯")
        button.clicked.connect(lambda: self.select_directory(dir_type))
        row.addWidget(button)
        layout.addLayout(row)
        return label

    def _build_sidebar(self):
        """Build the left control panel inside its scroll area."""
        left_panel = QWidget()
        # No border-right: the splitter handle is the separator now, and both
        # together read as a doubled line.
        left_panel.setStyleSheet("background-color: white;")
        # Floors the width the rows are laid out at. Without it widgetResizable
        # shrinks the panel to whatever the viewport is, so the content can never
        # overflow and the horizontal scroll bar below would never appear.
        left_panel.setMinimumWidth(_SIDEBAR_CONTENT_WIDTH)

        # ScrollArea for left panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)

        # Title Label
        title_label = QLabel("⚙ Configuration Panel")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #1b338c; padding: 10px;")
        left_layout.addWidget(title_label)

        self._build_directory_section(left_layout)
        self._build_camera_section(left_layout)
        self._add_section_header(left_layout, "🎯 Marker Settings", self.open_marker_dialog)
        self._add_section_header(left_layout, "📊 Uncertainty Settings", self.open_uncertainty_dialog)
        self._build_config_management_section(left_layout)
        self._build_run_section(left_layout)

        # Add expanding spacer to push content to top
        left_layout.addStretch(1)

        return scroll_area

    def _build_directory_section(self, left_layout):
        """Sidebar section: environment / wheel image directories."""
        self._add_section_header(left_layout, "📁 Directory Settings", margin_top=10)
        self.env_path_label = self._add_path_row(
            left_layout, f"Environment: <b>{self.config.image_directory}</b>", 'environment')
        self.wheel_path_label = self._add_path_row(
            left_layout, f"Wheel: <b>{self.config.wheel_directory}</b>", 'wheel')

    def _build_camera_section(self, left_layout):
        """Sidebar section: camera parameters."""
        self._add_section_header(left_layout, "📷 Camera Parameters", self.open_camera_param_dialog)

        self.vehicle_cam_count_label = QLabel(f"Vehicle Cameras: <b>{self.config.num_vehicle_cameras}</b> cameras")
        self.vehicle_cam_count_label.setStyleSheet("font-size: 12px; padding: 5px; color: #666; margin-left: 10px;")
        left_layout.addWidget(self.vehicle_cam_count_label)

    def _build_config_management_section(self, left_layout):
        """Sidebar section: save / load config."""
        self._add_section_header(left_layout, "💾 Config Management")

        save_load_layout = QHBoxLayout()
        save_load_layout.setSpacing(10)

        save_button = self._create_compact_button("💾 Save", "#388e3c")
        save_button.clicked.connect(self.save_config)
        save_load_layout.addWidget(save_button)

        load_button = self._create_compact_button("📂 Load", "#0288d1")
        load_button.clicked.connect(self.load_config)
        save_load_layout.addWidget(load_button)

        left_layout.addLayout(save_load_layout)

    def _build_run_section(self, left_layout):
        """Sidebar section: calibration run buttons."""
        self._add_section_header(left_layout, "🚀 Run Calibration")

        env_calibration_button = self._create_compact_button("🌍 Environment", "#d32f2f") # calibration starting point
        env_calibration_button.clicked.connect(self.run_environment_calibration)
        left_layout.addWidget(env_calibration_button)

        extrinsic_calibration_button = self._create_compact_button("🚗 Extrinsic", "#c2185b")
        extrinsic_calibration_button.clicked.connect(self.run_extrinsic_calibration)
        left_layout.addWidget(extrinsic_calibration_button)

    def _build_right_panel(self):
        """Build the right dashboard panel and its four tabs."""
        right_panel = QWidget()
        right_panel.setStyleSheet("background-color: #ffffff;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)

        display_title = QLabel("📋 Dashboard")
        display_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1b338c; margin-bottom: 10px;")
        right_layout.addWidget(display_title)

        self.right_tab_widget = QTabWidget()
        self.right_tab_widget.setStyleSheet(_TAB_WIDGET_STYLE)

        self.right_tab_widget.addTab(self._build_config_tab(), "⚙ Configuration")
        self.right_tab_widget.addTab(self._build_log_tab(), "📝 Calibration Log")
        self.right_tab_widget.addTab(self._build_viz_tab(), "📊 3D Visualization")
        self.right_tab_widget.addTab(self._build_projection_tab(), "📷 Projection")

        # Auto load on tab switch
        self.right_tab_widget.currentChanged.connect(self.on_tab_changed)

        right_layout.addWidget(self.right_tab_widget)
        return right_panel

    def _build_config_tab(self):
        """Dashboard tab showing the current configuration dump."""
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        self.config_display = QTextEdit()
        self.config_display.setReadOnly(True)
        self.config_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
            }
        """)
        config_layout.addWidget(self.config_display)
        return config_tab

    def _build_log_tab(self):
        """Dashboard tab with the progress bar and calibration log console."""
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #dee2e6;
                border-radius: 5px;
                text-align: center;
                background-color: #f8f9fa;
            }
            QProgressBar::chunk {
                background-color: #1b338c;
            }
        """)
        log_layout.addWidget(self.progress_bar)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Courier New', monospace;
                font-size: 18px;
            }
        """)
        log_layout.addWidget(self.log_display)

        clear_log_button = _action_button("Clear Log", "#6c757d", "#5a6268",
                                         padding='8px', font_size='12px')
        clear_log_button.clicked.connect(self.clear_log)
        log_layout.addWidget(clear_log_button)
        return log_tab

    def _build_viz_tab(self):
        """Dashboard tab with the 3D visualization canvas and its controls."""
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)

        # Visualization selection controls
        control_layout = QHBoxLayout()

        viz_type_label = QLabel("Select Visualization:")
        viz_type_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        control_layout.addWidget(viz_type_label)

        self.viz_type_combo = QComboBox()
        # Appended rather than reordered: load_visualization dispatches on the
        # combo index, so the existing entries must keep their indices.
        self.viz_type_combo.addItems([
            "Stage 1: Environment Marker Optimization",
            "Stage 2: Wheel Marker Optimization",
            "Stage 3: Vehicle Camera Optimization (Final)",
            "Marker Corner Point Cloud",
            "Camera Poses: Env Marker Cameras",
            "Camera Poses: Wheel Marker Cameras",
            "Stage 4: Vehicle Camera Optimization"
        ])
        self.viz_type_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                font-size: 13px;
                min-width: 250px;
            }
        """)
        self.viz_type_combo.currentIndexChanged.connect(self.load_visualization)
        control_layout.addWidget(self.viz_type_combo)

        control_layout.addStretch()

        refresh_viz_button = _action_button("🔄 Refresh", "#1b338c", "#162f7d")
        refresh_viz_button.clicked.connect(self.refresh_visualization)
        control_layout.addWidget(refresh_viz_button)

        # M2M Pose Graph button — opens a separate matplotlib window
        m2m_graph_button = _action_button("📐 M2M Pose Graph", "#6f42c1", "#5a32a3")
        m2m_graph_button.clicked.connect(self.show_m2m_pose_graph)
        control_layout.addWidget(m2m_graph_button)

        m2m_diag_button = _action_button("🔍 M2M Pair Diagnostic", "#c0392b", "#a93226")
        m2m_diag_button.clicked.connect(self.show_m2m_pair_diagnostic)
        control_layout.addWidget(m2m_diag_button)

        # Open3D pose scene — GPU-rendered window for the two camera-pose views
        open3d_button = _action_button("🧊 Open3D View", "#00897b", "#00695c")
        open3d_button.setToolTip(
            "Opens the selected camera-pose view in an Open3D window.\n"
            "\n"
            "Rendered on the GPU, so rotating stays smooth with hundreds of cameras.")
        open3d_button.clicked.connect(self.show_pose_scene_open3d)
        control_layout.addWidget(open3d_button)

        viz_layout.addLayout(control_layout)

        # Matplotlib Figure
        self.viz_figure = Figure(figsize=(10, 8))
        self.viz_canvas = FigureCanvas(self.viz_figure)
        self.viz_toolbar = NavigationToolbar(self.viz_canvas, viz_tab)
        self._fix_toolbar_fonts(self.viz_toolbar)

        viz_layout.addWidget(self.viz_toolbar)
        viz_layout.addWidget(self.viz_canvas)
        # Kept for _show_viz_entry: each figure owns its own canvas/toolbar
        # (created once in _make_viz_entry) and they are swapped into this
        # layout instead of re-targeting one shared canvas.
        self._viz_layout = viz_layout
        self._viz_tab = viz_tab
        return viz_tab

    def _build_projection_tab(self):
        """Dashboard tab listing per-camera projection overlays and poses."""
        projection_tab = QWidget()
        projection_layout = QVBoxLayout(projection_tab)
        projection_layout.setContentsMargins(10, 10, 10, 10)

        # Control buttons
        projection_control_layout = QHBoxLayout()
        projection_control_layout.addStretch()

        refresh_projection_button = _action_button("🔄 Refresh", "#1b338c", "#162f7d")
        refresh_projection_button.clicked.connect(self.load_projection_results)
        projection_control_layout.addWidget(refresh_projection_button)

        export_images_button = _action_button("💾 Export Images", "#28a745", "#218838")
        export_images_button.clicked.connect(self.export_projection_images)
        projection_control_layout.addWidget(export_images_button)

        export_poses_button = _action_button("📄 Export Poses", "#17a2b8", "#138496")
        export_poses_button.clicked.connect(self.export_vehicle_poses)
        projection_control_layout.addWidget(export_poses_button)

        projection_layout.addLayout(projection_control_layout)

        # Scroll area for projection results
        self.projection_scroll = QScrollArea()
        self.projection_scroll.setWidgetResizable(True)
        self.projection_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.projection_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # Container widget for projection content
        self.projection_content = QWidget()
        self.projection_content_layout = QVBoxLayout(self.projection_content)
        self.projection_content_layout.setSpacing(20)
        self.projection_content_layout.addStretch()

        self.projection_scroll.setWidget(self.projection_content)
        projection_layout.addWidget(self.projection_scroll)
        return projection_tab

    def _create_icon_button(self, icon_text):
        """Helper method to create small icon buttons (⋯ style)"""
        button = QPushButton(icon_text)
        button.setFixedSize(32, 32)
        button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                background-color: #e0e0e0;
                color: #333;
                border: none;
                border-radius: 4px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
        """)
        return button

    def _create_compact_button(self, text, color):
        """Helper method to create compact styled buttons"""
        button = QPushButton(text)
        button.setMinimumSize(120, 35)
        button.setStyleSheet(f"""
            QPushButton {{
                font-size: 13px;
                padding: 8px 12px;
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color, 0.8)};
            }}
        """)
        return button

    def _darken_color(self, hex_color, factor=0.9):
        """Darken a hex color"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(int(c * factor) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"

    def select_directory(self, dir_type):
        """Directory selection dialog"""
        title_map = {
            'environment': 'Select Environment Image Directory',
            'wheel': 'Select Wheel Image Directory'
        }

        dir_path = QFileDialog.getExistingDirectory(
            self,
            title_map.get(dir_type, "Select Directory"),
            self.project_root
        )

        if dir_path:
            # Convert to relative path
            try:
                rel_path = os.path.relpath(dir_path, self.project_root)
            except ValueError:
                # Use absolute path if different drive
                rel_path = dir_path

            # Add trailing slash (match config.py format)
            if not rel_path.endswith('/'):
                rel_path = rel_path + '/'

            # Update config
            if dir_type == 'environment':
                self.config.update_image_directory(rel_path)
                self.env_path_label.setText(f"Environment: <b>{rel_path}</b>")
            elif dir_type == 'wheel':
                self.config.update_wheel_directory(rel_path)
                self.wheel_path_label.setText(f"Wheel: <b>{rel_path}</b>")

            self.update_config_display()
            print(f"[config] {dir_type.capitalize()} directory updated: {rel_path}")

    def open_camera_param_dialog(self):
        """Open camera parameter input dialog"""
        dialog = CameraParameterDialog(self, self.config)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            params = dialog.get_parameters()
            if params:
                # Update environment camera
                env_intrinsic = params['environment']['intrinsic']
                env_dist = params['environment']['dist_coeffs']
                self.config.update_camera_params(
                    env_intrinsic[0], env_intrinsic[1], env_intrinsic[2], env_intrinsic[3],
                    env_dist[0], env_dist[1], env_dist[2], env_dist[3], env_dist[4]
                )

                # Update wheel camera
                wheel_intrinsic = params['wheel']['intrinsic']
                wheel_dist = params['wheel']['dist_coeffs']  # UI order: [k1, k2, k3, p1, p2]
                self.config.wheel_camera_params = wheel_intrinsic
                # Convert UI order [k1, k2, k3, p1, p2] → OpenCV order [k1, k2, p1, p2, k3]
                self.config.wheel_dist_coeffs = np.array([wheel_dist[0], wheel_dist[1],
                                                           wheel_dist[3], wheel_dist[4], wheel_dist[2]])

                # Update vehicle cameras (dynamic count)
                num_vehicles_from_dialog = len(params['vehicles'])

                # Filter only cameras with images selected (important!)
                vehicles_with_images = [(orig_idx, vp) for orig_idx, vp in enumerate(params['vehicles'])
                                        if len(vp.get('images', [])) > 0]
                num_cameras_with_images = len(vehicles_with_images)

                print(f"✓ Vehicle cameras from dialog: {num_vehicles_from_dialog} total, {num_cameras_with_images} with images")

                # Update config with only cameras that have images
                if num_cameras_with_images > 0:
                    self.config.update_num_vehicle_cameras(num_cameras_with_images)

                    for i, (orig_idx, vehicle_params) in enumerate(vehicles_with_images):
                        if i < len(self.config.vehicle_camera_configs):
                            if orig_idx != i:
                                # Cameras without images are compacted out, so the
                                # internal slot can differ from the GUI tab number.
                                print(f"   [camera-slot] GUI Cam{orig_idx} → internal Cam{i} (C{20000 + i})")
                            self.config.vehicle_camera_configs[i]['intrinsic'] = vehicle_params['intrinsic']
                            self.config.vehicle_camera_configs[i]['camera_model'] = vehicle_params.get('camera_model', 'pinhole')
                            self.config.vehicle_camera_configs[i]['images'] = vehicle_params.get('images', [])

                            if vehicle_params.get('camera_model') == 'fisheye':
                                # Fisheye KB: dist_coeffs is [k1, k2, k3, k4] (4-element)
                                vd = vehicle_params['dist_coeffs']
                                self.config.vehicle_camera_configs[i]['dist_coeffs'] = np.array(vd, dtype=np.float64)
                            else:
                                # Pinhole: UI order [k1, k2, k3, p1, p2] → OpenCV order [k1, k2, p1, p2, k3]
                                vd = vehicle_params['dist_coeffs']
                                self.config.vehicle_camera_configs[i]['dist_coeffs'] = np.array([vd[0], vd[1], vd[3], vd[4], vd[2]])

                    print(f"✓ Config updated with {num_cameras_with_images} cameras (only cameras with images)")
                else:
                    # Set config to empty state if no images (to skip Stage 3)
                    print("⚠  No cameras have images selected")
                    self.config.update_num_vehicle_cameras(0)  # Set to 0
                    print("✓ Config cleared - Stage 3 will be skipped")

                # Update camera count label
                self.vehicle_cam_count_label.setText(f"Vehicle Cameras: <b>{self.config.num_vehicle_cameras}</b> cameras")

                self.update_config_display()
                print("✓ All camera parameters updated:")
                print(f"  Environment: fx={env_intrinsic[0]}, fy={env_intrinsic[1]}, cx={env_intrinsic[2]}, cy={env_intrinsic[3]}")
                print(f"  Wheel: fx={wheel_intrinsic[0]}, fy={wheel_intrinsic[1]}, cx={wheel_intrinsic[2]}, cy={wheel_intrinsic[3]}")
                for i, vp in enumerate(params['vehicles']):
                    num_imgs = len(vp.get('images', []))
                    model_str = vp.get('camera_model', 'pinhole')
                    print(f"  Vehicle Cam{i}: fx={vp['intrinsic'][0]}, fy={vp['intrinsic'][1]}, cx={vp['intrinsic'][2]}, cy={vp['intrinsic'][3]} ({num_imgs} images) [model={model_str}]")

    def open_marker_dialog(self):
        """Open marker settings dialog"""
        dialog = MarkerConfigDialog(self, self.config)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_config_display()
            print("[config] Marker configuration updated")

    def open_uncertainty_dialog(self):
        """Open uncertainty settings dialog"""
        dialog = UncertaintyDialog(self, self.config)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_config_display()
            print("[config] Uncertainty configuration updated")

    def closeEvent(self, event):
        """Leave the process without letting Qt destroy a running worker.

        Qt aborts the process when a QThread is destroyed while still running,
        and the calibration worker sits inside long gtsam/OpenCV calls that
        cannot be interrupted cooperatively. Quitting through os._exit() also
        skips interpreter teardown, where Qt widgets, matplotlib canvases,
        gtsam values and the AprilTag detector's worker threads would be
        destroyed in Python's arbitrary GC order — the crash ("segmentation
        fault" / "malloc(): mismatching next->prev_size") that used to appear
        after a session. Every result is written to disk before this point,
        so nothing is lost by not unwinding.
        """
        thread = getattr(self, 'calibration_thread', None)
        if thread is not None and thread.isRunning():
            reply = QMessageBox.question(
                self, "Calibration Running",
                "A calibration is still running.\n\n"
                "Quitting now stops it immediately; results of the current "
                "run will not be saved.\n\nQuit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
        event.accept()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def append_log(self, text):
        """Add text to log window"""
        self.log_display.moveCursor(QTextCursor.MoveOperation.End)
        self.log_display.insertPlainText(text)
        self.log_display.moveCursor(QTextCursor.MoveOperation.End)

    def clear_log(self):
        """Clear log window"""
        self.log_display.clear()
        self.append_log("=== Log cleared ===\n")

    def on_calibration_system_ready(self, calib_system):
        """Receive CalibrationSystem object (save memory reference)"""
        self.calibration = calib_system

    def _show_pending_pose_tables(self):
        """Display marker-pose tables queued by the optimization stages.

        The stages run in a worker thread and only queue data
        (system.pending_pose_tables); windows are created here on the main
        thread. Replaces the old tkinter window that was spawned from a
        background thread.
        """
        try:
            system = getattr(getattr(self, 'calibration', None), 'system', None)
            tables = getattr(system, 'pending_pose_tables', None)
            if not tables:
                return
            if not hasattr(self, '_pose_table_dialogs'):
                self._pose_table_dialogs = []

            from PySide6.QtWidgets import (QDialog, QVBoxLayout, QPlainTextEdit,
                                           QPushButton, QApplication)
            from PySide6.QtGui import QFont

            while tables:
                info = tables.pop(0)

                dialog = QDialog(self)
                dialog.setWindowTitle(info['title'])
                dialog.resize(760, 520)
                layout = QVBoxLayout(dialog)

                text = QPlainTextEdit(info['pretty_text'])
                text.setReadOnly(True)
                font = QFont('Courier New')
                font.setStyleHint(QFont.Monospace)
                font.setPointSize(10)
                text.setFont(font)
                layout.addWidget(text)

                copy_button = QPushButton("Copy All Values (Excel tab-separated)")
                values_tsv = info['values_tsv']

                def _copy(_checked=False, btn=copy_button, tsv=values_tsv):
                    QApplication.clipboard().setText(tsv)
                    btn.setText("✓ Copied!")

                copy_button.clicked.connect(_copy)
                layout.addWidget(copy_button)

                dialog.show()                      # modeless
                self._pose_table_dialogs.append(dialog)
        except Exception as exc:
            self.append_log(f"⚠  Pose table display failed: {exc}\n")
            # This runs on the main thread, so a visible warning is safe —
            # without it the user waits for a popup that will never appear.
            QMessageBox.warning(
                self, "Pose Table",
                f"⚠ Marker pose table could not be displayed:\n\n{exc}\n\n"
                "The pose values were still printed to the Calibration Log tab.")

    def _vehicle_images_dir(self):
        """Common parent directory of the selected vehicle camera images.

        There is no single vehicle_directory in the config (images are picked
        per camera), so show the deepest directory that contains them all.
        """
        paths = [p for vc in self.config.vehicle_camera_configs
                 for p in vc.get('images', [])]
        if not paths:
            return "(not selected)"
        try:
            dirs = [os.path.dirname(os.path.abspath(p)) for p in paths]
            common = os.path.commonpath(dirs)
        except ValueError:
            return "(mixed locations)"
        n_cams = sum(1 for vc in self.config.vehicle_camera_configs if vc.get('images'))
        return f"{common}/  ({len(paths)} images, {n_cams} cameras)"

    def _update_viz_options(self):
        """Enable only the visualization types the current data can draw.

        Combo order: 0 Stage 1 environment markers, 1 Stage 2 wheel markers
        (needs wheel), 2 Stage 3 vehicle cameras (needs vehicle),
        3 Marker Corner Point Cloud, 4 Stage 1 pose scene,
        5 Stage 2 pose scene (needs wheel), 6 Stage 4 nodes-only vehicle view
        (needs vehicle, same data as Stage 3).
        """
        try:
            calib = getattr(self, 'calibration', None)
            has_wheel = bool(getattr(calib, 'wheel_camera_ids', None))
            has_vehicle = bool(getattr(calib, 'vehicle_camera_ids', None))
            available = {0: True, 1: has_wheel, 2: has_vehicle, 3: True,
                         4: True, 5: has_wheel, 6: has_vehicle}

            model = self.viz_type_combo.model()
            for idx, ok in available.items():
                item = model.item(idx)
                if item is not None:
                    item.setEnabled(ok)

            if not available.get(self.viz_type_combo.currentIndex(), True):
                # Silent reset: the caller decides what to load afterwards, so
                # this must not trigger load_visualization on its own (it may
                # run while the figure cache is stale).
                self.viz_type_combo.blockSignals(True)
                self.viz_type_combo.setCurrentIndex(0)   # Stage 1
                self.viz_type_combo.blockSignals(False)
        except Exception as exc:
            self.append_log(f"⚠  viz option update failed: {exc}\n")

    def on_calibration_finished(self, success, message, calibration_type='environment'):
        """Calibration completion callback"""
        # Restore stdout/stderr
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

        # Show pose tables prepared by the worker. This slot runs on the Qt
        # main thread, which is the only place windows may be created.
        self._show_pending_pose_tables()

        self.progress_bar.setVisible(False)

        # Display message
        self.append_log("\n" + "=" * 60 + "\n")
        self.append_log(message + "\n")
        self.append_log("=" * 60 + "\n")

        if elapsed_str is not None:
            message = message + f"\n\n⏱ Elapsed time: {elapsed_str}"

        if success:
            # Invalidate visualization cache — new calibration result available
            self._clear_viz_cache()

            # Grey out visualization types whose stage has no data (must run
            # after the cache clear so its silent index reset cannot leave a
            # stale figure on screen, and only on success so a failed run
            # keeps the previous state untouched).
            self._update_viz_options()

            # Select appropriate visualization based on calibration type.
            # The extrinsic worker reports success even when Stage 3 was
            # skipped, so fall back to a view that actually has data instead
            # of selecting the just-disabled vehicle entry.
            calib = getattr(self, 'calibration', None)
            has_wheel = bool(getattr(calib, 'wheel_camera_ids', None))
            has_vehicle = bool(getattr(calib, 'vehicle_camera_ids', None))
            if calibration_type == 'environment':
                target_idx = 0                        # Stage 1
            elif has_vehicle:
                target_idx = 2                        # Stage 3
            elif has_wheel:
                target_idx = 1                        # Stage 2 (Stage 3 skipped)
            else:
                target_idx = 0
            if self.viz_type_combo.currentIndex() == target_idx:
                # setCurrentIndex on the same index emits no signal, so the
                # visualization would silently never load — call it directly.
                self.load_visualization()
            else:
                self.viz_type_combo.setCurrentIndex(target_idx)

            # Switch to Visualization tab
            self.right_tab_widget.setCurrentIndex(2)
            if '⚠' in message:
                # Partial success (e.g. Stage 2/3 skipped) — use the warning
                # icon so the user does not mistake it for a full success.
                QMessageBox.warning(self, "Completed with Warnings",
                                    message + "\n\nVisualization loaded in 3D Visualization tab!")
            else:
                QMessageBox.information(self, "Success", message + "\n\nVisualization loaded in 3D Visualization tab!")
        else:
            QMessageBox.critical(self, "Error", message)

    @staticmethod
    def _fix_toolbar_fonts(widget):
        """Set a consistent font on a NavigationToolbar and all its children
        (prevents Qt font warnings; QLabel coordinate readouts styled too)."""
        from PySide6.QtWidgets import QLabel as QtLabel
        toolbar_font = QFont("Segoe UI", 9)
        widget.setFont(toolbar_font)
        if isinstance(widget, QtLabel):
            widget.setStyleSheet("QLabel { font-size: 9pt; }")
        for child in widget.findChildren(QWidget):
            child.setFont(toolbar_font)
            if isinstance(child, QtLabel):
                child.setStyleSheet("QLabel { font-size: 9pt; }")

    def _make_viz_entry(self, fig):
        """Create the canvas + toolbar that will accompany fig for its whole life.

        Re-targeting a live FigureCanvas to another figure is unsupported by
        matplotlib (stale pixel buffer → leftovers/garbage/segfault), and the
        event-callback registry lives on the *figure*, so re-attaching a cached
        figure to a fresh canvas leaves the previous canvas's toolbar callbacks
        behind ("Internal C++ object already deleted" on mouse move). Binding
        figure, canvas and toolbar 1:1:1 once avoids both failure modes; the
        widgets are cached alongside the figure and re-shown, never re-created.
        """
        canvas = FigureCanvas(fig)            # also does fig.set_canvas(canvas)
        toolbar = NavigationToolbar(canvas, self._viz_tab)
        self._fix_toolbar_fonts(toolbar)
        self._reconnect_3d_axes_events(fig, canvas)
        return {'fig': fig, 'canvas': canvas, 'toolbar': toolbar}

    def _show_viz_entry(self, entry):
        """Swap the entry's canvas/toolbar into the layout.

        The outgoing widgets are only hidden if they belong to a cached entry
        (they may be re-shown later); otherwise they are deleted."""
        if entry['canvas'] is self.viz_canvas:
            self.viz_canvas.draw_idle()
            return

        old_canvas, old_toolbar = self.viz_canvas, self.viz_toolbar

        idx = self._viz_layout.indexOf(old_toolbar)
        self._viz_layout.insertWidget(idx, entry['toolbar'])
        self._viz_layout.insertWidget(idx + 1, entry['canvas'])
        entry['toolbar'].show()
        entry['canvas'].show()

        self._viz_layout.removeWidget(old_toolbar)
        self._viz_layout.removeWidget(old_canvas)
        if any(e['canvas'] is old_canvas for e in self._viz_cache.values()):
            old_toolbar.hide()
            old_canvas.hide()
        else:
            old_toolbar.deleteLater()
            old_canvas.deleteLater()

        self.viz_figure = entry['fig']
        self.viz_canvas = entry['canvas']
        self.viz_toolbar = entry['toolbar']
        self.viz_canvas.draw()

    def _drop_viz_cache_entry(self, viz_type):
        """Remove one cache entry, deleting its widgets unless still displayed
        (a displayed one is deleted on the next _show_viz_entry swap)."""
        entry = self._viz_cache.pop(viz_type, None)
        if entry and entry['canvas'] is not self.viz_canvas:
            entry['toolbar'].deleteLater()
            entry['canvas'].deleteLater()

    def _clear_viz_cache(self):
        for viz_type in list(self._viz_cache):
            self._drop_viz_cache_entry(viz_type)

    def _reconnect_3d_axes_events(self, fig, canvas=None):
        """Reconnect Axes3D mouse event handlers to the embedded canvas.

        plt.figure() creates a figure with a temporary hidden canvas and Axes3D
        registers its _button_press/_button_release/_on_move callbacks on that
        original canvas.  After the figure moves onto a fresh embedded canvas
        those callbacks are still on the wrong canvas and 3-D rotation / zoom
        stops working.  mpl_connect deduplicates by function reference so
        calling this multiple times (e.g. on cache hit) is safe.
        """
        canvas = canvas if canvas is not None else self.viz_canvas
        for ax in fig.axes:
            if hasattr(ax, '_button_press'):
                canvas.mpl_connect('button_press_event',   ax._button_press)
                canvas.mpl_connect('button_release_event', ax._button_release)
                canvas.mpl_connect('motion_notify_event',  ax._on_move)
        # Scroll-wheel zoom (toolbar zoom button is disabled for 3D by matplotlib)
        canvas.mpl_connect('scroll_event', self._on_3d_scroll)

    def _configure_3d_axes_meter_scale(self, fig, equalize_limits=False):
        """Use 1 m ticks and the same physical scale on every 3-D axis."""
        from matplotlib.ticker import MultipleLocator

        for ax in fig.axes:
            if getattr(ax, 'name', None) != '3d':
                continue
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.yaxis.set_major_locator(MultipleLocator(1.0))
            ax.zaxis.set_major_locator(MultipleLocator(1.0))
            ax.tick_params(axis='both', labelsize=16)

            if equalize_limits:
                limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
                max_span = max(abs(high - low) for low, high in limits)
                max_span = max(max_span, 1.0)

                equal_limits = []
                for low, high in limits:
                    center = (low + high) / 2.0
                    half_span = max_span / 2.0
                    equal_limits.append((center - half_span, center + half_span))

                ax.set_xlim3d(*equal_limits[0])
                ax.set_ylim3d(*equal_limits[1])
                ax.set_zlim3d(*equal_limits[2])
                ax.set_box_aspect((1, 1, 1))

    def _on_3d_scroll(self, event):
        """Scroll-wheel zoom for the embedded 3D canvas.

        Uses Axes3D._zoom_data_limits() which scales the actual data-limit box.
        Scrolling up zooms in (scale < 1), scrolling down zooms out (scale > 1).
        The toolbar magnifying-glass button does nothing for 3D axes because
        Axes3D.can_zoom() returns False — scroll wheel is the replacement.
        """
        ax = event.inaxes
        if ax is None or not hasattr(ax, '_zoom_data_limits'):
            return
        scale = 1 / 1.12 if event.button == 'up' else 1.12
        ax._zoom_data_limits(scale, scale, scale)
        self.viz_canvas.draw_idle()

    def refresh_visualization(self):
        """Force redraw by clearing the cache for the current viz type, then reload."""
        viz_type = self.viz_type_combo.currentIndex()
        self._drop_viz_cache_entry(viz_type)
        self.load_visualization()

    def _load_calibration_context(self, verbose=True):
        """Collect the optimized values and the ID groups every 3D view needs.

        Prefers the in-memory calibration and falls back to
        optimization_results.pkl so the views still work in a fresh session.
        Shared by the embedded Matplotlib views and the Open3D window so both
        draw the same markers and cameras.

        Returns:
            dict with 'result', 'graph', 'detections', 'system' and the marker /
            camera ID groups, or None when no calibration data is available.
        """
        import pickle
        import gtsam
        from fgo_system import (create_uncertainty_system, build_apriltag_config,
                                observation_edges)

        def log(message):
            if verbose:
                self.append_log(message)

        result = graph = detections = None
        use_live_data = False

        # 1. Attempt to get from memory (running - latest data)
        if hasattr(self, 'calibration') and self.calibration and self.calibration.result:
            result = self.calibration.result
            graph = self.calibration.graph
            detections = self.calibration.detections
            use_live_data = True

        # 2. Read from pkl file (fallback - Environment only)
        else:
            pkl_file = os.path.join(self.project_root,
                                    getattr(self.config, 'save_file', "optimization_results.pkl"))
            if not os.path.exists(pkl_file):
                log("⚠  No calibration data found (optimization_results.pkl missing).\n")
                return None

            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, tuple):
                if len(data) < 3:
                    log("⚠  Invalid pkl format.\n")
                    return None
                graph = data[0]
                result = data[2]
                if len(data) >= 4:
                    detections = data[3]
            elif isinstance(data, dict):
                result = data.get('result')
                graph = data.get('graph')
                detections = data.get('detections')
            else:
                log(f"⚠  Unknown pkl format: {type(data)}\n")
                return None

        if result is None:
            log("⚠  No result data available.\n")
            return None

        # Extract marker/camera IDs from result.keys()
        marker_ids = []
        camera_ids = []
        wheel_camera_ids = []
        vehicle_camera_ids = []

        for key in result.keys():
            symbol = gtsam.Symbol(key)
            char = symbol.chr()
            index = symbol.index()

            if char == ord('M'):
                marker_ids.append(index)
            elif char == ord('C'):
                camera_ids.append(index)
                if 10000 <= index < 20000:
                    wheel_camera_ids.append(index)
                elif index >= 20000:
                    vehicle_camera_ids.append(index)

        # Get from Config if Vehicle cameras not in result
        if not vehicle_camera_ids:
            # When using live data: get from calibration object
            if use_live_data and hasattr(self.calibration, 'vehicle_camera_ids') and self.calibration.vehicle_camera_ids:
                vehicle_camera_ids = list(self.calibration.vehicle_camera_ids)
                log(f"   ✓ Vehicle camera IDs from calibration object: {vehicle_camera_ids}\n")
            # When using pkl file or also not in calibration: create from Config
            elif hasattr(self, 'config') and self.config.vehicle_camera_configs:
                num_vehicle_cameras = len(self.config.vehicle_camera_configs)
                vehicle_camera_ids = [20000 + i for i in range(num_vehicle_cameras)]
                log(f"   ✓ Vehicle camera IDs from config: {vehicle_camera_ids} ({num_vehicle_cameras} cameras)\n")
        else:
            log(f"   ✓ Vehicle camera IDs found in result: {vehicle_camera_ids}\n")

        # Debug: check if cameras actually exist in result
        if verbose:
            for vid in vehicle_camera_ids:
                key = gtsam.symbol('C', vid)
                exists = result.exists(key)
                self.append_log(f"      C{vid} exists in result: {exists}\n")
                if exists:
                    pose = result.atPose3(key)
                    pos = pose.translation()
                    self.append_log(f"      C{vid} position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n")

        # Create system using current GUI config
        system = create_uncertainty_system(
            camera_params=self.config.camera_params,
            apriltag_config=build_apriltag_config(self.config),
            marker_size=self.config.marker_size,
            marker_size2=self.config.marker_size2,
            image_directory=self.config.image_directory,
            wheel_directory=self.config.wheel_directory,
            dist_coeffs=self.config.dist_coeffs,
            wheel_camera_params=self.config.wheel_camera_params,
            wheel_dist_coeffs=self.config.wheel_dist_coeffs,
            marker_size_mapping=self.config.marker_size_mapping,
            first_marker_id=self.config.first_marker_id,
            vehicle_markers=self.config.vehicle_markers,
            image_extensions=self.config.image_extensions,
            vehicle_camera_configs=self.config.vehicle_camera_configs,
        )

        # Set detections data to system (for M2M connection lines)
        if detections:
            system.all_detections_for_viz = detections

        wheel_markers = set(self.config.vehicle_markers)

        # Camera→marker links actually optimized, from every stage's graph
        # (the pkl fallback only holds the Stage 1 graph).
        graphs = (getattr(self.calibration, 'observation_graphs', None)
                  if use_live_data else None) or [graph]

        return {
            'result': result,
            'graph': graph,
            'edges': observation_edges(graphs),
            'detections': detections,
            'system': system,
            'use_live_data': use_live_data,
            'marker_ids': marker_ids,
            'env_marker_ids': [m for m in marker_ids if m not in wheel_markers],
            'wheel_markers': sorted(wheel_markers),
            'camera_ids': camera_ids,
            'env_camera_ids': sorted(c for c in camera_ids if c < 1000),
            'wheel_camera_ids': sorted(wheel_camera_ids),
            'vehicle_camera_ids': vehicle_camera_ids,
        }

    def load_visualization(self):
        """Load and display latest visualization results (hybrid: memory first, pkl fallback)"""
        try:
            viz_type = self.viz_type_combo.currentIndex()

            # ── Cache hit: re-show the entry's own canvas/toolbar ──
            if viz_type in self._viz_cache:
                entry = self._viz_cache[viz_type]
                self._configure_3d_axes_meter_scale(entry['fig'])
                self._show_viz_entry(entry)
                self.append_log(f"✓ Visualization loaded from cache (type={viz_type})\n")
                return

            context = self._load_calibration_context()
            if context is None:
                # Draw the placeholder into a throwaway figure — clearing
                # self.viz_figure would poison a displayed CACHED entry,
                # making its next cache hit show "No data" instead of the
                # plot. Not cached → deleted automatically on next swap.
                placeholder = Figure(figsize=(10, 8))
                ax = placeholder.add_subplot(111)
                ax.text(0.5, 0.5, "No data found.\nRun calibration first.\n\nExpected: optimization_results.pkl",
                       ha='center', va='center', fontsize=12, color='gray')
                ax.axis('off')
                self._show_viz_entry(self._make_viz_entry(placeholder))
                return

            result = context['result']
            system = context['system']
            marker_ids = context['marker_ids']
            env_marker_ids = context['env_marker_ids']
            camera_ids = context['camera_ids']
            env_camera_ids = context['env_camera_ids']
            wheel_camera_ids = context['wheel_camera_ids']
            vehicle_camera_ids = context['vehicle_camera_ids']

            # Call appropriate visualization function based on viz_type
            temp_fig = None

            if viz_type == 2:  # Stage 3: Vehicle Camera Optimization (Vehicle-centric)
                temp_fig, _, _ = system.visualize_poses_3d_vehicle_centric(
                    result,
                    marker_ids=marker_ids,
                    camera_ids=vehicle_camera_ids,
                    save_path=None,
                )

            elif viz_type == 6:  # Stage 4: Stage 3 in pose-scene style, faint edges
                temp_fig = system.visualize_stage4_markers_and_cameras(
                    result,
                    marker_ids=marker_ids,
                    vehicle_camera_ids=vehicle_camera_ids,
                    edges=context['edges'],
                )
                self.append_log(f"   ✓ Stage 4 pose scene: {len(marker_ids)} markers, "
                                f"{len(vehicle_camera_ids)} vehicle cameras\n")
                _, _, ref_label = system.vehicle_scene_reference(result)
                if ref_label:
                    self.append_log(f"   Origin: {ref_label}\n")

            elif viz_type == 0:  # Stage 1: Environment Marker Optimization
                # Environment cameras (0-999) and markers with wheel markers excluded
                self.append_log(f"   Markers: {len(env_marker_ids)} "
                                f"(excluding wheel markers: {context['wheel_markers']})\n")
                temp_fig = system.visualize_poses_3d_optimized(
                    result,
                    marker_ids=env_marker_ids,
                    save_path=None,
                    plot_type=0
                )

            elif viz_type == 1:  # Stage 2: Wheel Marker Optimization
                temp_fig = system.visualize_poses_3d_optimized_wheel_only(
                    result,
                    marker_ids=marker_ids,
                    save_path=None
                )

            elif viz_type == 3:  # Marker Corner Point Cloud
                corner_pointcloud = system.create_marker_corner_pointcloud_from_result(
                    result, marker_ids
                )
                temp_fig = system.visualize_corner_pointcloud_3d(corner_pointcloud)
                self.append_log(f"   ✓ Corner point cloud: {len(marker_ids)} markers, "
                                f"{len(corner_pointcloud['corner_points'])} corner points\n")

            elif viz_type in (4, 5):  # Boards as oriented quads + camera frustums
                if viz_type == 4:
                    temp_fig = system.visualize_stage1_markers_and_cameras(
                        result, marker_ids=env_marker_ids, camera_ids=env_camera_ids,
                        edges=context['edges'])
                    self.append_log(f"   ✓ Stage 1 pose scene: {len(env_marker_ids)} markers, "
                                    f"{len(env_camera_ids)} env cameras\n")
                    if not env_camera_ids:
                        self.append_log("   ⚠  No C<1000 poses in the result — the environment "
                                        "cameras were dropped by a later stage. Re-run the "
                                        "calibration to get them carried over.\n")
                else:
                    temp_fig = system.visualize_stage2_markers_and_cameras(
                        result, marker_ids=marker_ids,
                        wheel_camera_ids=wheel_camera_ids,
                        env_camera_ids=env_camera_ids,
                        edges=context['edges'])
                    self.append_log(f"   ✓ Stage 2 pose scene: {len(marker_ids)} markers, "
                                    f"{len(wheel_camera_ids)} wheel cameras, "
                                    f"{len(env_camera_ids)} env cameras (context)\n")
                    _, _, ref_label = system.vehicle_scene_reference(result)
                    if ref_label:
                        self.append_log(f"   Origin: {ref_label}\n")
                    else:
                        self.append_log("   ⚠  Vehicle frame unavailable (wheel markers "
                                        "missing) — using the default reference marker.\n")
                    if not wheel_camera_ids:
                        self.append_log("   ⚠  No C10000~19999 poses in the result — Stage 2 "
                                        "either did not run or its cameras were dropped by "
                                        "Stage 3. Re-run the calibration.\n")

            # Set temp_fig directly to canvas
            if temp_fig:
                self._configure_3d_axes_meter_scale(temp_fig, equalize_limits=True)

                # One canvas/toolbar per figure, cached together and re-shown
                entry = self._make_viz_entry(temp_fig)
                self._viz_cache[viz_type] = entry
                self._show_viz_entry(entry)

                self.append_log(f"✓ Visualization loaded: {len(marker_ids)} markers, {len(camera_ids)} cameras\n")
            else:
                self.append_log("⚠  Failed to generate visualization\n")

        except Exception as e:
            self.append_log(f"⚠  Error loading visualization: {str(e)}\n")
            import traceback
            traceback.print_exc()
            # Make the failure visible on the canvas itself — the log tab is
            # hidden at this point, and an untouched canvas would present the
            # previous (stale) plot as if it were the fresh result.
            try:
                from matplotlib.figure import Figure as _Fig
                err_fig = _Fig(figsize=(10, 8))
                ax = err_fig.add_subplot(111)
                ax.text(0.5, 0.5,
                        f"⚠ Visualization failed to load:\n\n{e}\n\n"
                        "See the Calibration Log tab for the full traceback.",
                        ha='center', va='center', fontsize=11, color='firebrick',
                        wrap=True)
                ax.axis('off')
                self._show_viz_entry(self._make_viz_entry(err_fig))
            except Exception:
                pass   # never let the error display itself crash the GUI

    def show_pose_scene_open3d(self):
        """Open the selected Stage 1 / Stage 2 pose scene in an Open3D window.

        The same boards and frustums the Matplotlib views draw, rendered through
        OpenGL: Matplotlib redraws every edge in software on each mouse move, so
        a few hundred cameras make the embedded view unusable to rotate.
        """
        try:
            try:
                import open3d  # noqa: F401
            except ImportError:
                self.append_log("⚠  Open3D view skipped: open3d is not installed "
                                "(pip install open3d)\n")
                QMessageBox.warning(
                    self, "Open3D Not Installed",
                    "Open3D is required for the GPU-rendered pose scene.\n\n"
                    "Install it with:\n    pip install open3d\n\n"
                    "The Matplotlib views in this tab keep working meanwhile.")
                return

            context = self._load_calibration_context(verbose=False)
            if context is None:
                QMessageBox.warning(self, "No Data",
                                    "No calibration data found.\nPlease run calibration first.")
                return

            system = context['system']
            result = context['result']
            viz_type = self.viz_type_combo.currentIndex()

            # Only the pose-scene entries have an Open3D counterpart; any other
            # selection falls back to Stage 1 rather than doing nothing.
            ref_pose = ref_marker_id = None
            if viz_type in (5, 6):
                marker_ids = context['marker_ids']
                if viz_type == 5:
                    camera_groups = system.stage2_camera_groups(
                        context['wheel_camera_ids'], context['env_camera_ids'])
                    window_name = 'Camera Poses: wheel marker cameras'
                else:
                    camera_groups = system.stage4_camera_groups(
                        context['vehicle_camera_ids'])
                    window_name = 'Stage 4: vehicle camera poses'
                # Same origin the embedded Matplotlib view uses.
                ref_pose, ref_marker_id, ref_label = system.vehicle_scene_reference(result)
                if ref_label:
                    window_name += f' [origin: {ref_label}]'
            else:
                if viz_type != 4:
                    self.append_log(
                        "[viz] Open3D view covers the pose scenes only — showing Stage 1. "
                        "Select a 'Camera Poses' or Stage 4 entry to choose another.\n")
                marker_ids = context['env_marker_ids']
                camera_groups = system.stage1_camera_groups(context['env_camera_ids'])
                window_name = 'Camera Poses: env marker cameras'

            self.append_log(f"[viz] Opening Open3D window: {window_name} "
                            f"(the GUI waits until that window is closed)\n")
            QApplication.processEvents()

            scene = system.show_pose_scene_open3d(
                result, marker_ids=marker_ids, camera_groups=camera_groups,
                window_name=window_name, ref_pose=ref_pose,
                ref_marker_id=ref_marker_id, edges=context['edges'])

            board_count = sum(len(quads) for quads in scene['quads'].values())
            camera_counts = {name: len(poses) for name, poses in scene['cameras'].items()}
            total_cameras = sum(camera_counts.values())

            if not len(scene['points']):
                self.append_log("⚠  Open3D scene is empty: no M*/C* poses matched "
                                "the selected stage.\n")
                QMessageBox.information(
                    self, "Empty Scene",
                    "No markers or cameras were found for this stage.\n\n"
                    f"Markers in result: {len(context['marker_ids'])}\n"
                    f"Env cameras: {len(context['env_camera_ids'])}\n"
                    f"Wheel cameras: {len(context['wheel_camera_ids'])}")
                return

            detail = ", ".join(f"{name}: {count}" for name, count in camera_counts.items())
            self.append_log(f"   ✓ Open3D scene: {board_count} boards, "
                            f"{total_cameras} cameras ({detail})\n")

        except Exception as e:
            self.append_log(f"⚠  Open3D view error: {str(e)}\n")
            import traceback
            traceback.print_exc()

    def show_m2m_pose_graph(self):
        """Collect calibration data and open the M2M Pose Graph in a separate matplotlib window."""
        _figs_before = None
        try:
            import pickle
            import gtsam
            matplotlib.use('QtAgg')
            # Snapshot so a failure can close only the figures this call created.
            # A figure left registered but never drawn is what shows up later as
            # an empty white "Figure 1" window on the next plt.show().
            _figs_before = set(plt.get_fignums())

            result = None
            detections = None

            # Prefer live data from memory
            if hasattr(self, 'calibration') and self.calibration and self.calibration.result:
                result = self.calibration.result
                detections = self.calibration.detections
            else:
                # Fallback: load from pkl
                pkl_file = os.path.join(self.project_root, getattr(self.config, 'save_file', "optimization_results.pkl"))
                if not os.path.exists(pkl_file):
                    QMessageBox.warning(self, "No Data",
                                        "No calibration data found.\nPlease run calibration first.")
                    return
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, tuple) and len(data) >= 3:
                    result = data[2]
                    detections = data[3] if len(data) >= 4 else None
                elif isinstance(data, dict):
                    result = data.get('result')
                    detections = data.get('detections')

            if result is None or detections is None:
                QMessageBox.warning(self, "No Data",
                                    "Result or detection data is missing.\nPlease run calibration first.")
                return

            # min_pair_obs is a Stage-1/2 filter, so counting Stage-3 vehicle
            # images here would draw edges and colour "filtering targets" that
            # no stage ever evaluated.
            detections, n_vehicle_skipped = self._exclude_vehicle_detections(detections)
            if n_vehicle_skipped:
                self.append_log(
                    f"[viz] M2M Pose Graph: excluded {n_vehicle_skipped} "
                    f"vehicle-camera image(s) (env + wheel images only)\n"
                )

            # Collect marker IDs from result
            marker_ids = []
            for key in result.keys():
                sym = gtsam.Symbol(key)
                if sym.chr() == ord('M'):
                    marker_ids.append(sym.index())

            wheel_markers = set(self.config.vehicle_markers) if hasattr(self, 'config') else set()
            cfg_min_pair_obs = getattr(self.config, 'min_pair_obs', 2) if hasattr(self, 'config') else 2

            # ── Graph 1: Before outlier removal (show all edges) ─────────────
            fig_before = self._build_m2m_pose_graph_fig(
                result, detections, marker_ids, wheel_markers,
                title_tag="Before min_pair_obs Filter",
                min_pair_count=0,
                highlight_min_count=cfg_min_pair_obs
            )
            if fig_before is None:
                QMessageBox.warning(self, "No Data",
                                    "No M2M observations found in the detection data.")
                return

            # ── Graph 2: After min_pair_obs threshold filter ─────────────────
            fig_after = self._build_m2m_pose_graph_fig(
                result, detections, marker_ids, wheel_markers,
                title_tag=f"min_pair_obs≥{cfg_min_pair_obs} filter",
                min_pair_count=cfg_min_pair_obs
            )

            stats_columns, stats_rows = self._collect_m2m_edge_statistics_table(
                detections,
                detections,
                marker_ids,
                after_min_pair_count=cfg_min_pair_obs,
            )

            self.append_log(
                f"[viz] M2M Pose Graph: {len(detections)} images "
                f"(min_pair_obs≥{cfg_min_pair_obs} filter applied to edges only)\n"
            )
            # Show exactly the figures built above. A blanket plt.show() would
            # also pop any stray figure left over from an earlier failed call.
            for _fig in (fig_before, fig_after):
                if _fig is not None:
                    _fig.show()

            if stats_rows:
                # Non-modal so the two graph windows stay usable next to it;
                # the attribute keeps the dialog from being garbage collected.
                self._m2m_stats_dialog = M2MEdgeStatisticsDialog(
                    stats_columns, stats_rows, cfg_min_pair_obs, parent=self)
                self._m2m_stats_dialog.show()

        except Exception as e:
            if _figs_before is not None:
                try:
                    for _n in set(plt.get_fignums()) - _figs_before:
                        plt.close(_n)
                except Exception:
                    pass
            self.append_log(f"⚠  M2M Pose Graph error: {str(e)}\n")
            import traceback
            traceback.print_exc()

    def _exclude_vehicle_detections(self, detections):
        """Drop Stage-3 vehicle-camera images from a detection dict.

        Vehicle images never produce M2M edges in the solver: Stage 3 pins every
        marker with a Constrained prior and initializes each camera from the
        frozen map, so their co-observations carry no marker-to-marker
        information. Showing them in an M2M view only invents edges that no
        stage ever used.

        Returns:
            tuple: (filtered detections, number of images removed)
        """
        import re

        vehicle_keys = set()
        calib = getattr(self, 'calibration', None)
        if calib is not None and getattr(calib, 'vehicle_detections', None):
            vehicle_keys |= set(calib.vehicle_detections.keys())

        # The pkl-loaded path has no vehicle_detections dict to compare against,
        # so fall back to the Stage-3 key format built in fgo_system:
        # f"cam{cam_idx}_{filename}".
        vehicle_keys |= {k for k in detections if re.match(r'cam\d+_', str(k))}

        filtered = {k: v for k, v in detections.items() if k not in vehicle_keys}
        return filtered, len(detections) - len(filtered)

    def _collect_m2m_pair_poses(self, detections, marker_ids):
        """Collect consistently directed T_Mlo_to_Mhi observations by marker pair."""
        from collections import defaultdict

        marker_id_set = set(marker_ids)
        pair_poses = defaultdict(list)

        for det_list in detections.values():
            id_to_pose = {
                d['marker_id']: d['pose_se3']
                for d in det_list
                if d.get('marker_id') in marker_id_set and 'pose_se3' in d
            }
            ids = sorted(id_to_pose)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    lo, hi = ids[i], ids[j]
                    T_lo_to_hi = id_to_pose[lo].inverse().compose(id_to_pose[hi])
                    pair_poses[(lo, hi)].append(T_lo_to_hi)

        return pair_poses

    def _calculate_m2m_pose_statistics(self, poses):
        """Calculate an SE(3) mean and tangent-space variance for one M2M edge."""
        import gtsam

        if not poses:
            return None

        translations = np.asarray(
            [np.asarray(p.translation(), dtype=float).reshape(3) for p in poses]
        )
        translation_mean = np.mean(translations, axis=0)
        translation_lengths = np.linalg.norm(translations, axis=1)

        # Iterative intrinsic mean on SO(3). Residuals are rotation vectors in
        # the tangent space of the current mean, avoiding Euler-angle wrapping.
        rotation_mean = poses[0].rotation()
        for _ in range(50):
            rotation_residuals = np.asarray([
                gtsam.Rot3.Logmap(rotation_mean.inverse().compose(p.rotation()))
                for p in poses
            ])
            mean_step = np.mean(rotation_residuals, axis=0)
            if np.linalg.norm(mean_step) < 1e-10:
                break
            rotation_mean = rotation_mean.compose(gtsam.Rot3.Expmap(mean_step))

        rotation_residuals = np.asarray([
            gtsam.Rot3.Logmap(rotation_mean.inverse().compose(p.rotation()))
            for p in poses
        ])

        # Population covariance (ddof=0) keeps single-observation pairs at zero.
        translation_component_var = np.var(translations, axis=0, ddof=0)
        rotation_component_var_rad2 = np.var(rotation_residuals, axis=0, ddof=0)

        identity = gtsam.Rot3()
        rotation_mean_angle_deg = np.degrees(np.linalg.norm(
            gtsam.Rot3.Logmap(identity.inverse().compose(rotation_mean))
        ))

        return {
            'count': len(poses),
            'translation_mean': translation_mean,
            'translation_mean_norm_m': float(np.linalg.norm(translation_mean)),
            'translation_length_mean_m': float(np.mean(translation_lengths)),
            'translation_length_var_m2': float(
                np.var(translation_lengths, ddof=0)
            ),
            'translation_component_var_m2': translation_component_var,
            'translation_var_trace_m2': float(np.sum(translation_component_var)),
            'rotation_mean': rotation_mean,
            'rotation_mean_angle_deg': float(rotation_mean_angle_deg),
            'rotation_component_var_deg2': np.degrees(
                np.sqrt(rotation_component_var_rad2)
            ) ** 2,
            'rotation_var_trace_deg2': float(
                np.sum(rotation_component_var_rad2) * (180.0 / np.pi) ** 2
            ),
        }

    @staticmethod
    def _format_stat_value(value):
        """Readable fixed-point, falling back to scientific for tiny variances."""
        if value == 0:
            return "0"
        if abs(value) < 1e-3:
            return f"{value:.2e}"
        return f"{value:.4f}"

    def _collect_m2m_edge_statistics_table(
            self, before_detections, after_detections, marker_ids,
            after_min_pair_count=0):
        """Build the before/after M2M statistics as (columns, rows).

        Each cell is a (display_text, sort_key) pair so the table can sort
        numerically while still showing '-' for pairs the filter dropped.
        """
        before_poses = self._collect_m2m_pair_poses(before_detections, marker_ids)
        after_poses = self._collect_m2m_pair_poses(after_detections, marker_ids)
        if after_min_pair_count > 1:
            # Mirror the after-graph's min_pair_obs filter so the After
            # columns show what the optimiser actually keeps (dropped pairs
            # show 0 / '-').
            after_poses = {pair: poses for pair, poses in after_poses.items()
                           if len(poses) >= after_min_pair_count}
        all_pairs = sorted(set(before_poses) | set(after_poses))
        if not all_pairs:
            return None, None

        columns = [
            'M2M edge',
            'Before N', 'Before mean |t| (m)', 'Before var |t| (m²)',
            'Before mean R (°)', 'Before var R (°²)',
            'After N', 'After mean |t| (m)', 'After var |t| (m²)',
            'After mean R (°)', 'After var R (°²)',
        ]
        rows = []

        for lo, hi in all_pairs:
            before = self._calculate_m2m_pose_statistics(
                before_poses.get((lo, hi), [])
            )
            after = self._calculate_m2m_pose_statistics(
                after_poses.get((lo, hi), [])
            )

            def value(stats, key):
                if stats is None:
                    return ('-', float('-inf'))
                number = float(stats[key])
                return (self._format_stat_value(number), number)

            def count(stats):
                number = stats['count'] if stats is not None else 0
                return (str(number), float(number))

            rows.append([
                (f'M{lo}→M{hi}', lo * 1e6 + hi),
                count(before),
                value(before, 'translation_length_mean_m'),
                value(before, 'translation_length_var_m2'),
                value(before, 'rotation_mean_angle_deg'),
                value(before, 'rotation_var_trace_deg2'),
                count(after),
                value(after, 'translation_length_mean_m'),
                value(after, 'translation_length_var_m2'),
                value(after, 'rotation_mean_angle_deg'),
                value(after, 'rotation_var_trace_deg2'),
            ])

        return columns, rows

    def show_m2m_pair_diagnostic(self):
        """Show detailed diagnostic plots for a specific M2M pair (user-specified).

        Displays:
          1. Translation scatter (XYZ) for all observations.
          2. Rotation angle from median pose per observation.
          3. Per-observation C2M uncertainty (rotation and translation).
          4. Summary table: camera filename, translation, rotation angle from median.
        """
        _figs_before = None
        try:
            import pickle
            import gtsam
            import math
            from fgo_system import median_pose3
            matplotlib.use('QtAgg')
            _figs_before = set(plt.get_fignums())

            # ── Ask user for marker pair ──────────────────────────────────────
            text, ok = QInputDialog.getText(
                self, "M2M Pair Diagnostic",
                "Enter marker pair (e.g.  10 57  or  10,57):"
            )
            if not ok or not text.strip():
                return

            parts = text.replace(',', ' ').split()
            if len(parts) != 2:
                QMessageBox.warning(self, "Input Error", "Please enter exactly two marker IDs.")
                return
            try:
                ma, mb = int(parts[0]), int(parts[1])
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Marker IDs must be integers.")
                return
            lo, hi = min(ma, mb), max(ma, mb)

            # ── Load data ────────────────────────────────────────────────────
            detections = None
            if hasattr(self, 'calibration') and self.calibration and self.calibration.result:
                detections = self.calibration.detections
            else:
                pkl_file = os.path.join(self.project_root, getattr(self.config, 'save_file', "optimization_results.pkl"))
                if not os.path.exists(pkl_file):
                    QMessageBox.warning(self, "No Data", "No calibration data found.")
                    return
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, tuple) and len(data) >= 3:
                    detections = data[3] if len(data) >= 4 else None
                elif isinstance(data, dict):
                    detections = data.get('detections')

            if detections is None:
                QMessageBox.warning(self, "No Data", "Detection data is missing.")
                return

            # Stage-3 vehicle cameras contribute no M2M constraint, so keep them
            # out of this diagnostic entirely (env + wheel images only).
            detections, n_vehicle_skipped = self._exclude_vehicle_detections(detections)
            if n_vehicle_skipped:
                self.append_log(
                    f"[m2m-diag] excluded {n_vehicle_skipped} vehicle-camera image(s) "
                    f"(env + wheel images only)\n"
                )

            # ── Collect observations for this pair ───────────────────────────
            obs_poses   = []   # list of gtsam.Pose3 (T_lo_to_hi)
            obs_fnames  = []   # list of filename strings

            for fn, det_list in detections.items():
                id_to_pose = {}
                for d in det_list:
                    mid = d['marker_id']
                    if 'pose_se3' in d:
                        id_to_pose[mid] = d['pose_se3']
                if lo not in id_to_pose or hi not in id_to_pose:
                    continue
                T_cam_lo = id_to_pose[lo]
                T_cam_hi = id_to_pose[hi]
                T_lo_to_hi = T_cam_lo.inverse().compose(T_cam_hi)
                obs_poses.append(T_lo_to_hi)
                obs_fnames.append(fn)

            n = len(obs_poses)
            if n == 0:
                QMessageBox.information(self, "No Data",
                                        f"No co-observations found for M{lo}-M{hi}.")
                return

            # ── Translation arrays ───────────────────────────────────────────
            trans = np.array([np.array(obs_poses[i].translation()).flatten() for i in range(n)], dtype=float)

            # ── Representative pose (median) and rotation angles ─────────────
            rep_pose = median_pose3(obs_poses)
            rot_angles_deg = []
            for p in obs_poses:
                if rep_pose is not None:
                    R_rel = rep_pose.rotation().inverse().compose(p.rotation())
                    angle = float(np.linalg.norm(gtsam.Rot3.Logmap(R_rel))) * 180.0 / math.pi
                else:
                    angle = 0.0
                rot_angles_deg.append(angle)

            # ── Build text summary ───────────────────────────────────────────
            summary_lines = [
                f"M{lo}-M{hi}  |  {n} observations",
                "",
                f"{'#':>3}  {'tx':>8}  {'ty':>8}  {'tz':>8}  {'RotDeg':>7}  Filename",
            ]
            for i in range(n):
                tx, ty, tz = trans[i]
                short_fn = os.path.basename(obs_fnames[i])
                summary_lines.append(
                    f"{i:>3}  {tx:>8.3f}  {ty:>8.3f}  {tz:>8.3f}  "
                    f"{rot_angles_deg[i]:>7.2f}°  {short_fn}"
                )

            summary_text = "\n".join(summary_lines)
            self.append_log(f"\n{'=' * 60}\n{summary_text}\n{'=' * 60}\n")

            # ── Per-observation uncertainty extraction ────────────────────────
            # For each co-observation image, extract C2M uncertainty of M_lo and M_hi
            RAD2DEG = 180.0 / math.pi
            CM      = 100.0  # m → cm

            unc_lo_rot_deg   = []  # mean rotation σ of M_lo (deg)
            unc_lo_trans_cm  = []  # mean translation σ of M_lo (cm)
            unc_hi_rot_deg   = []
            unc_hi_trans_cm  = []

            for fn in obs_fnames:
                det_list = detections.get(fn, [])
                id_to_det = {d['marker_id']: d for d in det_list}

                # An absent covariance is not a zero covariance. NaN keeps those
                # bars off the plot; a 0 would read as a confidently measured zero.
                def _extract_sigma(mid):
                    det = id_to_det.get(mid)
                    if det is None:
                        return math.nan, math.nan
                    cov = det.get('uncertainty_matrix')
                    if cov is None:
                        return math.nan, math.nan
                    # uncertainty_matrix order: [pos(0-2), rot(3-5)]
                    # sqrt(trace) = L2 norm of uncertainty vector → preserves large values
                    sigma_trans = float(np.sqrt(np.trace(np.abs(cov[:3, :3]))))
                    sigma_rot   = float(np.sqrt(np.trace(np.abs(cov[3:, 3:]))))
                    return sigma_rot, sigma_trans

                r_lo, t_lo = _extract_sigma(lo)
                r_hi, t_hi = _extract_sigma(hi)
                unc_lo_rot_deg.append(r_lo * RAD2DEG)
                unc_lo_trans_cm.append(t_lo * CM)
                unc_hi_rot_deg.append(r_hi * RAD2DEG)
                unc_hi_trans_cm.append(t_hi * CM)

            unc_lo_rot_deg  = np.array(unc_lo_rot_deg)
            unc_lo_trans_cm = np.array(unc_lo_trans_cm)
            unc_hi_rot_deg  = np.array(unc_hi_rot_deg)
            unc_hi_trans_cm = np.array(unc_hi_trans_cm)

            # ── Plotting ─────────────────────────────────────────────────────
            fig = plt.figure(figsize=(14, 9))
            fig.suptitle(f"M{lo}–M{hi} Pair Diagnostic  (n={n})", fontsize=14, fontweight='bold')

            COLOR_ALL = 'steelblue'
            point_colors = [COLOR_ALL] * n

            idx_arr = np.arange(n)

            # ── Roll / Pitch extraction ───────────────────────────────────────
            from scipy.spatial.transform import Rotation as _Rot
            rolls_deg   = []
            pitches_deg = []
            for p in obs_poses:
                R_mat = p.rotation().matrix()
                r_obj = _Rot.from_matrix(R_mat)
                yaw_p, pitch_p, roll_p = r_obj.as_euler('ZYX', degrees=True)
                rolls_deg.append(roll_p)
                pitches_deg.append(pitch_p)
            rolls_deg   = np.array(rolls_deg)
            pitches_deg = np.array(pitches_deg)

            # layout: 2 rows x 2 cols
            gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

            # ── Panel 1: Translation X-Y scatter ─────────────────────────────
            ax1 = fig.add_subplot(gs[0, 0])
            for i in range(n):
                ax1.scatter(trans[i, 0], trans[i, 1], color=point_colors[i],
                            s=60, zorder=3)
                ax1.annotate(str(i), (trans[i, 0], trans[i, 1]),
                             fontsize=7, ha='left', va='bottom')
            ax1.set_title('Translation X-Y  (M2M)')
            ax1.set_xlabel('tx (m)')
            ax1.set_ylabel('ty (m)')
            ax1.grid(True, alpha=0.3)

            # ── Panel 2: Roll-Pitch scatter ───────────────────────────────────
            ax2 = fig.add_subplot(gs[0, 1])
            for i in range(n):
                ax2.scatter(rolls_deg[i], pitches_deg[i], color=point_colors[i],
                            s=60, zorder=3)
                ax2.annotate(str(i), (rolls_deg[i], pitches_deg[i]),
                             fontsize=7, ha='left', va='bottom')
            ax2.set_title('Roll-Pitch  (M2M rotation)')
            ax2.set_xlabel('Roll (°)')
            ax2.set_ylabel('Pitch (°)')
            ax2.grid(True, alpha=0.3)

            # ── Panel 3: Rotation angle from representative pose ──────────────
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.bar(idx_arr, rot_angles_deg, color=point_colors, edgecolor='none')
            ax3.set_title('Rotation Angle from Median Pose (°)')
            ax3.set_xlabel('Obs index')
            ax3.set_ylabel('Angle (°)')

            # ── Panel 4: Translation Z per observation ────────────────────────
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.bar(idx_arr, trans[:, 2], color=point_colors, edgecolor='none')
            ax4.set_title('Translation Z per Observation')
            ax4.set_xlabel('Obs index')
            ax4.set_ylabel('tz (m)')

            plt.tight_layout()
            fig.show()

            # ── Figure 2: C2M Uncertainty (2x2) ─────────────────────────────
            fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
            fig2.suptitle(f"C2M Uncertainty  M{lo} & M{hi}  (n={n})",
                          fontsize=13, fontweight='bold')

            unc_data = [
                # (row, col, marker_id, values, ylabel, title)
                (0, 0, lo, unc_lo_rot_deg,   '√trace(Σ_rot) (°)',   f'M{lo}  Rotation Uncertainty  √trace(Σ_rot) [°]'),
                (0, 1, hi, unc_hi_rot_deg,   '√trace(Σ_rot) (°)',   f'M{hi}  Rotation Uncertainty  √trace(Σ_rot) [°]'),
                (1, 0, lo, unc_lo_trans_cm,  '√trace(Σ_trans) (cm)',f'M{lo}  Translation Uncertainty  √trace(Σ_trans) [cm]'),
                (1, 1, hi, unc_hi_trans_cm,  '√trace(Σ_trans) (cm)',f'M{hi}  Translation Uncertainty  √trace(Σ_trans) [cm]'),
            ]

            for row, col, mid, values, ylabel, title in unc_data:
                ax = axes2[row, col]
                ax.bar(idx_arr, values, color=point_colors, edgecolor='none', width=0.7)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Obs index')
                ax.set_ylabel(ylabel)
                ax.grid(True, axis='y', alpha=0.3)
                if not np.any(np.isfinite(values)):
                    # Nothing to autoscale to, so give the empty axes a readable
                    # range and say outright why it is empty.
                    ax.set_xlim(-0.5, max(n - 0.5, 0.5))
                    ax.set_ylim(0.0, 1.0)
                    ax.text(0.5, 0.5, 'covariance not available',
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=11, color='#b03030',
                            bbox=dict(boxstyle='round', facecolor='#fff3cd',
                                      edgecolor='#d0b060'))

            fig2.tight_layout()
            fig2.show()

            # ── Figure 3: Obs index → image name lookup table ────────────────
            self._show_obs_index_table(
                obs_fnames, trans, rot_angles_deg,
                title=f"M{lo}–M{hi}  Observation Index → Image Name  (n={n})"
            )

        except Exception as e:
            if _figs_before is not None:
                try:
                    for _n in set(plt.get_fignums()) - _figs_before:
                        plt.close(_n)
                except Exception:
                    pass
            self.append_log(f"⚠  M2M Pair Diagnostic error: {str(e)}\n")
            import traceback
            traceback.print_exc()

    def _show_obs_index_table(self, obs_fnames, trans, rot_angles_deg,
                              title="Observation Index → Image Name",
                              rows_per_block=25, max_blocks=3):
        """Show legend figure(s) mapping the plot's observation indices to image names.

        The scatter/bar panels can only carry the integer index, so this table is
        the only place the actual filename appears. Rows are split into
        side-by-side blocks, and once more than ``rows_per_block * max_blocks``
        rows are needed the list continues on further figures — squeezing more
        columns into one figure makes the filenames unreadable.
        """
        import math

        n = len(obs_fnames)
        if n == 0:
            return []

        col_labels = ['#', 'Image name', 'tx (m)', 'ty (m)', 'tz (m)', 'Rot (°)']
        col_widths = [0.07, 0.50, 0.11, 0.11, 0.11, 0.10]

        per_page = rows_per_block * max_blocks
        n_pages  = int(math.ceil(n / float(per_page)))
        figs = []

        for page in range(n_pages):
            p0 = page * per_page
            p1 = min(p0 + per_page, n)
            page_n = p1 - p0

            n_blocks   = int(math.ceil(page_n / float(rows_per_block)))
            block_rows = int(math.ceil(page_n / float(n_blocks)))

            fig = plt.figure(figsize=(8.0 * n_blocks,
                                      max(1.8, 0.26 * (block_rows + 1) + 0.9)))
            page_tag = f"   [{page + 1}/{n_pages}]" if n_pages > 1 else ""
            fig.suptitle(title + page_tag, fontsize=13, fontweight='bold')
            gs = fig.add_gridspec(1, n_blocks, wspace=0.08,
                                  left=0.02, right=0.98, top=0.88, bottom=0.03)

            for b in range(n_blocks):
                i0 = p0 + b * block_rows
                i1 = min(i0 + block_rows, p1)
                ax = fig.add_subplot(gs[0, b])
                ax.axis('off')

                cell_text = []
                for i in range(i0, i1):
                    cell_text.append([
                        str(i),
                        os.path.basename(obs_fnames[i]),
                        f"{trans[i, 0]:.3f}",
                        f"{trans[i, 1]:.3f}",
                        f"{trans[i, 2]:.3f}",
                        f"{rot_angles_deg[i]:.2f}",
                    ])

                # An explicit bbox keeps every block's row pitch identical instead
                # of letting the last (shorter) block stretch its rows.
                frac = (i1 - i0 + 1) / float(block_rows + 1)
                table = ax.table(cellText=cell_text, colLabels=col_labels,
                                 colWidths=col_widths, cellLoc='left',
                                 bbox=[0.0, 1.0 - frac, 1.0, frac])
                table.auto_set_font_size(False)
                table.set_fontsize(8)

                for (row, col), cell in table.get_celld().items():
                    cell.set_edgecolor('#cccccc')
                    if row == 0:
                        cell.set_facecolor('#34495e')
                        cell.set_text_props(color='white', fontweight='bold')
                    elif row % 2 == 0:
                        cell.set_facecolor('#f4f6f7')
                    if col != 1:
                        cell.set_text_props(ha='right' if row > 0 else 'center')

            fig.show()
            figs.append(fig)

        return figs

    def _build_m2m_pose_graph_fig(self, result, detections, marker_ids, wheel_markers,
                                   title_tag="", min_pair_count=0,
                                   highlight_min_count=0):
        """Build a 3D directed graph showing raw M2M pose measurements vs optimized nodes.

        For every image that detects ≥2 markers, each co-detected pair (mi, mj) with mi < mj
        yields one raw T_mi_to_mj measurement. The edge is drawn as:

            start : optimized world position of mi  (T_W_Mi.translation())
            end   : T_W_Mi ⊕ T_mi_to_mj  (where camera *thinks* mj is, given mi's pose)

        where  T_mi_to_mj = T_cam_mi⁻¹ · T_cam_mj  (computed from raw detections).

        If the same pair (mi, mj) appears in N images, N independent edges are drawn.
        Each edge endpoint should ideally land on mj's optimized node — deviation shows
        raw detection noise / inconsistency.

        Edge colours (by pair type):
            env-env    → steelblue
            env-wheel  → mediumseagreen
            wheel-wheel → darkorange
            crimson    → min_pair_obs filtering target: either an endpoint
                         marker was dropped from the optimization entirely, or
                         (with highlight_min_count > 0, used for the "before"
                         graph) the pair's observation count is below the
                         threshold and the filter will discard it.
        """
        import gtsam
        from collections import defaultdict

        if result is None or detections is None:
            return None

        wheel_set = set(wheel_markers)

        # ── 1. Collect optimized full Pose3 for each marker ───────────────────
        #       (need full pose, not just translation, to transform T_mi_to_mj)
        marker_pose_opt = {}   # marker_id → gtsam.Pose3  (T_W_M)
        marker_pos_opt  = {}   # marker_id → np.array([x, y, z])
        for mid in marker_ids:
            key = gtsam.symbol('M', mid)
            if result.exists(key):
                p = result.atPose3(key)
                marker_pose_opt[mid] = p
                t = p.translation()
                marker_pos_opt[mid] = np.array([t[0], t[1], t[2]])

        # ── 2. For each image compute every raw T_mi_to_mj (mi < mj) ─────────
        #       raw_edges[(mi, mj)] = list of np.array end-points in world frame
        #
        #  Pass 1: count total observations per pair across all images so we can
        #          honour min_pair_count BEFORE collecting endpoints.
        pair_total_count = defaultdict(int)
        img_id_maps = {}  # filename → {marker_id: pose_se3}
        for filename, det_list in detections.items():
            id_to_pose = {}
            for d in det_list:
                mid = d['marker_id']
                # Keep every detected marker — including ones filtered out before
                # optimization (e.g. dropped by min_pair_obs). Their pairs can
                # still be drawn anchored at an optimized partner, so the
                # "before filter" graph shows the true pre-filter state.
                if 'pose_se3' in d:
                    id_to_pose[mid] = d['pose_se3']
            img_id_maps[filename] = id_to_pose
            ids = sorted(id_to_pose.keys())
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    pair_total_count[(ids[a], ids[b])] += 1

        #  Pass 2: collect endpoints — skip pairs below the count threshold
        raw_edges = defaultdict(list)   # (mi, mj) → [p_mj_estimated, ...]
        for filename, id_to_pose in img_id_maps.items():
            ids = sorted(id_to_pose.keys())
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    mi, mj = ids[a], ids[b]   # mi < mj guaranteed by sort

                    # Skip pairs that are below the minimum observation threshold
                    if min_pair_count > 0 and pair_total_count[(mi, mj)] < min_pair_count:
                        continue

                    T_cam_mi = id_to_pose[mi]
                    T_cam_mj = id_to_pose[mj]

                    # Raw relative pose: T_mi_to_mj (in mi's frame)
                    T_mi_to_mj = T_cam_mi.inverse().compose(T_cam_mj)

                    # Anchor at whichever side has an optimized pose; a marker
                    # missing from the result is drawn as the estimated end.
                    if mi in marker_pose_opt:
                        anchor = mi
                        T_est = marker_pose_opt[mi].compose(T_mi_to_mj)
                    elif mj in marker_pose_opt:
                        anchor = mj
                        T_est = marker_pose_opt[mj].compose(T_mi_to_mj.inverse())
                    else:
                        continue      # neither side optimized — nothing to anchor
                    t = T_est.translation()
                    raw_edges[(mi, mj)].append((anchor, np.array([t[0], t[1], t[2]])))

        if not raw_edges:
            return None

        # ── 3. Build figure ────────────────────────────────────────────────────
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        drawn_etypes = set()
        total_edges = 0

        excluded_endpoints = defaultdict(list)   # markers with no optimized node
        for (mi, mj), endpoints in sorted(raw_edges.items()):
            # edge colour by pair type
            mi_wheel = mi in wheel_set
            mj_wheel = mj in wheel_set
            if mi not in marker_pos_opt or mj not in marker_pos_opt:
                # one side was filtered out before optimization
                color, etype = 'crimson', 'min_pair_obs filtering target'
            elif highlight_min_count > 0 and \
                    pair_total_count[(mi, mj)] < highlight_min_count:
                # pair seen fewer times than min_pair_obs — the filter will
                # drop it even though both markers survive via other pairs
                color, etype = 'crimson', 'min_pair_obs filtering target'
            elif mi_wheel and mj_wheel:
                color, etype = 'darkorange',     'wheel-wheel'
            elif not mi_wheel and not mj_wheel:
                color, etype = 'steelblue',      'env-env'
            else:
                color, etype = 'mediumseagreen', 'env-wheel'

            excl_mid = (mi if mi not in marker_pos_opt
                        else (mj if mj not in marker_pos_opt else None))

            for anchor, p_end in endpoints:
                p_start = marker_pos_opt[anchor]
                if excl_mid is not None:
                    excluded_endpoints[excl_mid].append(p_end)
                lbl = etype if etype not in drawn_etypes else ""
                drawn_etypes.add(etype)

                # Draw raw-measurement edge
                ax.plot3D([p_start[0], p_end[0]],
                          [p_start[1], p_end[1]],
                          [p_start[2], p_end[2]],
                          color=color, linewidth=0.9, alpha=0.45, label=lbl)

                # Arrowhead at 75 % along the edge (toward p_end)
                frac = 0.75
                arrow_origin = p_start + frac * (p_end - p_start)
                direction = p_end - p_start
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    uv = direction / dist
                    arrow_len = dist * 0.12
                    ax.quiver(arrow_origin[0], arrow_origin[1], arrow_origin[2],
                              uv[0] * arrow_len, uv[1] * arrow_len, uv[2] * arrow_len,
                              color=color, alpha=0.7,
                              arrow_length_ratio=0.5, linewidth=0.8)

                total_edges += 1

        # ── 4. Draw optimized marker nodes ────────────────────────────────────
        env_plotted = False
        wheel_plotted = False
        for mid, pos in sorted(marker_pos_opt.items()):
            is_wheel = mid in wheel_set
            c = 'darkorange' if is_wheel else 'royalblue'
            first_wheel = is_wheel and not wheel_plotted
            first_env   = not is_wheel and not env_plotted
            node_lbl = ('Wheel marker (opt.)' if is_wheel else 'Env marker (opt.)') \
                       if first_wheel or first_env else ""
            if is_wheel:
                wheel_plotted = True
            else:
                env_plotted = True

            ax.scatter(*pos, c=c, s=80, zorder=6, label=node_lbl,
                       edgecolors='white', linewidths=0.8)
            ax.text(pos[0], pos[1], pos[2], f' M{mid}',
                    fontsize=8, fontweight='bold', color=c)

        # ── 4.5 Filtered-out markers: label at their raw-pair estimate ────────
        for mid, pts in sorted(excluded_endpoints.items()):
            center = np.mean(np.asarray(pts), axis=0)
            ax.text(center[0], center[1], center[2], f' M{mid} (filtered)',
                    fontsize=8, fontweight='bold', color='crimson')

        # ── 5. Decorations ────────────────────────────────────────────────────
        n_pairs = len(raw_edges)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        tag_line = f'\n[{title_tag}]' if title_tag else ''
        ax.set_title(
            f'M2M Raw Pose Observations{tag_line}\n'
            f'{n_pairs} unique pairs · {total_edges} total edges\n'
            f'(edge end ≈ optimized node  →  good detection)'
        )

        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l and l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), loc='upper left', fontsize=9)

        fig.tight_layout()
        return fig

    def on_tab_changed(self, index):
        """Method called on tab switch"""
        try:
            # Get tab text to identify which tab was selected
            tab_text = self.right_tab_widget.tabText(index)

            # Auto load when switching to Projection tab
            if "Projection" in tab_text:
                self.append_log("[gui] Projection tab selected. Loading data...\n")
                self.load_projection_results()
        except Exception as e:
            self.append_log(f"⚠ Tab change error: {str(e)}\n")

    def load_projection_results(self):
        """Display projection results and poses in vehicle frame for vehicle cameras"""
        try:
            # Clear previous content
            while self.projection_content_layout.count() > 1:  # Keep the stretch at the end
                item = self.projection_content_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            # Check if calibration is available
            if hasattr(self, 'calibration'):
                self.append_log(f"   - Calibration is not None: {self.calibration is not None}\n")
                if self.calibration is not None:
                    self.append_log(f"   - Has vehicle_camera_ids: {hasattr(self.calibration, 'vehicle_camera_ids')}\n")
                    if hasattr(self.calibration, 'vehicle_camera_ids'):
                        self.append_log(f"   - Vehicle camera IDs: {self.calibration.vehicle_camera_ids}\n")

            if not hasattr(self, 'calibration') or self.calibration is None:
                no_data_label = QLabel("⚠ No calibration data available.\n\nPlease run Extrinsic Calibration first.")
                no_data_label.setAlignment(Qt.AlignCenter)
                no_data_label.setStyleSheet("font-size: 14px; color: #666; padding: 50px;")
                self.projection_content_layout.insertWidget(0, no_data_label)
                self.append_log("⚠ Projection: No calibration data available\n")
                return

            # Check if vehicle cameras exist
            if not hasattr(self.calibration, 'vehicle_camera_ids') or not self.calibration.vehicle_camera_ids:
                no_data_label = QLabel("⚠ No vehicle camera data available.\n\nPlease run Extrinsic Calibration to add vehicle cameras.")
                no_data_label.setAlignment(Qt.AlignCenter)
                no_data_label.setStyleSheet("font-size: 14px; color: #666; padding: 50px;")
                self.projection_content_layout.insertWidget(0, no_data_label)
                self.append_log("⚠ Projection: No vehicle cameras found\n")
                return

            self.append_log("\n[viz] Loading projection results...\n")

            import gtsam

            result = self.calibration.result
            vehicle_camera_ids = self.calibration.vehicle_camera_ids

            # Check if vehicle coordinate frame is cached
            if not hasattr(self.calibration.system, 'cached_vehicle_frame') or self.calibration.system.cached_vehicle_frame is None:
                self.append_log("⚠ Vehicle coordinate frame not cached. Computing...\n")
                self.calibration.system.add_vehicle_coordinate_system_post_optimization(result)

            vehicle_frame = self.calibration.system.cached_vehicle_frame
            vehicle_center = vehicle_frame['center']
            R_world_to_vehicle = vehicle_frame['R_world_to_vehicle']

            self.append_log(f"✓ Vehicle frame center: [{vehicle_center[0]:.3f}, {vehicle_center[1]:.3f}, {vehicle_center[2]:.3f}]\n")

            projection_images = self._generate_projection_images(result, vehicle_camera_ids)

            # ── Display results for each camera ──────────────────────────────
            for camera_id in sorted(vehicle_camera_ids):
                camera_key = gtsam.symbol('C', camera_id)

                if not result.exists(camera_key):
                    self.append_log(f"⚠ Cam{camera_id - 20000} not found in result\n")
                    continue

                pose_matrix = self._camera_pose_in_vehicle_frame(
                    result.atPose3(camera_key), vehicle_center, R_world_to_vehicle)
                camera_widget = self._build_projection_camera_widget(
                    camera_id, pose_matrix, projection_images)

                self.projection_content_layout.insertWidget(self.projection_content_layout.count() - 1, camera_widget)

                self.append_log(f"✓ Loaded projection for Cam{camera_id - 20000}\n")

            self.append_log(f"✓ Projection results loaded for {len(vehicle_camera_ids)} cameras\n")

        except Exception as e:
            self.append_log(f"⚠  Error loading projection results: {str(e)}\n")
            import traceback
            traceback.print_exc()

    def _compute_vehicle_reprojection_errors(self, result, vehicle_camera_ids):
        """Compute per-camera, per-board reprojection errors for vehicle cameras."""
        import gtsam

        if not (hasattr(self.calibration, 'vehicle_detections')
                and self.calibration.vehicle_detections):
            return None

        marker_ids = []
        for key in result.keys():
            symbol = gtsam.Symbol(key)
            if chr(symbol.chr()) == 'M':
                marker_ids.append(symbol.index())

        corner_pointcloud = self.calibration.system.create_marker_corner_pointcloud_from_result(
            result, marker_ids
        )

        camera_params_mapping = {}
        for vid in vehicle_camera_ids:
            vidx = vid - 20000
            if vidx < len(self.config.vehicle_camera_configs):
                vcfg = self.config.vehicle_camera_configs[vidx]
                camera_params_mapping[vid] = {
                    'intrinsic':     vcfg['intrinsic'],
                    'dist_coeffs':   vcfg['dist_coeffs'],
                    'camera_model':  vcfg.get('camera_model', 'pinhole')
                }

        return self.calibration.system.compute_reprojection_errors_with_corner_pointcloud(
            result,
            self.calibration.vehicle_detections,
            corner_pointcloud,
            camera_params_mapping,
        )

    def _compute_relative_camera_poses(self, result, vehicle_camera_ids, reference_camera_id=None):
        """Camera-to-camera poses expressed in the reference camera's frame.

        These are independent of the vehicle coordinate frame, so marker jitter
        that shifts the vehicle origin does not move them.

        Returns (reference_label, {label: pose_info}) or (None, {}).
        """
        import gtsam
        from scipy.spatial.transform import Rotation as R

        available = [cid for cid in sorted(vehicle_camera_ids)
                     if result.exists(gtsam.symbol('C', cid))]
        if not available:
            return None, {}

        if reference_camera_id is None or reference_camera_id not in available:
            # Tool's "cam0" is vehicle camera index 0 (id 20000); fall back to the
            # lowest available index when cam0 was not calibrated.
            reference_camera_id = 20000 if 20000 in available else available[0]

        T_world_ref = result.atPose3(gtsam.symbol('C', reference_camera_id))
        T_ref_world = T_world_ref.inverse()

        relative_poses = {}
        for camera_id in available:
            T_ref_cam = T_ref_world.compose(result.atPose3(gtsam.symbol('C', camera_id)))

            rotation = T_ref_cam.rotation().matrix()
            translation = T_ref_cam.translation()

            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation
            pose_matrix[:3, 3] = translation

            yaw, pitch, roll = R.from_matrix(rotation).as_euler('ZYX', degrees=True)

            relative_poses[f"Cam{camera_id - 20000}"] = {
                "xyzrpy": [float(translation[0]), float(translation[1]), float(translation[2]),
                           float(roll), float(pitch), float(yaw)],
                "matrix": pose_matrix.tolist(),
                "translation": translation.tolist(),
                "rotation_matrix": rotation.tolist(),
                "euler_angles_deg": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
                "baseline_m": float(np.linalg.norm(translation)),
            }

        return f"Cam{reference_camera_id - 20000}", relative_poses

    def _summarize_reprojection_errors(self, reprojection_results):
        """Aggregate every valid corner error across all vehicle cameras.

        Returns mean/min/max/std over the individual corner errors (not the
        per-board means), plus the per-camera mean spread.
        """
        if not reprojection_results:
            return None

        corner_errors = []
        camera_means = []

        for error_info in reprojection_results.get('camera_errors', {}).values():
            detections = error_info.get('detections') or []
            if not detections:
                continue
            for det in detections:
                for corner in det.get('corner_details', []):
                    if corner.get('valid') and np.isfinite(corner['error']):
                        corner_errors.append(float(corner['error']))
            camera_means.append(float(error_info['camera_mean_error']))

        if not corner_errors:
            return None

        corner_errors = np.asarray(corner_errors)
        return {
            'mean': float(np.mean(corner_errors)),
            'min': float(np.min(corner_errors)),
            'max': float(np.max(corner_errors)),
            'std': float(np.std(corner_errors)),
            'rmse': float(np.sqrt(np.mean(corner_errors ** 2))),
            'num_corners': int(corner_errors.size),
            'num_cameras': len(camera_means),
            'camera_mean_min': float(np.min(camera_means)) if camera_means else 0.0,
            'camera_mean_max': float(np.max(camera_means)) if camera_means else 0.0,
        }

    @staticmethod
    def _format_reprojection_summary_lines(summary, indent="  "):
        """Human-readable lines for an _summarize_reprojection_errors() result."""
        return [
            f"{indent}Overall mean : {summary['mean']:.2f} px",
            f"{indent}Overall min  : {summary['min']:.2f} px",
            f"{indent}Overall max  : {summary['max']:.2f} px",
            f"{indent}Overall std  : {summary['std']:.2f} px",
            f"{indent}Overall RMSE : {summary['rmse']:.2f} px",
            f"{indent}Corners used : {summary['num_corners']} "
            f"(from {summary['num_cameras']} cameras)",
            f"{indent}Camera mean range: {summary['camera_mean_min']:.2f} ~ "
            f"{summary['camera_mean_max']:.2f} px",
        ]

    def _generate_projection_images(self, result, vehicle_camera_ids):
        """Render the reprojection overlays for every vehicle camera in one pass."""
        projection_images = None
        try:
            reprojection_results = self._compute_vehicle_reprojection_errors(result, vehicle_camera_ids)

            if reprojection_results and hasattr(self.calibration, 'vehicle_detections') and self.calibration.vehicle_detections:
                vehicle_detections = self.calibration.vehicle_detections

                # Create image_path_mapping for ALL cameras
                image_path_mapping = {}
                self.append_log("  Creating image_path_mapping for all cameras...\n")
                self.append_log(f"    Vehicle detections filenames: {list(vehicle_detections.keys())}\n")

                # Collect all selected images from all vehicle cameras
                all_selected_images = []
                for vid in vehicle_camera_ids:
                    vidx = vid - 20000
                    if vidx < len(self.config.vehicle_camera_configs):
                        vcfg = self.config.vehicle_camera_configs[vidx]
                        selected_imgs = vcfg.get('images', [])
                        all_selected_images.extend(selected_imgs)

                self.append_log(f"    All selected images: {[os.path.basename(p) for p in all_selected_images]}\n")

                # Detection keys are "cam{idx}_{basename}", so resolve each
                # camera against its own selected images first — two cameras
                # may select files with identical basenames.
                for vid in vehicle_camera_ids:
                    vidx = vid - 20000
                    if vidx >= len(self.config.vehicle_camera_configs):
                        continue
                    for img_path in self.config.vehicle_camera_configs[vidx].get('images', []):
                        key = f"cam{vidx}_{os.path.basename(img_path)}"
                        if key in vehicle_detections:
                            image_path_mapping[key] = img_path
                            self.append_log(f"    ✓ Mapped {key} → {os.path.basename(img_path)}\n")
                # Legacy keys (bare basenames, e.g. from an old pkl) fall back
                # to substring matching.
                for filename in vehicle_detections.keys():
                    if filename in image_path_mapping:
                        continue
                    for img_path in all_selected_images:
                        basename = os.path.basename(img_path)
                        if (basename in filename or filename in basename or
                            filename in img_path or basename == filename):
                            image_path_mapping[filename] = img_path
                            self.append_log(f"    ✓ Mapped {filename} → {basename}\n")
                            break

                # Fallback mapping if needed
                if len(image_path_mapping) < len(vehicle_detections) and all_selected_images:
                    unmapped = [f for f in vehicle_detections.keys() if f not in image_path_mapping]
                    for idx, filename in enumerate(unmapped):
                        if idx < len(all_selected_images):
                            image_path_mapping[filename] = all_selected_images[idx]
                            self.append_log(f"    ⚠ Fallback mapping: {filename} → {os.path.basename(all_selected_images[idx])}\n")

                self.append_log(f"  Total mappings created: {len(image_path_mapping)}\n")

                # 5. Visualize projection results (get images for ALL cameras at once!)
                self.append_log("  Generating projection images for all cameras...\n")
                projection_images = self.calibration.system.visualize_corner_reprojection_results(
                    reprojection_results,
                    image_path_mapping=image_path_mapping,
                    save_dir=None,
                    return_images=True
                )

                if projection_images:
                    self.append_log(f"  ✓ Generated {len(projection_images)} projection images\n")
                    # Keep them so the Export Images button can write these
                    # exact renderings to disk.
                    self._projection_images = dict(projection_images)
                else:
                    self.append_log("  ⚠ No projection images generated\n")

        except Exception as proj_gen_error:
            self.append_log(f"  ⚠ Projection generation failed: {str(proj_gen_error)}\n")
            import traceback
            traceback.print_exc()

        return projection_images

    @staticmethod
    def _camera_pose_in_vehicle_frame(T_cam_in_world, vehicle_center, R_world_to_vehicle):
        """Return the 4x4 pose of one camera expressed in the vehicle frame.

        atPose3 returns T_cam_in_world (camera-to-world): translation = camera
        origin in world coords, rotation = R_cam_to_world. The old name
        "T_world_to_camera" was inverted.
        """
        camera_pos_world = T_cam_in_world.translation()
        camera_rot_world = T_cam_in_world.rotation().matrix()

        # Position: p_vehicle = R_world_to_vehicle * (p_world - vehicle_center)
        camera_pos_vehicle = R_world_to_vehicle @ (camera_pos_world - vehicle_center)
        # Rotation: R_cam_to_vehicle = R_world_to_vehicle * R_cam_to_world
        camera_rot_vehicle = R_world_to_vehicle @ camera_rot_world

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = camera_rot_vehicle
        pose_matrix[:3, 3] = camera_pos_vehicle
        return pose_matrix

    @staticmethod
    def _display_pixmap_from_bgr(image_bgr, display_width=800):
        """Scale a BGR image to display_width and convert it to a QPixmap."""
        from PySide6.QtGui import QImage, QPixmap

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        display_height = int(h * display_width / w)
        img_resized = cv2.resize(img_rgb, (display_width, display_height))
        q_img = QImage(img_resized.data, display_width, display_height,
                       display_width * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def _fill_projection_image_label(self, image_label, camera_id, projection_images,
                                     selected_images):
        """Show the reprojection overlay for one camera, or the original image."""
        try:
            if projection_images and camera_id in projection_images:
                image_label.setPixmap(
                    self._display_pixmap_from_bgr(projection_images[camera_id]))
                image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
            else:
                raise Exception("No projection image available")

        except Exception as proj_error:
            # Fallback: show the ORIGINAL image, but say so —
            # silently presenting it styled like a projection
            # overlay lets the user "verify" reprojection
            # quality against an image that has no overlay.
            self.append_log(
                f"⚠  Cam{camera_id - 20000}: no projection overlay "
                f"({proj_error}) — showing original image\n")
            first_image_path = selected_images[0]
            img = cv2.imread(first_image_path)
            if img is not None:
                banner_h = max(40, img.shape[0] // 20)
                cv2.rectangle(img, (0, 0), (img.shape[1], banner_h),
                              (0, 0, 180), -1)
                cv2.putText(img, "NO PROJECTION OVERLAY - original image",
                            (12, int(banner_h * 0.7)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            banner_h / 55.0, (255, 255, 255), 2)
                image_label.setPixmap(self._display_pixmap_from_bgr(img))
                image_label.setStyleSheet("background-color: #f0f0f0; border: 2px solid #c62828;")

    def _build_projection_camera_widget(self, camera_id, pose_matrix, projection_images):
        """Build one camera's projection card: title, overlay image and pose text."""
        rotation = pose_matrix[:3, :3]
        translation = pose_matrix[:3, 3]

        # Euler angles (Roll, Pitch, Yaw) - ZYX extrinsic (vehicle standard)
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rotation)
        yaw, pitch, roll = r.as_euler('ZYX', degrees=True)

        camera_widget = QWidget()
        camera_layout = QVBoxLayout(camera_widget)
        camera_layout.setContentsMargins(10, 10, 10, 10)
        camera_layout.setSpacing(10)

        # Camera title (consistent naming: Cam0, Cam1, ...)
        cam_idx = camera_id - 20000
        title_label = QLabel(f"📷 Cam{cam_idx}")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1b338c;")
        camera_layout.addWidget(title_label)

        image_label = QLabel("🔄 Loading projection image...")
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; min-height: 400px;")
        camera_layout.addWidget(image_label)

        if cam_idx < len(self.config.vehicle_camera_configs):
            selected_images = self.config.vehicle_camera_configs[cam_idx].get('images', [])
            if selected_images:
                self._fill_projection_image_label(image_label, camera_id,
                                                  projection_images, selected_images)

        pose_text = f"""
<b style="font-size: 14px; color: #1b338c;">4x4 Transformation Matrix (Vehicle Frame):</b>
<pre style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: 'Courier New', monospace; font-size: 11px;">
[[{pose_matrix[0,0]:9.6f}, {pose_matrix[0,1]:9.6f}, {pose_matrix[0,2]:9.6f}, {pose_matrix[0,3]:9.6f}],
 [{pose_matrix[1,0]:9.6f}, {pose_matrix[1,1]:9.6f}, {pose_matrix[1,2]:9.6f}, {pose_matrix[1,3]:9.6f}],
 [{pose_matrix[2,0]:9.6f}, {pose_matrix[2,1]:9.6f}, {pose_matrix[2,2]:9.6f}, {pose_matrix[2,3]:9.6f}],
 [{pose_matrix[3,0]:9.6f}, {pose_matrix[3,1]:9.6f}, {pose_matrix[3,2]:9.6f}, {pose_matrix[3,3]:9.6f}]]
</pre>

<b style="font-size: 13px;">Translation (x, y, z):</b> <span style="font-family: 'Courier New', monospace;">[{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}] m</span><br>
<b style="font-size: 13px;">Rotation (Roll, Pitch, Yaw):</b> <span style="font-family: 'Courier New', monospace;">[{roll:.3f}°, {pitch:.3f}°, {yaw:.3f}°]</span>
"""

        pose_label = QLabel(pose_text)
        pose_label.setWordWrap(True)
        pose_label.setTextFormat(Qt.RichText)
        pose_label.setStyleSheet("padding: 10px; background-color: white; border: 1px solid #ddd; border-radius: 5px;")
        camera_layout.addWidget(pose_label)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #ccc; margin: 10px 0;")
        camera_layout.addWidget(separator)

        return camera_widget

    def export_projection_images(self):
        """Save the projection overlay images generated by Load Projection Results."""
        try:
            calib = getattr(self, 'calibration', None)
            if not getattr(calib, 'vehicle_camera_ids', None):
                QMessageBox.warning(
                    self, "Export Images",
                    "No vehicle camera data available.\n\n"
                    "Please run Extrinsic Calibration to add vehicle cameras."
                )
                return

            images = getattr(self, '_projection_images', None)
            if not images:
                QMessageBox.information(
                    self, "Export Images",
                    "No projection images available yet.\n\n"
                    "Open the 📷 Projection tab (or press 🔄 Refresh there) first — "
                    "the images generated there are what this button saves."
                )
                return

            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Projection Images",
                "",
                QFileDialog.ShowDirsOnly
            )
            if not save_dir:
                return

            self.append_log(f"\n[export] Exporting projection images to: {save_dir}\n")
            saved = []
            for camera_id, image in sorted(images.items()):
                name = (f"projection_cam{camera_id - 20000}.png"
                        if camera_id >= 20000 else f"projection_C{camera_id}.png")
                out_path = os.path.join(save_dir, name)
                if cv2.imwrite(out_path, image):
                    saved.append(name)
                    self.append_log(f"  ✓ {name}\n")
                else:
                    self.append_log(f"  ⚠ failed to write {name}\n")

            QMessageBox.information(
                self, "Export Images",
                f"Saved {len(saved)} image(s) to:\n{save_dir}"
            )

        except Exception as e:
            self.append_log(f"⚠  Error exporting images: {str(e)}\n")

    def export_vehicle_poses(self):
        """Save camera poses in vehicle frame to file"""
        try:
            import json

            if (not hasattr(self, 'calibration') or self.calibration is None
                    or not getattr(self.calibration, 'vehicle_camera_ids', None)):
                QMessageBox.warning(
                    self, "Export Poses",
                    "No vehicle camera data available.\n\n"
                    "Please run Extrinsic Calibration to add vehicle cameras."
                )
                return

            # Select save file
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Vehicle Camera Poses",
                "vehicle_camera_poses.txt",
                "Text Files (*.txt);;JSON Files (*.json);;YAML Files (*.yaml);;All Files (*)"
            )

            if not save_path:
                return

            self.append_log(f"\n[export] Exporting vehicle camera poses to: {save_path}\n")

            import gtsam
            from scipy.spatial.transform import Rotation as R

            result = self.calibration.result
            vehicle_camera_ids = self.calibration.vehicle_camera_ids

            # Check if vehicle coordinate frame is cached
            if not hasattr(self.calibration.system, 'cached_vehicle_frame') or self.calibration.system.cached_vehicle_frame is None:
                self.calibration.system.add_vehicle_coordinate_system_post_optimization(result)

            vehicle_frame = self.calibration.system.cached_vehicle_frame
            vehicle_center = vehicle_frame['center']
            R_world_to_vehicle = vehicle_frame['R_world_to_vehicle']

            # Collect pose data
            poses_data = {}

            for camera_id in sorted(vehicle_camera_ids):
                camera_key = gtsam.symbol('C', camera_id)

                if not result.exists(camera_key):
                    continue

                # Get camera pose in world frame
                T_world_to_camera = result.atPose3(camera_key)
                camera_pos_world = T_world_to_camera.translation()
                camera_rot_world = T_world_to_camera.rotation().matrix()

                # Transform to vehicle frame
                camera_pos_vehicle = R_world_to_vehicle @ (camera_pos_world - vehicle_center)
                camera_rot_vehicle = R_world_to_vehicle @ camera_rot_world

                # Build 4x4 transformation matrix
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = camera_rot_vehicle
                pose_matrix[:3, 3] = camera_pos_vehicle

                # Extract rotation and translation
                rotation = pose_matrix[:3, :3]
                translation = pose_matrix[:3, 3]

                # Compute Euler angles - ZYX extrinsic (vehicle standard)
                r = R.from_matrix(rotation)
                yaw, pitch, roll = r.as_euler('ZYX', degrees=True)

                poses_data[f"Cam{camera_id - 20000}"] = {
                    "xyzrpy": [float(translation[0]), float(translation[1]), float(translation[2]),
                               float(roll), float(pitch), float(yaw)],
                    "matrix": pose_matrix.tolist(),
                    "translation": translation.tolist(),
                    "rotation_matrix": rotation.tolist(),
                    "euler_angles_deg": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)}
                }

                x, y, z = float(translation[0]), float(translation[1]), float(translation[2])
                self.append_log(
                    f"  Cam{camera_id - 20000}  "
                    f"[x={x:.4f} m, y={y:.4f} m, z={z:.4f} m, "
                    f"roll={roll:.3f}°, pitch={pitch:.3f}°, yaw={yaw:.3f}°]\n"
                )

            ref_label, relative_poses = self._compute_relative_camera_poses(result, vehicle_camera_ids)
            if relative_poses:
                self.append_log(f"\n[export] Relative camera poses (reference: {ref_label}):\n")
                for cam_label, pose_info in relative_poses.items():
                    if cam_label == ref_label:
                        continue
                    x, y, z, roll, pitch, yaw = pose_info["xyzrpy"]
                    self.append_log(
                        f"  {ref_label}→{cam_label}  "
                        f"[x={x:.4f} m, y={y:.4f} m, z={z:.4f} m, "
                        f"roll={roll:.3f}°, pitch={pitch:.3f}°, yaw={yaw:.3f}°, "
                        f"baseline={pose_info['baseline_m']:.4f} m]\n"
                    )

            reprojection_results = self._compute_vehicle_reprojection_errors(result, vehicle_camera_ids)
            error_summary = self._summarize_reprojection_errors(reprojection_results)
            if reprojection_results:
                self.append_log("\n[export] Reprojection errors (per board):\n")
                for camera_id in sorted(vehicle_camera_ids):
                    error_info = reprojection_results['camera_errors'].get(camera_id)
                    if not error_info:
                        continue
                    cam_label = f"Cam{camera_id - 20000}"
                    self.append_log(f"  {cam_label}:\n")
                    for det in sorted(error_info['detections'], key=lambda d: d['marker_id']):
                        self.append_log(
                            f"    M{det['marker_id']}: {det['mean_error']:.2f} px\n"
                        )
                    self.append_log(
                        f"    Camera mean: {error_info['camera_mean_error']:.2f} px\n"
                    )

            if error_summary:
                self.append_log("\n[export] Total reprojection error (all cameras, all corners):\n")
                for line in self._format_reprojection_summary_lines(error_summary):
                    self.append_log(line + "\n")

            # Save based on file extension
            file_ext = save_path.split('.')[-1].lower()

            # Keep the per-camera keys at the top level for existing consumers and
            # attach the relative poses under a separate key.
            structured_data = dict(poses_data)
            if relative_poses:
                structured_data["relative_poses"] = {
                    "reference_camera": ref_label,
                    "poses": relative_poses,
                }

            if file_ext == 'json':
                with open(save_path, 'w') as f:
                    json.dump(structured_data, f, indent=2)
            elif file_ext == 'yaml':
                try:
                    import yaml  # type: ignore
                    with open(save_path, 'w') as f:
                        yaml.dump(structured_data, f, default_flow_style=False)
                except ImportError:
                    self.append_log("⚠ PyYAML not installed. Saving as JSON instead.\n")
                    save_path = save_path.replace('.yaml', '.json')
                    with open(save_path, 'w') as f:
                        json.dump(structured_data, f, indent=2)
            else:  # txt or other
                with open(save_path, 'w') as f:
                    f.write("=" * 80 + "\n")
                    f.write("Vehicle Camera Poses (Vehicle Coordinate Frame)\n")
                    f.write("=" * 80 + "\n\n")

                    for cam_id, pose_info in poses_data.items():
                        f.write(f"Camera {cam_id}:\n")
                        f.write("-" * 80 + "\n")

                        matrix = np.array(pose_info["matrix"])
                        f.write("4x4 Transformation Matrix:\n")
                        for row in matrix:
                            f.write(f"  [{row[0]:9.6f}, {row[1]:9.6f}, {row[2]:9.6f}, {row[3]:9.6f}]\n")

                        xyzrpy = pose_info["xyzrpy"]
                        f.write(f"\n[x, y, z, roll, pitch, yaw]: "
                                f"[{xyzrpy[0]:.6f}, {xyzrpy[1]:.6f}, {xyzrpy[2]:.6f}, "
                                f"{xyzrpy[3]:.4f}, {xyzrpy[4]:.4f}, {xyzrpy[5]:.4f}]  (m, deg)\n")

                        trans = pose_info["translation"]
                        f.write(f"Translation (x, y, z): [{trans[0]:.6f}, {trans[1]:.6f}, {trans[2]:.6f}] m\n")

                        euler = pose_info["euler_angles_deg"]
                        f.write(f"Rotation (Roll, Pitch, Yaw): [{euler['roll']:.4f}, {euler['pitch']:.4f}, {euler['yaw']:.4f}]°\n")

                        if reprojection_results:
                            camera_key_id = int(cam_id.replace('Cam', '')) + 20000
                            error_info = reprojection_results['camera_errors'].get(camera_key_id)
                            if error_info and error_info['detections']:
                                f.write("\nReprojection Errors (per board):\n")
                                for det in sorted(error_info['detections'], key=lambda d: d['marker_id']):
                                    f.write(f"  M{det['marker_id']}: {det['mean_error']:.2f} px\n")
                                f.write(f"  Camera mean: {error_info['camera_mean_error']:.2f} px\n")

                        f.write("\n")

                    if relative_poses:
                        f.write("=" * 80 + "\n")
                        f.write(f"Relative Camera Poses (reference: {ref_label}, "
                                f"vehicle-frame independent)\n")
                        f.write("=" * 80 + "\n\n")

                        for cam_label, pose_info in relative_poses.items():
                            if cam_label == ref_label:
                                continue
                            f.write(f"{ref_label} -> {cam_label}:\n")
                            f.write("-" * 80 + "\n")

                            matrix = np.array(pose_info["matrix"])
                            f.write("4x4 Transformation Matrix:\n")
                            for row in matrix:
                                f.write(f"  [{row[0]:9.6f}, {row[1]:9.6f}, {row[2]:9.6f}, {row[3]:9.6f}]\n")

                            x, y, z, roll, pitch, yaw = pose_info["xyzrpy"]
                            f.write(f"\n[x, y, z, roll, pitch, yaw]: "
                                    f"[{x:.6f}, {y:.6f}, {z:.6f}, "
                                    f"{roll:.4f}, {pitch:.4f}, {yaw:.4f}]  (m, deg)\n")
                            f.write(f"Baseline: {pose_info['baseline_m']:.6f} m\n\n")

                        f.write("=" * 80 + "\n")
                        f.write(f"REL_CAMERA_POSES  (reference: {ref_label})\n")
                        f.write("=" * 80 + "\n")
                        f.write("REL_CAMERA_POSES = {\n")
                        f.write("    #     x          y          z          roll       pitch      yaw\n")
                        for cam_label, pose_info in relative_poses.items():
                            x, y, z, roll, pitch, yaw = pose_info["xyzrpy"]
                            f.write(f"    {cam_label.replace('Cam', '')}: "
                                    f"[{x:.6f}, {y:.6f}, {z:.6f}, "
                                    f"{roll:.4f}, {pitch:.4f}, {yaw:.4f}],\n")
                        f.write("}\n\n")

                    if error_summary:
                        f.write("=" * 80 + "\n")
                        f.write("TOTAL REPROJECTION ERROR (all cameras, all corners)\n")
                        f.write("=" * 80 + "\n")
                        for line in self._format_reprojection_summary_lines(error_summary):
                            f.write(line + "\n")
                        f.write("\n")

                    # Same poses as a dict literal, ready to paste into an external
                    # accuracy-check script.
                    f.write("=" * 80 + "\n")
                    f.write("EST_CAMERA_POSES  (dict literal, ready to paste)\n")
                    f.write("=" * 80 + "\n")
                    f.write("EST_CAMERA_POSES = {\n")
                    f.write("    #     x          y          z          roll       pitch      yaw\n")
                    for cam_id, pose_info in poses_data.items():
                        x, y, z, roll, pitch, yaw = pose_info["xyzrpy"]
                        f.write(f"    {cam_id.replace('Cam', '')}: "
                                f"[{x:.6f}, {y:.6f}, {z:.6f}, "
                                f"{roll:.4f}, {pitch:.4f}, {yaw:.4f}],\n")
                    f.write("}\n")

            self.append_log(f"✓ Exported poses for {len(poses_data)} cameras\n")

            QMessageBox.information(
                self,
                "Export Successful",
                f"Vehicle camera poses saved to:\n{save_path}"
            )

        except Exception as e:
            self.append_log(f"⚠  Error exporting poses: {str(e)}\n")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Failed to export poses:\n{str(e)}")

    def update_config_display(self):
        """Display current config info in right panel"""

        def _fmt_intrinsic(p):
            if p is None:
                return "⚠  NOT SET — open Camera Parameters dialog"
            return f"fx={p[0]:.4f}, fy={p[1]:.4f}, cx={p[2]:.4f}, cy={p[3]:.4f}"

        def _fmt_dist(d):
            if d is None:
                return "⚠  NOT SET — open Camera Parameters dialog"
            if len(d) == 4:
                return f"[Fisheye KB] k1={d[0]:.6f}, k2={d[1]:.6f}, k3={d[2]:.6f}, k4={d[3]:.6f}"
            return f"k1={d[0]:.6f}, k2={d[1]:.6f}, p1={d[2]:.6f}, p2={d[3]:.6f}, k3={d[4]:.6f}"

        # Wheel camera info
        wheel_dist = self.config.wheel_dist_coeffs
        wheel_cam_text = f"""
  Wheel Camera:
    Intrinsic: {_fmt_intrinsic(self.config.wheel_camera_params)}
    Distortion: {_fmt_dist(wheel_dist)}
"""

        # Generate display string for vehicle camera configs (dynamic)
        vehicle_cam_text = ""
        for i, cam_config in enumerate(self.config.vehicle_camera_configs):
            intrinsic = cam_config['intrinsic']
            dist = cam_config['dist_coeffs']
            num_images = len(cam_config.get('images', []))
            vehicle_cam_text += f"""
  Vehicle Cam{i}:
    Intrinsic: {_fmt_intrinsic(intrinsic)}
    Distortion: {_fmt_dist(dist)}
    Images: {num_images} files
"""

        config_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                   HIERARCHICAL FGO CONFIGURATION                 ║
╚══════════════════════════════════════════════════════════════════╝

📁 DIRECTORY SETTINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Environment Images : {self.config.image_directory}
  Wheel Images       : {self.config.wheel_directory}
  Vehicle Images     : {self._vehicle_images_dir()}

📷 CAMERA PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Number of Vehicle Cameras: {self.config.num_vehicle_cameras}
  
  Environment Camera:
    Intrinsic: {_fmt_intrinsic(self.config.camera_params)}
    Distortion: {_fmt_dist(self.config.dist_coeffs)}
{wheel_cam_text}{vehicle_cam_text}
🎯 MARKER & APRILTAG SETTINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Env Marker Size    : {self.config.marker_size * 2:.3f} m  (half: {self.config.marker_size:.4f} m)
  Wheel Marker Size  : {self.config.marker_size2 * 2:.3f} m  (half: {self.config.marker_size2:.4f} m)
  World-Origin Board ID : {self.config.first_marker_id}  (picks an existing board_id; does NOT affect grouping)
  Vehicle Board IDs  : {self.config.vehicle_markers}  (FL, RL, RR, FR; must match computed board_ids below)
  Tag Family         : {getattr(self.config, 'apriltag_family', 'tagStandard41h12')}
  Env Board Grid     : {getattr(self.config, 'env_grid_rows', 2)}×{getattr(self.config, 'env_grid_cols', 2)}, gap {getattr(self.config, 'env_tag_gap', 0.246914):g} m, ★env_first_tag_id={getattr(self.config, 'env_first_tag_id', 10)} (controls board_id grouping)
  Wheel Board Grid   : {getattr(self.config, 'wheel_grid_rows', 2)}×{getattr(self.config, 'wheel_grid_cols', 2)}, gap {getattr(self.config, 'wheel_tag_gap', 0.154321):g} m (board_id looked up directly from Vehicle Board IDs)
  Excluded Board IDs : {getattr(self.config, 'excluded_apriltag_ids', []) or 'None'}

📊 UNCERTAINTY SETTINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Covariance Model   : MARSCOT (Unscented Transform through PnP)
  Geo Skip Rot       : {getattr(self.config, 'geo_skip_rot_deg', 4.0)} deg
  Geo Skip Pos       : {getattr(self.config, 'geo_skip_pos_m', 0.1)} m
  Min Pair Obs       : {getattr(self.config, 'min_pair_obs', 2)}

⚙ OPTIMIZATION SETTINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Use Saved Results  : {self.config.use_saved_results}
  Save File          : {self.config.save_file}
"""
        self.config_display.setPlainText(config_text)

    @staticmethod
    def _vehicle_configs_source(cfg):
        """Render cfg.vehicle_camera_configs as the source text for src/config.py."""
        vehicle_blocks = []
        for vc in cfg.vehicle_camera_configs:
            intr  = list(vc['intrinsic']) if vc['intrinsic'] is not None else [1.0, 1.0, 0.0, 0.0]
            dist  = list(vc['dist_coeffs']) if vc['dist_coeffs'] is not None else [0.0]*5
            model = vc.get('camera_model', 'pinhole')

            # repr() keeps Windows backslashes escaped, and one path per line
            # keeps the generated config readable.
            images = list(vc.get('images', []))
            if images:
                images_str = ("[\n"
                              + ",\n".join(f"                    {p!r}" for p in images)
                              + "\n                ]")
            else:
                images_str = "[]"

            vehicle_blocks.append(
                f"            {{\n"
                f"                'intrinsic': {intr},\n"
                f"                'camera_model': '{model}',\n"
                f"                'dist_coeffs': np.array({dist}),\n"
                f"                'images': {images_str}\n"
                f"            }}"
            )
        return ",\n".join(vehicle_blocks)

    def save_config(self):
        """Save current config values to src/config.py (full rewrite of __init__ values)."""
        try:
            cfg = self.config
            config_path = os.path.join(self.project_root, 'src', 'config.py')
            vehicle_configs_str = self._vehicle_configs_source(cfg)

            content = f'''\
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

        self.image_directory = {repr(cfg.image_directory)}   # must be set from GUI before running
        self.wheel_directory = {repr(cfg.wheel_directory)}   # must be set from GUI before running
        self.vehicle_directory = ''   # unused: images managed via vehicle_camera_configs
        self.image_extensions = {cfg.image_extensions}

        # ============================================================================
        # Marker settings
        # ============================================================================
        self.marker_size = {cfg.marker_size}  # half of marker size (for environment markers)
        self.marker_size2 = {cfg.marker_size2}  # half of marker size (for vehicle wheel markers)

        # ============================================================================
        # AprilTag / AprilGrid settings (AprilTag 3 detector)
        # ============================================================================
        # Family: tagStandard41h12, tagStandard52h13, tag36h11, tag25h9, tag16h5,
        #         tagCircle21h7, tagCircle49h12, tagCustom48h12
        self.apriltag_family = {repr(getattr(cfg, 'apriltag_family', 'tagStandard41h12'))}

        # AprilTag 3 detector tuning. quad_decimate downsamples only the quad
        # search; corners are refined at full resolution afterwards.
        self.quad_decimate = {getattr(cfg, 'quad_decimate', 2.0)}
        self.quad_sigma = {getattr(cfg, 'quad_sigma', 0.0)}
        self.refine_edges = {getattr(cfg, 'refine_edges', 1)}
        self.decode_sharpening = {getattr(cfg, 'decode_sharpening', 0.25)}
        self.nthreads = {getattr(cfg, 'nthreads', 0)}  # 0 = one thread per CPU core

        # Board layout: n×m AprilTags are printed on one physical marker board.
        # board_id = min(tag_ids_on_board) = first_tag_id + group_index * (rows*cols)
        # tag_gap is the printed gap between neighbouring tag edges, in meters.
        self.env_grid_rows    = {getattr(cfg, 'env_grid_rows', 2)}
        self.env_grid_cols    = {getattr(cfg, 'env_grid_cols', 2)}
        self.env_tag_gap      = {getattr(cfg, 'env_tag_gap', 0.246914)}
        self.env_first_tag_id = {getattr(cfg, 'env_first_tag_id', 10)}   # min tag_id of first env board

        self.wheel_grid_rows    = {getattr(cfg, 'wheel_grid_rows', 2)}
        self.wheel_grid_cols    = {getattr(cfg, 'wheel_grid_cols', 2)}
        self.wheel_tag_gap      = {getattr(cfg, 'wheel_tag_gap', 0.154321)}
        # NOTE: wheel board_id is looked up directly from vehicle_markers below
        # (no separate wheel_first_tag_id anchor).

        # marker size mapping by board ID
        self.marker_size_mapping = {{}}

        # World-origin board ID (GTSAM M1). board_id = min(tag_ids_on_board)
        self.first_marker_id = {cfg.first_marker_id}

        # Vehicle wheel board IDs [fl, rl, rr, fr]. board_id = min(tag_ids_on_board)
        self.vehicle_markers = {cfg.vehicle_markers}

        # ============================================================================
        # Camera parameter settings
        # ============================================================================
        # Environment camera parameters [fx, fy, cx, cy]
        # Initial values shown in GUI — overwritten when user saves Camera Parameters dialog
        self.camera_params = {list(cfg.camera_params)}

        # Distortion coefficients [k1, k2, p1, p2, k3] (OpenCV order)
        self.dist_coeffs = np.array({list(cfg.dist_coeffs)})

        # Wheel camera parameters
        # Initial values shown in GUI — overwritten when user saves Camera Parameters dialog
        self.wheel_camera_params = {list(cfg.wheel_camera_params)}
        self.wheel_dist_coeffs = np.array({list(cfg.wheel_dist_coeffs)})

        # ============================================================================
        # Vehicle camera settings
        # ============================================================================
        # number of vehicle cameras (adjustable dynamically)
        self.num_vehicle_cameras = {len(cfg.vehicle_camera_configs)}

        # vehicle camera settings (index-ordered)
        self.vehicle_camera_configs = [
{vehicle_configs_str}
        ]

        # ============================================================================
        # Observation filters
        # ============================================================================
        # Every observation's 6x6 pose covariance comes from MARSCOT (Unscented
        # Transform through PnP); there is no alternative model to select.

        # Geometric consistency thresholds (Stage 3: vehicle camera observations
        # inconsistent with the frozen marker map beyond these limits are
        # excluded from the factor graph)
        self.geo_skip_rot_deg  = {cfg.geo_skip_rot_deg}
        self.geo_skip_pos_m    = {cfg.geo_skip_pos_m}

        # M2M pair minimum observation filter
        self.min_pair_obs = {getattr(cfg, 'min_pair_obs', 2)}

        # AprilTag board IDs excluded from detection: any env/wheel image in
        # which one of these IDs is detected is skipped entirely (empty list =
        # no exclusion). Stage 3 vehicle images are NOT filtered by this.
        self.excluded_apriltag_ids = {list(getattr(cfg, 'excluded_apriltag_ids', None) or [])!r}

        # ============================================================================
        # Environment optimization settings
        # ============================================================================
        # Default for run_optimization(use_saved=None) — library/CLI use only;
        # the GUI always passes an explicit use_saved value.
        self.use_saved_results = {cfg.use_saved_results}  # True: Load saved results, False: Run new
        # Stage 1 result file (relative to the working directory)
        self.save_file = {getattr(cfg, 'save_file', 'optimization_results.pkl')!r}

    def update_num_vehicle_cameras(self, num_cameras):
        """update number of vehicle cameras"""
        # allow 0 cameras (no images selected)
        if num_cameras < 0:
            num_cameras = 0

        current_num = len(self.vehicle_camera_configs)
        self.num_vehicle_cameras = num_cameras

        if num_cameras > current_num:
            for i in range(current_num, num_cameras):
                self.vehicle_camera_configs.append({{
                    'intrinsic': [1.0, 1.0, 0.0, 0.0],
                    'camera_model': 'pinhole',
                    'dist_coeffs': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                    'images': []
                }})
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
'''

            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)

            QMessageBox.information(self, "Success", "Configuration saved to src/config.py successfully!")
            print("✓ Configuration saved to src/config.py")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")
            print(f"✗ Error saving configuration: {e}")

    def load_config(self):
        """Load config from config.py"""
        try:
            # Re-read src/config.py from disk. The module is cached under
            # 'src.config'; without a reload this would silently return the
            # values from app startup, not what Save Config just wrote.
            import importlib
            import src.config
            importlib.reload(src.config)
            self.config = src.config.CalibrationConfig()

            # Update UI
            self.env_path_label.setText(f"Environment: <b>{self.config.image_directory}</b>")
            self.wheel_path_label.setText(f"Wheel: <b>{self.config.wheel_directory}</b>")
            self.update_config_display()

            QMessageBox.information(self, "Success", "Configuration loaded from src/config.py successfully!")
            print("✓ Configuration loaded from src/config.py")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")
            print(f"✗ Error loading configuration: {e}")

    def run_environment_calibration(self):
        """Run environment optimization (Environment Optimization)"""
        try:
            # Initialize previous calibration data (important!)
            self.calibration = None
            self._projection_images = None   # overlays belong to the old result
            self.append_log("[gui] Cleared previous calibration data\n")

            # Confirmation dialog
            reply = QMessageBox.question(
                self,
                "Run Environment Optimization",
                "This will run the environment factor graph optimization.\n\n"
                f"Environment Images: {self.config.image_directory}\n"
                f"Wheel Images: {self.config.wheel_directory}\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return

            # Switch to Log tab
            self.right_tab_widget.setCurrentIndex(1)
            self.clear_log()
            self.append_log("=" * 60 + "\n")
            self.append_log("Starting Environment Optimization\n")
            self.append_log("=" * 60 + "\n\n")

            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress

            self.calibration_thread = CalibrationThread(self.config, 'environment')
            self.calibration_thread.finished.connect(self.on_calibration_finished)
            self.calibration_thread.progress.connect(self.append_log)
            self.calibration_thread.calibration_system.connect(self.on_calibration_system_ready)

            # Redirect stdout/stderr to the GUI log
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            sys.stdout = self.stdout_redirector
            sys.stderr = self.stderr_redirector
            self.calibration_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run environment calibration:\n{str(e)}")
            self.append_log(f"✗ Error in environment calibration: {e}\n")
            import traceback
            traceback.print_exc()

    def run_extrinsic_calibration(self):
        """Run extrinsic calibration"""
        self._run_pipeline_calibration(
            'extrinsic', "Run Extrinsic Calibration",
            "This will run the full hierarchical extrinsic calibration.",
            "Starting Extrinsic Calibration (Full Pipeline)")

    def _run_pipeline_calibration(self, calibration_type, dialog_title, description, log_title):
        """Launch the Environment → Wheel → Vehicle pipeline ('extrinsic')"""
        try:
            # Initialize previous calibration data (important!)
            self.calibration = None
            self._projection_images = None   # overlays belong to the old result
            self.append_log("[gui] Cleared previous calibration data\n")

            # Confirmation dialog
            reply = QMessageBox.question(
                self,
                dialog_title,
                f"{description}\n\n"
                f"Environment Images: {self.config.image_directory}\n"
                f"Wheel Images: {self.config.wheel_directory}\n"
                f"Vehicle Images: {self._vehicle_images_dir()}\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                return

            # Switch to Log tab
            self.right_tab_widget.setCurrentIndex(1)
            self.clear_log()
            self.append_log("=" * 60 + "\n")
            self.append_log(log_title + "\n")
            self.append_log("=" * 60 + "\n\n")

            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress

            self.calibration_thread = CalibrationThread(self.config, calibration_type)
            self.calibration_thread.finished.connect(self.on_calibration_finished)
            self.calibration_thread.progress.connect(self.append_log)
            self.calibration_thread.calibration_system.connect(self.on_calibration_system_ready)

            # Redirect stdout/stderr to the GUI log
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            sys.stdout = self.stdout_redirector
            sys.stderr = self.stderr_redirector
            self.calibration_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run {calibration_type} calibration:\n{str(e)}")
            self.append_log(f"✗ Error in {calibration_type} calibration: {e}\n")
            import traceback
            traceback.print_exc()


def fit_dialog_to_screen(dialog, width, height, minimum=(480, 360)):
    """Open a dialog at the requested size, shrunk to whatever the screen can show.

    Dialogs stay resizable: a fixed size taller than the monitor hides the bottom
    buttons with no way to reach them.
    """
    screen = QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        width = min(width, int(available.width() * 0.95))
        height = min(height, int(available.height() * 0.90))
    dialog.setMinimumSize(min(minimum[0], width), min(minimum[1], height))
    dialog.resize(width, height)


def wrap_in_scroll_area(widget, minimum_height):
    """Keep a panel usable on short screens by scrolling instead of clipping it.

    The panel keeps a minimum height, so the scrollbar only shows up once the window
    is too short to lay the controls out in full.
    """
    widget.setMinimumHeight(minimum_height)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    return scroll


def make_dialog_scrollable(dialog, content_layout, footer_layout=None,
                           preferred_size=(600, 900)):
    """Move a dialog's content into a scroll area and pin the buttons underneath it."""
    content = QWidget()
    content.setLayout(content_layout)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    outer = QVBoxLayout(dialog)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    outer.addWidget(scroll, 1)

    if footer_layout is not None:
        footer_layout.setContentsMargins(12, 8, 12, 8)
        footer = QWidget()
        footer.setLayout(footer_layout)
        footer.setStyleSheet("background-color: #f8f9fa; border-top: 1px solid #dee2e6;")
        outer.addWidget(footer)

    fit_dialog_to_screen(dialog, *preferred_size)


class _SortableTableItem(QTableWidgetItem):
    """Table cell that sorts on a numeric key instead of its display string.

    Without this, '10.0' sorts before '9.0' and '-' (a dropped pair) lands in
    the middle of the numbers.
    """

    def __init__(self, text, sort_key):
        super().__init__(text)
        self._sort_key = sort_key

    def __lt__(self, other):
        if isinstance(other, _SortableTableItem):
            return self._sort_key < other._sort_key
        return super().__lt__(other)


class SortableTableDialog(QDialog):
    """Read-only result table with numeric sorting, clipboard copy and CSV export.

    Replaces the matplotlib table figures, whose cells could not be selected or
    sorted and clipped their columns once the row count grew.

    Rows are sequences of (display_text, sort_key) pairs. column_color, when
    given, maps a column index to a (header_colour, cell_colour) pair.
    """

    def __init__(self, columns, rows, *, title, heading, subtitle,
                 column_color=None, csv_name="table.csv", footer_rows=None,
                 preferred_size=(1400, 760), parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._columns = list(columns)
        self._column_color = column_color
        self._csv_name = csv_name
        # Plain string rows appended below the table on copy/export. Aggregates
        # belong here rather than in the table itself, where sorting would
        # shuffle them in among the data rows.
        self._footer_rows = [list(row) for row in (footer_rows or [])]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        heading_label = QLabel(heading)
        heading_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1b338c;")
        layout.addWidget(heading_label)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("font-size: 12px; color: #666;")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)

        self.table = self._build_table(rows)
        layout.addWidget(self.table, 1)

        layout.addLayout(self._build_button_row())

        fit_dialog_to_screen(self, *preferred_size, minimum=(760, 400))

    def _colors_for(self, column_index):
        if self._column_color is None:
            return None, None
        return self._column_color(column_index)

    def _build_table(self, rows):
        table = QTableWidget(len(rows), len(self._columns))
        table.setHorizontalHeaderLabels(self._columns)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setStyleSheet(
            "QTableWidget { gridline-color: #ffffff; font-size: 12px; }"
            "QTableWidget::item { padding: 4px 8px; }"
        )

        for row_index, row in enumerate(rows):
            for column_index, (text, sort_key) in enumerate(row):
                item = _SortableTableItem(text, sort_key)
                _, cell_color = self._colors_for(column_index)
                if cell_color is not None:
                    item.setBackground(cell_color)
                if column_index == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                else:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row_index, column_index, item)

        for column_index in range(len(self._columns)):
            header_item = table.horizontalHeaderItem(column_index)
            header_color, _ = self._colors_for(column_index)
            if header_color is not None:
                header_item.setBackground(header_color)
            font = header_item.font()
            font.setBold(True)
            header_item.setFont(font)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        table.setSortingEnabled(True)
        # Enabling sorting applies the header's default indicator, which is not
        # guaranteed to be ascending — pin the initial order to the first column.
        table.sortItems(0, Qt.SortOrder.AscendingOrder)
        return table

    def _build_button_row(self):
        row = QHBoxLayout()

        copy_button = _action_button("📋 Copy", "#6c757d", "#5a6268")
        copy_button.clicked.connect(self.copy_to_clipboard)
        row.addWidget(copy_button)

        export_button = _action_button("💾 Export CSV", "#17a2b8", "#138496")
        export_button.clicked.connect(self.export_csv)
        row.addWidget(export_button)

        row.addStretch()

        close_button = _action_button("Close", "#1b338c", "#162f7d")
        close_button.clicked.connect(self.accept)
        row.addWidget(close_button)
        return row

    def _table_as_rows(self):
        """Header, body in the order currently displayed, then any footer rows."""
        out = [list(self._columns)]
        for row_index in range(self.table.rowCount()):
            out.append([
                self.table.item(row_index, column_index).text()
                for column_index in range(self.table.columnCount())
            ])
        out.extend(self._footer_rows)
        return out

    def copy_to_clipboard(self):
        text = "\n".join("\t".join(row) for row in self._table_as_rows())
        QApplication.clipboard().setText(text)

    def export_csv(self):
        import csv

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Table", self._csv_name,
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        # utf-8-sig so Excel renders the '→', '²' and '°' in the headers.
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerows(self._table_as_rows())
        QMessageBox.information(self, "Export Successful", f"Saved to:\n{path}")


def _m2m_column_color(column_index):
    """Grey label, orange for the Before block, blue for the After block."""
    if column_index == 0:
        return QColor('#d9d9d9'), QColor('#f0f0f0')
    if column_index < 6:
        return QColor('#f4a261'), QColor('#fde2cf')
    return QColor('#4ea8de'), QColor('#dceeff')


class M2MEdgeStatisticsDialog(SortableTableDialog):
    """Before/after view of the min_pair_obs filter, one row per marker pair."""

    def __init__(self, columns, rows, after_min_pair_count, parent=None):
        super().__init__(
            columns, rows,
            title="M2M Edge Statistics",
            heading="M2M Edge Statistics — Before vs After min_pair_obs Filter",
            subtitle=(f"After columns require N ≥ {after_min_pair_count}   |   "
                      f"{len(rows)} marker pairs   |   click a column header to sort"),
            column_color=_m2m_column_color,
            csv_name="m2m_edge_statistics.csv",
            parent=parent,
        )


class CameraParameterDialog(QDialog):
    """Camera intrinsic parameters + distortion coefficients input dialog (tab-based)"""
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config
        # Work on a copy of the vehicle camera slots: tabs are rebuilt from
        # this list, so the real config stays intact until the caller applies
        # the dialog result on OK — Cancel (or merely lowering the camera
        # count spin box) can no longer destroy saved intrinsics/image lists.
        import copy
        self._vehicle_configs = (copy.deepcopy(config.vehicle_camera_configs)
                                 if config else [])
        self.parent_window = parent
        self.setWindowTitle("Input Camera Parameters")
        self.setModal(True)
        fit_dialog_to_screen(self, 700, 800)

        layout = QVBoxLayout(self)

        # === Vehicle Camera Count Control ===
        count_group = QGroupBox("🚗 Vehicle Camera Management")
        count_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        count_layout = QHBoxLayout()

        count_label = QLabel("Number of Vehicle Cameras:")
        count_label.setStyleSheet("font-size: 13px; font-weight: normal;")
        count_layout.addWidget(count_label)

        self.camera_count_spin = QSpinBox()
        self.camera_count_spin.setMinimum(1)
        self.camera_count_spin.setMaximum(10)
        self.camera_count_spin.setValue(config.num_vehicle_cameras if config else 4)
        self.camera_count_spin.setMinimumWidth(80)
        self.camera_count_spin.setMinimumHeight(35)
        self.camera_count_spin.setStyleSheet("""
            QSpinBox {
                font-size: 14px;
                padding: 5px 10px;
                min-width: 80px;
                min-height: 35px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                height: 17px;
            }
        """)
        self.camera_count_spin.valueChanged.connect(self.on_camera_count_changed)
        count_layout.addWidget(self.camera_count_spin)

        count_layout.addStretch()
        count_group.setLayout(count_layout)
        layout.addWidget(count_group)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(_TAB_WIDGET_STYLE)

        # Store input fields by camera
        self.camera_inputs = {}

        # Create initial tabs
        self._create_all_tabs()

        layout.addWidget(self.tab_widget)

        layout.addLayout(_dialog_button_row(self._validate_and_accept, self.reject))

    def _create_all_tabs(self):
        """Create all tabs (Environment + Wheel + Vehicle cameras)"""
        # Remove all existing tabs
        while self.tab_widget.count() > 0:
            self.tab_widget.removeTab(0)

        self.camera_inputs.clear()

        # Environment Camera tab
        env_intrinsic = self.config.camera_params if self.config else None
        env_dist = self.config.dist_coeffs if self.config else None
        env_tab = self._create_camera_tab("Environment", env_intrinsic, env_dist)
        self.tab_widget.addTab(env_tab, "🚗 Environment cam")

        # Wheel Camera tab (with Copy button)
        wheel_intrinsic = self.config.wheel_camera_params if self.config else None
        wheel_dist = self.config.wheel_dist_coeffs if self.config else None
        wheel_tab = self._create_camera_tab_with_copy("Wheel", wheel_intrinsic, wheel_dist)
        self.tab_widget.addTab(wheel_tab, "🛞 Wheel cam")

        # Vehicle Camera tabs (dynamically created)
        num_vehicles = self.camera_count_spin.value()
        for i in range(num_vehicles):
            if i < len(self._vehicle_configs):
                intrinsic = self._vehicle_configs[i]['intrinsic']
                dist_coeffs = self._vehicle_configs[i]['dist_coeffs']
            else:
                # Brand-new slot: leave the fields empty. get_parameters()
                # skips validation for image-less tabs, but a tab that gets
                # images MUST have real intrinsics typed in — prefilling
                # numbers here would let fx=fy=1.0 slip into the config.
                intrinsic = None
                dist_coeffs = None

            cam_tab = self._create_camera_tab(f"vehicle_{i}", intrinsic, dist_coeffs)
            self.tab_widget.addTab(cam_tab, f"📷 Cam{i}")

            # Restore saved image info
            if i < len(self._vehicle_configs):
                saved_images = self._vehicle_configs[i].get('images', [])
                if saved_images:
                    self.camera_inputs[f'vehicle_{i}']['images'] = saved_images
                    label = self.camera_inputs[f'vehicle_{i}']['image_info']
                    label.setText(self._image_info_text(saved_images))
                    label.setToolTip("\n".join(saved_images))
                    self.camera_inputs[f'vehicle_{i}']['image_info'].setStyleSheet("""
                        font-size: 12px;
                        padding: 8px;
                        background-color: #e8f5e9;
                        border-radius: 5px;
                        margin-top: 5px;
                        color: #2e7d32;
                    """)

    def on_camera_count_changed(self, value):
        """Called when camera count changes"""
        # Rebuild tabs only — the config is applied by the caller on OK, so
        # lowering the count here (or pressing Cancel afterwards) no longer
        # destroys saved camera slots. Slots beyond the saved list start
        # empty; shrinking then re-growing restores the saved values.

        # Recreate tabs
        current_tab_index = self.tab_widget.currentIndex()
        self._create_all_tabs()

        # Restore previous tab index if possible (Environment/Wheel tabs)
        if current_tab_index < 2:
            self.tab_widget.setCurrentIndex(current_tab_index)

    def _create_camera_tab(self, cam_id, intrinsic, dist_coeffs):
        """Create individual camera parameter input tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        form_layout = QFormLayout()
        label_style = "font-size: 13px; font-weight: 500;"

        # === Intrinsic Parameters ===
        intrinsic_label = QLabel("<b style='font-size: 15px;'>📷 Intrinsic Parameters</b>")
        form_layout.addRow(intrinsic_label)

        inputs = {}

        def _make_field(label_text, value, placeholder):
            lbl = QLabel(label_text)
            lbl.setStyleSheet(label_style)
            inp = QLineEdit()
            if value is not None:
                inp.setText(str(value))
            inp.setStyleSheet("padding: 6px; font-size: 13px;")
            inp.setPlaceholderText(placeholder)
            inp.setMinimumHeight(32)
            return lbl, inp

        intr = intrinsic
        fx_label, fx_input = _make_field("fx (focal length x):", intr[0] if intr is not None else None, "required")
        form_layout.addRow(fx_label, fx_input); inputs['fx'] = fx_input

        fy_label, fy_input = _make_field("fy (focal length y):", intr[1] if intr is not None else None, "required")
        form_layout.addRow(fy_label, fy_input); inputs['fy'] = fy_input

        cx_label, cx_input = _make_field("cx (principal point x):", intr[2] if intr is not None else None, "required")
        form_layout.addRow(cx_label, cx_input); inputs['cx'] = cx_input

        cy_label, cy_input = _make_field("cy (principal point y):", intr[3] if intr is not None else None, "required")
        form_layout.addRow(cy_label, cy_input); inputs['cy'] = cy_input

        scroll_layout.addLayout(form_layout)

        # === Camera Model + Distortion (vehicle cameras: dynamic; others: static) ===
        if cam_id.startswith('vehicle_'):
            # ── Camera Model Selector ──────────────────────────────────────────
            cam_idx = int(cam_id.split('_')[1])
            saved_model = 'pinhole'
            if cam_idx < len(self._vehicle_configs):
                saved_model = self._vehicle_configs[cam_idx].get('camera_model', 'pinhole')

            model_section_widget = QWidget()
            model_section_layout = QFormLayout(model_section_widget)
            model_section_layout.setContentsMargins(0, 8, 0, 0)

            model_title = QLabel("<b style='font-size: 15px;'>🎥 Camera Model</b>")
            model_section_layout.addRow(model_title)

            model_combo = QComboBox()
            model_combo.addItems(['Pinhole', 'Fisheye KB'])
            model_combo.setCurrentText('Fisheye KB' if saved_model == 'fisheye' else 'Pinhole')
            model_combo.setMinimumHeight(32)
            model_combo.setStyleSheet("""
                QComboBox {
                    padding: 4px 8px;
                    font-size: 13px;
                    background-color: white;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                }
                QComboBox::drop-down {
                    width: 20px;
                }
                QComboBox QAbstractItemView {
                    background-color: white;
                    selection-background-color: #1b338c;
                    selection-color: white;
                }
            """)
            model_lbl = QLabel("Model:")
            model_lbl.setStyleSheet(label_style)
            model_section_layout.addRow(model_lbl, model_combo)
            inputs['model_combo'] = model_combo
            scroll_layout.addWidget(model_section_widget)

            # ── Pinhole Distortion Widget ──────────────────────────────────────
            pinhole_widget = QWidget()
            pinhole_layout = QFormLayout(pinhole_widget)
            pinhole_layout.setContentsMargins(0, 8, 0, 0)

            ph_title = QLabel("<b style='font-size: 15px;'>🔧 Distortion Coefficients (Pinhole)</b>")
            pinhole_layout.addRow(ph_title)

            dist = dist_coeffs if (dist_coeffs is not None and len(dist_coeffs) == 5) else None

            def _ph(lbl_txt, val, ph):
                lbl = QLabel(lbl_txt); lbl.setStyleSheet(label_style)
                inp = QLineEdit()
                if val is not None: inp.setText(str(val))
                inp.setStyleSheet("padding: 6px; font-size: 13px;")
                inp.setPlaceholderText(ph); inp.setMinimumHeight(32)
                return lbl, inp

            _, k1_ph  = _ph("k1 (radial):",      dist[0] if dist is not None else None, "0.0")
            _, k2_ph  = _ph("k2 (radial):",      dist[1] if dist is not None else None, "0.0")
            _, k3_ph  = _ph("k3 (radial):",      dist[4] if dist is not None else None, "0.0")
            _, p1_ph  = _ph("p1 (tangential):",  dist[2] if dist is not None else None, "0.0")
            _, p2_ph  = _ph("p2 (tangential):",  dist[3] if dist is not None else None, "0.0")
            for lbl_txt, inp_w, key in [
                ("k1 (radial):", k1_ph, 'k1'), ("k2 (radial):", k2_ph, 'k2'),
                ("k3 (radial):", k3_ph, 'k3'), ("p1 (tangential):", p1_ph, 'p1'),
                ("p2 (tangential):", p2_ph, 'p2'),
            ]:
                lbl = QLabel(lbl_txt); lbl.setStyleSheet(label_style)
                pinhole_layout.addRow(lbl, inp_w)
                inputs[key] = inp_w
            scroll_layout.addWidget(pinhole_widget)

            # ── Fisheye KB Distortion Widget ───────────────────────────────────
            fisheye_widget = QWidget()
            fisheye_layout = QFormLayout(fisheye_widget)
            fisheye_layout.setContentsMargins(0, 8, 0, 0)

            fs_title = QLabel("<b style='font-size: 15px;'>🔧 Distortion Coefficients (Fisheye KB)</b>")
            fisheye_layout.addRow(fs_title)
            fs_hint = QLabel("<i style='color:#666;font-size:12px;'>θ_d = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹</i>")
            fisheye_layout.addRow(fs_hint)

            kb_dist = dist_coeffs if (dist_coeffs is not None and len(dist_coeffs) == 4) else None

            for lbl_txt, kb_val, key in [
                ("k1:", kb_dist[0] if kb_dist is not None else None, 'kb1'),
                ("k2:", kb_dist[1] if kb_dist is not None else None, 'kb2'),
                ("k3:", kb_dist[2] if kb_dist is not None else None, 'kb3'),
                ("k4:", kb_dist[3] if kb_dist is not None else None, 'kb4'),
            ]:
                lbl = QLabel(lbl_txt); lbl.setStyleSheet(label_style)
                inp = QLineEdit()
                if kb_val is not None: inp.setText(str(kb_val))
                inp.setStyleSheet("padding: 6px; font-size: 13px;")
                inp.setPlaceholderText("0.0"); inp.setMinimumHeight(32)
                fisheye_layout.addRow(lbl, inp)
                inputs[key] = inp
            scroll_layout.addWidget(fisheye_widget)

            # ── Toggle logic ───────────────────────────────────────────────────
            def _toggle_dist_model(model_text,
                                   _pw=pinhole_widget, _fw=fisheye_widget):
                _pw.setVisible(model_text == 'Pinhole')
                _fw.setVisible(model_text == 'Fisheye KB')

            model_combo.currentTextChanged.connect(_toggle_dist_model)
            _toggle_dist_model(model_combo.currentText())

        else:
            # ── Env / Wheel cameras: static pinhole distortion in form_layout ──
            dist_form_layout = QFormLayout()
            dist_form_layout.setContentsMargins(0, 8, 0, 0)

            distortion_label = QLabel("<b style='font-size: 15px;'>🔧 Distortion Coefficients</b>")
            dist_form_layout.addRow(distortion_label)

            dist = dist_coeffs
            for lbl_txt, val, key in [
                ("k1 (radial):",     dist[0] if dist is not None else None, 'k1'),
                ("k2 (radial):",     dist[1] if dist is not None else None, 'k2'),
                ("k3 (radial):",     dist[4] if dist is not None else None, 'k3'),
                ("p1 (tangential):", dist[2] if dist is not None else None, 'p1'),
                ("p2 (tangential):", dist[3] if dist is not None else None, 'p2'),
            ]:
                lbl = QLabel(lbl_txt); lbl.setStyleSheet(label_style)
                inp = QLineEdit()
                if val is not None: inp.setText(str(val))
                inp.setStyleSheet("padding: 6px; font-size: 13px;")
                inp.setPlaceholderText("required (0.0 = no distortion)")
                inp.setMinimumHeight(32)
                dist_form_layout.addRow(lbl, inp)
                inputs[key] = inp
            scroll_layout.addLayout(dist_form_layout)

        # === Add image selection section if vehicle camera ===
        if cam_id.startswith('vehicle_'):
            # Separator
            separator2 = QLabel()
            separator2.setFixedHeight(20)
            scroll_layout.addWidget(separator2)

            # Image Selection Section
            image_section_label = QLabel("<b style='font-size: 15px;'>📁 Image Selection</b>")
            scroll_layout.addWidget(image_section_label)

            select_images_button = QPushButton("📂 Select Images for this Camera")
            select_images_button.setMinimumHeight(40)
            select_images_button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: 500;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
            """)
            select_images_button.clicked.connect(lambda: self.select_camera_images(cam_id))
            scroll_layout.addWidget(select_images_button)

            # Image info label
            image_info_label = QLabel("No images selected")
            image_info_label.setStyleSheet("""
                font-size: 12px;
                padding: 8px;
                background-color: #f5f5f5;
                border-radius: 5px;
                margin-top: 5px;
                color: #666;
            """)
            image_info_label.setWordWrap(True)
            scroll_layout.addWidget(image_info_label)

            inputs['image_info'] = image_info_label
            inputs['images'] = []

        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Save input fields
        self.camera_inputs[cam_id] = inputs

        return tab

    def _create_camera_tab_with_copy(self, cam_id, intrinsic, dist_coeffs):
        """Create wheel camera parameter input tab (with Copy button)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        form_layout = QFormLayout()
        label_style = "font-size: 13px; font-weight: 500;"

        # Copy from Environment button (placed at top)
        copy_button = QPushButton("📋 Copy from Environment Camera")
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: #0288d1;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 13px;
                padding: 10px;
                margin-bottom: 10px;
            }
            QPushButton:hover {
                background-color: #0277bd;
            }
        """)
        copy_button.clicked.connect(self.copy_env_to_wheel)
        scroll_layout.addWidget(copy_button)

        # === Intrinsic Parameters ===
        intrinsic_label = QLabel("<b style='font-size: 15px;'>📷 Intrinsic Parameters</b>")
        form_layout.addRow(intrinsic_label)

        inputs = {}

        def _make_field_w(label_text, value, placeholder):
            lbl = QLabel(label_text)
            lbl.setStyleSheet(label_style)
            inp = QLineEdit()
            if value is not None:
                inp.setText(str(value))
            inp.setStyleSheet("padding: 6px; font-size: 13px;")
            inp.setPlaceholderText(placeholder)
            inp.setMinimumHeight(32)
            return lbl, inp

        intr = intrinsic
        fx_label, fx_input = _make_field_w("fx (focal length x):", intr[0] if intr is not None else None, "required")
        form_layout.addRow(fx_label, fx_input); inputs['fx'] = fx_input

        fy_label, fy_input = _make_field_w("fy (focal length y):", intr[1] if intr is not None else None, "required")
        form_layout.addRow(fy_label, fy_input); inputs['fy'] = fy_input

        cx_label, cx_input = _make_field_w("cx (principal point x):", intr[2] if intr is not None else None, "required")
        form_layout.addRow(cx_label, cx_input); inputs['cx'] = cx_input

        cy_label, cy_input = _make_field_w("cy (principal point y):", intr[3] if intr is not None else None, "required")
        form_layout.addRow(cy_label, cy_input); inputs['cy'] = cy_input

        # Separator
        separator = QLabel()
        separator.setFixedHeight(10)
        form_layout.addRow(separator)

        # === Distortion Coefficients ===
        distortion_label = QLabel("<b style='font-size: 15px;'>🔧 Distortion Coefficients</b>")
        form_layout.addRow(distortion_label)

        # dist is stored in OpenCV order: [k1, k2, p1, p2, k3]
        # Display in UI order: k1, k2, k3, p1, p2
        dist = dist_coeffs
        k1_label, k1_input = _make_field_w("k1 (radial):", dist[0] if dist is not None else None, "required (0.0 = no distortion)")
        form_layout.addRow(k1_label, k1_input); inputs['k1'] = k1_input

        k2_label, k2_input = _make_field_w("k2 (radial):", dist[1] if dist is not None else None, "required (0.0 = no distortion)")
        form_layout.addRow(k2_label, k2_input); inputs['k2'] = k2_input

        k3_label, k3_input = _make_field_w("k3 (radial):", dist[4] if dist is not None else None, "required (0.0 = no distortion)")
        form_layout.addRow(k3_label, k3_input); inputs['k3'] = k3_input

        p1_label, p1_input = _make_field_w("p1 (tangential):", dist[2] if dist is not None else None, "required (0.0 = no distortion)")
        form_layout.addRow(p1_label, p1_input); inputs['p1'] = p1_input

        p2_label, p2_input = _make_field_w("p2 (tangential):", dist[3] if dist is not None else None, "required (0.0 = no distortion)")
        form_layout.addRow(p2_label, p2_input); inputs['p2'] = p2_input

        scroll_layout.addLayout(form_layout)

        # Add image selection UI if vehicle camera
        if cam_id.startswith('vehicle_'):
            separator2 = QLabel()
            separator2.setFixedHeight(20)
            scroll_layout.addWidget(separator2)

            # === Image Selection ===
            image_section_label = QLabel("<b style='font-size: 15px;'>📁 Image Selection</b>")
            scroll_layout.addWidget(image_section_label)

            # Image selection button
            select_images_button = QPushButton("📂 Select Images for this Camera")
            select_images_button.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 13px;
                    padding: 10px;
                    margin-top: 5px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
            select_images_button.clicked.connect(lambda: self.select_camera_images(cam_id))
            scroll_layout.addWidget(select_images_button)

            # Display selected image info
            image_info_label = QLabel("No images selected")
            image_info_label.setStyleSheet("""
                font-size: 12px;
                padding: 8px;
                background-color: #f0f0f0;
                border-radius: 5px;
                margin-top: 5px;
                color: #666;
            """)
            image_info_label.setWordWrap(True)
            scroll_layout.addWidget(image_info_label)
            inputs['image_info'] = image_info_label

            # Store image paths (hidden)
            inputs['images'] = []

        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Save input fields
        self.camera_inputs[cam_id] = inputs

        return tab

    def copy_env_to_wheel(self):
        """Copy environment camera parameters to wheel camera"""
        env_inputs = self.camera_inputs['Environment']
        wheel_inputs = self.camera_inputs['Wheel']

        for key in ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'p1', 'p2']:
            wheel_inputs[key].setText(env_inputs[key].text())

        QMessageBox.information(self, "Success", "Environment parameters copied to Wheel camera!")

    @staticmethod
    def _image_info_text(paths):
        """Full paths of the selected images, so a restored config is verifiable."""
        if not paths:
            return "No images selected"
        lines = [f"✓ {len(paths)} image(s) selected:"]
        shown = list(paths)[:10]
        lines += [f"   {p}" for p in shown]
        if len(paths) > len(shown):
            lines.append(f"   ... (+{len(paths) - len(shown)} more)")
        return "\n".join(lines)

    def select_camera_images(self, cam_id):
        """Select image files for specific camera"""
        # Select multiple images with file dialog
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tiff)")

        # Set initial directory (project root)
        if self.config:
            file_dialog.setDirectory(self.config.project_root)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                # Save image paths
                self.camera_inputs[cam_id]['images'] = sorted(selected_files)

                # Update info label with the full paths, so what is about to be
                # saved into the config is visible verbatim.
                paths = sorted(selected_files)
                label = self.camera_inputs[cam_id]['image_info']
                label.setText(self._image_info_text(paths))
                label.setToolTip("\n".join(paths))
                label.setStyleSheet("""
                    font-size: 12px;
                    padding: 8px;
                    background-color: #e8f5e9;
                    border-radius: 5px;
                    margin-top: 5px;
                    color: #2e7d32;
                """)

                print(f"✓ Camera {cam_id}: {len(selected_files)} images selected")

    def _validate_and_accept(self):
        """Close only when the fields parse.

        get_parameters() already warns on invalid input; keeping the dialog
        open lets the user fix the field instead of losing every edit (accept
        first would close the dialog before the caller's validation ran).
        """
        if self.get_parameters() is not None:
            self.accept()

    def get_parameters(self):
        """Get parameters for all cameras — validates all required fields are filled"""
        def _parse_fields(inputs, field_keys, cam_label):
            """Parse float values from input fields; raises ValueError with field name if empty/invalid."""
            values = []
            for key in field_keys:
                text = inputs[key].text().strip()
                if not text:
                    raise ValueError(f"[{cam_label}] '{key}' is empty — please enter a value.")
                values.append(float(text))
            return values

        try:
            params = {}

            # Environment camera (required before Stage 1)
            env_inputs = self.camera_inputs['Environment']
            params['environment'] = {
                'intrinsic': _parse_fields(env_inputs, ['fx', 'fy', 'cx', 'cy'], 'Environment'),
                'dist_coeffs': _parse_fields(env_inputs, ['k1', 'k2', 'k3', 'p1', 'p2'], 'Environment')
            }

            # Wheel camera (required before Stage 2)
            wheel_inputs = self.camera_inputs['Wheel']
            params['wheel'] = {
                'intrinsic': _parse_fields(wheel_inputs, ['fx', 'fy', 'cx', 'cy'], 'Wheel'),
                'dist_coeffs': _parse_fields(wheel_inputs, ['k1', 'k2', 'k3', 'p1', 'p2'], 'Wheel')
            }

            # Vehicle cameras (required before Stage 3)
            params['vehicles'] = []
            num_vehicles = self.camera_count_spin.value()
            for i in range(num_vehicles):
                vehicle_inputs = self.camera_inputs[f'vehicle_{i}']

                if not vehicle_inputs.get('images'):
                    # No images → the caller discards this slot anyway, so an
                    # empty freshly-added tab must not block OK. Placeholder
                    # values keep the list aligned for slot compaction and
                    # are never written to the config.
                    params['vehicles'].append({
                        'intrinsic':    [1.0, 1.0, 0.0, 0.0],
                        'dist_coeffs':  [0.0, 0.0, 0.0, 0.0, 0.0],
                        'camera_model': 'pinhole',
                        'images':       []
                    })
                    continue

                model_combo = vehicle_inputs.get('model_combo')
                camera_model = 'fisheye' if (model_combo and model_combo.currentText() == 'Fisheye KB') else 'pinhole'

                if camera_model == 'fisheye':
                    dist = _parse_fields(vehicle_inputs, ['kb1', 'kb2', 'kb3', 'kb4'], f'Vehicle Cam{i} (Fisheye)')
                else:
                    dist = _parse_fields(vehicle_inputs, ['k1', 'k2', 'k3', 'p1', 'p2'], f'Vehicle Cam{i} (Pinhole)')

                vehicle_param = {
                    'intrinsic':     _parse_fields(vehicle_inputs, ['fx', 'fy', 'cx', 'cy'], f'Vehicle Cam{i}'),
                    'dist_coeffs':   dist,
                    'camera_model':  camera_model,
                    'images':        vehicle_inputs.get('images', [])
                }
                params['vehicles'].append(vehicle_param)

            return params

        except ValueError as e:
            QMessageBox.warning(self, "Input Error",
                f"Please fill in all required fields before saving.\n\n{e}")
            return None


def _build_grid_layout_diagram_pixmap():
    """
    Render the Board Grid Layout example on top of an actual captured 2x2
    AprilGrid photo (marker_info.png), so 'marker_size' and 'tag_gap' point
    at the real regions they measure instead of an abstract shape.

    marker_info.png is a front-facing 2x2 tagStandard41h12 board image. The
    board fills the image, so its annotation geometry is derived directly
    from the current bitmap dimensions instead of using capture-specific
    perspective coordinates.
    """
    import io
    from matplotlib.image import imread
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    from PySide6.QtGui import QPixmap

    photo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'marker_info.png')
    img = imread(photo_path)
    # Keep the real capture visible, but mute it slightly so the geometry
    # annotations remain readable instead of competing with the tag pattern.
    display_img = img.copy()
    display_img[..., :3] = display_img[..., :3] * 0.72 + 0.28

    # The flat board artwork fills marker_info.png edge to edge.
    image_height, image_width = img.shape[:2]
    TL = (0.0, 0.0)
    TR = (float(image_width - 1), 0.0)
    BL = (0.0, float(image_height - 1))
    BR = (float(image_width - 1), float(image_height - 1))

    def bilerp(u, v):
        x = (1-u)*(1-v)*TL[0] + u*(1-v)*TR[0] + (1-u)*v*BL[0] + u*v*BR[0]
        y = (1-u)*(1-v)*TL[1] + u*(1-v)*TR[1] + (1-u)*v*BL[1] + u*v*BR[1]
        return x, y

    def quad(u0, u1, v0, v1):
        return [bilerp(u0, v0), bilerp(u1, v0), bilerp(u1, v1), bilerp(u0, v1)]

    # Each tag occupies 9 of the 20.25-cell board side (4/9); the remaining
    # 1/9 in the middle is the true printed gap between the two tags' outer
    # (border-included) squares. tagStandard41h12's detector-reported square
    # is the centred 5/9 of a tag's own 9-cell square (2/9 border each side).
    OUTER = [(0.0, 4/9), (5/9, 1.0)]
    RED = [(a + (b - a) * 2/9, b - (b - a) * 2/9) for a, b in OUTER]

    slots = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}  # local_index -> (col, row)
    first_tag_id = 10

    margin = 40
    x0, y0 = min(TL[0], BL[0]) - margin, min(TL[1], TR[1]) - margin
    x1, y1 = max(TR[0], BR[0]) + margin, max(BL[1], BR[1]) + margin

    fig = Figure(figsize=(4.2, 4.05), dpi=110)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.imshow(display_img)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis('off')

    # Full board outline.
    ax.add_patch(Polygon([TL, TR, BR, BL], closed=True, fill=False,
                          edgecolor='#00bcd4', linewidth=2.2))

    def label(x, y, text, color, fontsize=9.5, fontweight='bold'):
        ax.text(x, y, text, ha='center', va='center', color=color,
                fontsize=fontsize, fontweight=fontweight,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.88,
                          edgecolor='none'))

    red_quads = {}
    for local_index, (col, row) in slots.items():
        # Printed pattern incl. border — NOT what marker_size/tag_gap measure.
        ax.add_patch(Polygon(quad(*OUTER[col], *OUTER[row]), closed=True, fill=False,
                              edgecolor='#fbc02d', linewidth=1.4, linestyle=(0, (5, 4))))

        rq = quad(*RED[col], *RED[row])
        red_quads[local_index] = rq
        ax.add_patch(Polygon(rq, closed=True, fill=False,
                              edgecolor='#e53935', linewidth=2.4))

        cx, cy = np.mean([p[0] for p in rq]), np.mean([p[1] for p in rq])
        tag_id = first_tag_id + local_index
        label(cx, cy, f"ID {tag_id}\nslot {local_index}", '#172033', fontsize=8.5)

    # marker_size: half-width of the top-left red square. Draw the dimension
    # line below the boundary so the <-> arrowheads do not disappear into it.
    tl_rq = red_quads[0]  # [top-left, top-right, bottom-right, bottom-left]
    b_mid = ((tl_rq[3][0] + tl_rq[2][0]) / 2, (tl_rq[3][1] + tl_rq[2][1]) / 2)
    b_right = tl_rq[2]
    dimension_y = (b_mid[1] + b_right[1]) / 2 + 55
    dimension_start = (b_mid[0], dimension_y)
    dimension_end = (b_right[0], dimension_y)
    ax.plot([b_mid[0], dimension_start[0]], [b_mid[1], dimension_start[1]],
            color='#e53935', lw=1.0)
    ax.plot([b_right[0], dimension_end[0]], [b_right[1], dimension_end[1]],
            color='#e53935', lw=1.0)
    ax.annotate('', xy=dimension_end, xytext=dimension_start,
                arrowprops=dict(arrowstyle='<->', color='#e53935', lw=1.8))
    label((dimension_start[0] + dimension_end[0]) / 2, dimension_y + 75,
          'Tag Edge Size (Half)', '#c62828', fontsize=7.0)

    # tag_gap: edge-to-edge distance between the detected boundaries of two
    # neighbouring tags. Keep it separate from the printed-pattern boundary.
    top_left_rq, top_right_rq = red_quads[0], red_quads[1]
    gap_start = (
        (top_left_rq[1][0] + top_left_rq[2][0]) / 2,
        (top_left_rq[1][1] + top_left_rq[2][1]) / 2,
    )
    gap_end = (
        (top_right_rq[0][0] + top_right_rq[3][0]) / 2,
        (top_right_rq[0][1] + top_right_rq[3][1]) / 2,
    )
    ax.annotate('', xy=gap_end, xytext=gap_start,
                arrowprops=dict(arrowstyle='<->', color='#d500f9', lw=1.8))
    label((gap_start[0] + gap_end[0]) / 2, gap_start[1] + 65,
          'Tag Edge Gap', '#8e24aa', fontsize=7.2)

    # Keep every visual convention in one legend. board_id is a grouping rule,
    # not a point on the image, so it belongs here rather than in an arrow callout.
    legend_handles = [
        Line2D([0], [0], color='#e53935', lw=2.4,
               label='Detected Tag Boundary'),
        Line2D([0], [0], color='#00bcd4', lw=2.2,
               label='AprilGrid Boundary'),
        Line2D([0], [0], color='#d500f9', lw=2.0,
               label='Tag Edge Gap'),
        Line2D([0], [0], color='#fbc02d', lw=1.4, linestyle=(0, (5, 4)),
               label='AprilTag Boundary'),
        Line2D([0], [0], color='#166534', marker='s', markersize=5,
               markerfacecolor='white', lw=0, label='board_id = min(tag_ids)'),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', ncol=2, fontsize=6.4,
        frameon=True, fancybox=True, framealpha=0.95, edgecolor='#c7cdd6',
        handlelength=2.6, columnspacing=1.2, handletextpad=0.6,
        bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='white')
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    return pixmap


class MarkerConfigDialog(QDialog):
    """Marker + AprilTag settings dialog"""
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Marker & AprilTag Configuration")
        self.setModal(True)

        layout = QVBoxLayout()

        def section_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; margin-top: 8px;")
            return lbl

        def field_input(value, height=35):
            w = QLineEdit(str(value))
            w.setMinimumHeight(height)
            return w

        # ── AprilTag sizes ────────────────────────────────────────────
        layout.addWidget(section_label("📐 AprilTag Sizes"))
        size_form = QFormLayout()

        self.size_input = field_input(config.marker_size if config else 0.4)
        self.size_input.setToolTip(
            "Half the side of the square the detector reports for environment\n"
            "markers.\n"
            "\n"
            "For tagStandard41h12, measure the white centre square (5/9 of the\n"
            "printed pattern).")
        size_form.addRow(QLabel("Env Detected Tag Edge Size (half, m):"), self.size_input)

        self.size2_input = field_input(config.marker_size2 if config else 0.2)
        self.size2_input.setToolTip(
            "Half the side of the square the detector reports for wheel markers.")
        size_form.addRow(QLabel("Wheel Detected Tag Edge Size (half, m):"), self.size2_input)

        layout.addLayout(size_form)

        # ── Board IDs ───────────────────────────────────────────────
        layout.addWidget(section_label("🔢 Board IDs"))
        id_form = QFormLayout()

        self.first_id_input = field_input(config.first_marker_id if config else 1)
        self.first_id_input.setToolTip(
            "Which already-detected board_id becomes the world origin.\n"
            "\n"
            "board_id = min(tag_ids_on_board), e.g. tags [22, 23, 24, 25] -> 22.\n"
            "This does not control tag grouping: environment grouping uses\n"
            "'Env Tag ID Start' below, wheel grouping uses 'Vehicle Wheel Board IDs'.\n"
            "\n"
            "Must match a board_id that is actually detected, otherwise the smallest\n"
            "detected board_id is used instead.")
        id_form.addRow(QLabel("World-Origin Board ID (pick existing board_id):"), self.first_id_input)

        self.vehicle_input = field_input(
            ','.join(map(str, config.vehicle_markers)) if config else "29,30,32,31")
        self.vehicle_input.setPlaceholderText("e.g., 29,30,32,31")
        self.vehicle_input.setToolTip(
            "Which board_ids identify the four wheel boards, in the order\n"
            "[FL, RL, RR, FR].\n"
            "\n"
            "board_id = min(tag_ids_on_board), not an individual AprilTag ID. These\n"
            "values anchor wheel tag grouping directly, so there is no separate\n"
            "wheel first-tag-id field.")
        id_form.addRow(QLabel("Vehicle Wheel Board IDs [fl,rl,rr,fr]:"), self.vehicle_input)

        excluded_ids = getattr(config, 'excluded_apriltag_ids', []) if config else []
        self.excluded_ids_input = field_input(
            ", ".join(str(i) for i in excluded_ids))
        self.excluded_ids_input.setPlaceholderText("e.g. 18, 82  (comma-separated)")
        self.excluded_ids_input.setToolTip(
            "Boards whose presence disqualifies an image: any env or wheel image in\n"
            "which one of them is detected is skipped entirely, not just that board.\n"
            "Stage 3 vehicle images are not filtered.\n"
            "\n"
            "Any tag ID printed on the board works - it is mapped to that board's ID\n"
            "on OK (e.g. 11 -> 10 for a 2x2 board starting at 10).\n"
            "\n"
            "Leave empty to disable.")
        id_form.addRow(QLabel("Excluded Board IDs (skip images):"), self.excluded_ids_input)

        layout.addLayout(id_form)

        # ── AprilTag family ─────────────────────────────────────────
        layout.addWidget(section_label("🧩 AprilTag Family"))
        fam_form = QFormLayout()

        self.family_combo = QComboBox()
        self.family_combo.addItems(list(APRILTAG_FAMILY_GEOMETRY))
        self.family_combo.setMinimumHeight(35)
        current_family = str(getattr(config, 'apriltag_family', 'tagStandard41h12')
                             if config else 'tagStandard41h12')
        self.family_combo.setCurrentText(
            current_family if current_family in APRILTAG_FAMILY_GEOMETRY else 'tagStandard41h12')
        fam_form.addRow(QLabel("AprilTag Family:"), self.family_combo)
        layout.addLayout(fam_form)

        self.family_combo.currentTextChanged.connect(self._update_geometry_tooltip)
        self._update_geometry_tooltip()

        # ── AprilGrid layout ───────────────────────────────────────
        layout.addWidget(section_label("🗂 AprilGrid Layout"))

        diagram_label = QLabel()
        diagram_label.setPixmap(_build_grid_layout_diagram_pixmap())
        diagram_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(diagram_label)

        grid_form = QFormLayout()

        def gv(attr, default):
            return getattr(config, attr, default) if config else default

        self.env_rows_input      = field_input(gv('env_grid_rows', 2))
        self.env_cols_input      = field_input(gv('env_grid_cols', 2))
        self.env_gap_input       = field_input(gv('env_tag_gap', 0.246914))
        self.env_first_tag_input = field_input(gv('env_first_tag_id', 10))
        self.env_gap_input.setToolTip(
            "Edge-to-edge distance between neighbouring detected tag squares, in\n"
            "metres.\n"
            "\n"
            "This is neither the centre-to-centre pitch nor the printed-pattern gap.")
        self.env_first_tag_input.setToolTip(
            "The AprilTag ID of the very first slot (top-left) on the first\n"
            "environment board. This is what controls how tags are grouped into\n"
            "boards.\n"
            "\n"
            "    board_id = env_first_tag_id + group_index * (rows * cols)\n"
            "\n"
            "e.g. with 10 and a 2x2 grid the boards are M10, M14, M18, ...\n"
            "\n"
            "Different from 'World-Origin Board ID' in the Board IDs section.")

        self.wheel_rows_input = field_input(gv('wheel_grid_rows', 2))
        self.wheel_cols_input = field_input(gv('wheel_grid_cols', 2))
        self.wheel_gap_input  = field_input(gv('wheel_tag_gap', 0.154321))
        self.wheel_gap_input.setToolTip(
            "Edge-to-edge distance between neighbouring detected tag squares, in\n"
            "metres.\n"
            "\n"
            "This is neither the centre-to-centre pitch nor the printed-pattern gap.")
        # No "wheel first tag id" field: wheel board_ids are looked up
        # directly from 'Vehicle Wheel Board IDs' above, so there is no
        # separate anchor value that could get out of sync.

        grid_form.addRow(QLabel("Env Rows:"),         self.env_rows_input)
        grid_form.addRow(QLabel("Env Cols:"),         self.env_cols_input)
        grid_form.addRow(QLabel("Env Tag Edge Gap (m):"), self.env_gap_input)
        grid_form.addRow(QLabel("Env Tag ID Start:"), self.env_first_tag_input)
        grid_form.addRow(QLabel("Wheel Rows:"),        self.wheel_rows_input)
        grid_form.addRow(QLabel("Wheel Cols:"),        self.wheel_cols_input)
        grid_form.addRow(QLabel("Wheel Tag Edge Gap (m):"), self.wheel_gap_input)
        layout.addLayout(grid_form)

        # ── Info ────────────────────────────────────────────────────
        info = QLabel(
            "💡 Env board_id = env_first_tag_id + group × (rows×cols).\n"
            "Wheel board_id is looked up directly from 'Vehicle Wheel Board IDs' above\n"
            "(no separate first-tag-id needed for wheels).")
        info.setStyleSheet(
            "font-size: 12px; color: #666; padding: 8px;"
            " background-color: #fff3cd; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── Buttons ─────────────────────────────────────────────────
        btn_layout = _dialog_button_row(self.accept_and_save, self.reject)
        make_dialog_scrollable(self, layout, btn_layout, preferred_size=(640, 1300))

    def _update_geometry_tooltip(self):
        family = self.family_combo.currentText()
        geometry = APRILTAG_FAMILY_GEOMETRY[family]
        border, total = geometry['width_at_border'], geometry['total_width']
        square = ("white centre square" if geometry['reversed_border'] else "outer black square")
        self.family_combo.setToolTip(
            f"How to measure the tag size for {family}.\n"
            "\n"
            f"Measure the {square}: it spans {border}/{total} cells\n"
            f"({border / total * 100:.1f}% of the printed pattern).\n"
            "\n"
            "Enter half of that side length in the size fields above.")

    def accept_and_save(self):
        try:
            self.config.marker_size  = float(self.size_input.text())
            self.config.marker_size2 = float(self.size2_input.text())
            self.config.first_marker_id = int(self.first_id_input.text())
            vehicle_str = self.vehicle_input.text().strip()
            self.config.vehicle_markers = [int(x.strip()) for x in vehicle_str.split(',')]
            raw_ids = self.excluded_ids_input.text().strip()
            entered_exclusions = ([int(x.strip()) for x in raw_ids.split(',') if x.strip()]
                                  if raw_ids else [])
            self.config.apriltag_family = self.family_combo.currentText()

            rows_cols = {
                'env_grid_rows':   int(self.env_rows_input.text()),
                'env_grid_cols':   int(self.env_cols_input.text()),
                'wheel_grid_rows': int(self.wheel_rows_input.text()),
                'wheel_grid_cols': int(self.wheel_cols_input.text()),
            }
            for name, val in rows_cols.items():
                if val < 1:
                    raise ValueError(f"{name} must be 1 or greater.")
            gaps = {
                'env_tag_gap':   float(self.env_gap_input.text()),
                'wheel_tag_gap': float(self.wheel_gap_input.text()),
            }
            for name, val in gaps.items():
                if val < 0:
                    raise ValueError(f"{name} must be 0 or greater.")
            first_ids = {
                'env_first_tag_id': int(self.env_first_tag_input.text()),
            }
            for name, val in first_ids.items():
                if val < 0:
                    raise ValueError(f"{name} must be 0 or greater.")
            for name, val in {**rows_cols, **gaps, **first_ids}.items():
                setattr(self.config, name, val)

            # Map every entered ID onto its board_id, so any tag printed on a
            # board works (e.g. 11 → 10 for a 2x2 board starting at 10). The
            # detection filter only ever sees board_ids, so a raw tag ID would
            # otherwise be silently ignored. Runs after the grid values above
            # are applied, since the grouping depends on them.
            from fgo_system import build_apriltag_config, resolve_board_layout
            apriltag_config = build_apriltag_config(self.config)
            board_ids, remapped = [], []
            for tag_id in entered_exclusions:
                board_id = resolve_board_layout(
                    tag_id, apriltag_config, self.config.vehicle_markers)[0]
                if board_id not in board_ids:
                    board_ids.append(board_id)
                if board_id != tag_id:
                    remapped.append(f"{tag_id} → {board_id}")
            self.config.excluded_apriltag_ids = sorted(board_ids)
            if remapped:
                self.excluded_ids_input.setText(
                    ", ".join(str(i) for i in sorted(board_ids)))
                QMessageBox.information(
                    self, "Excluded Board IDs",
                    "Entered tag IDs were mapped to the board they belong to:\n\n"
                    + "\n".join(remapped)
                    + f"\n\nStored exclusions: {sorted(board_ids)}")

            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {str(e)}")


class UncertaintyDialog(QDialog):
    """Uncertainty settings dialog"""
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Uncertainty Configuration")
        self.setModal(True)

        layout = QVBoxLayout()

        form_layout = QFormLayout()
        label_style = "font-size: 14px; font-weight: 500;"

        def add_section(title):
            section = QLabel(title)
            section.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #333; margin-top: 8px;")
            form_layout.addRow(section)

        # ── Geometric Consistency thresholds (Stage 3 observation filter) ──
        add_section("📐 Geometric Consistency (Stage 3 Vehicle Obs. Filter)")

        geo_skip_rot_label = QLabel("Geo Skip Rot (deg):")
        geo_skip_rot_label.setStyleSheet(label_style)
        self.geo_skip_rot_input = QLineEdit()
        self.geo_skip_rot_input.setText(str(config.geo_skip_rot_deg) if config else "4.0")
        self.geo_skip_rot_input.setMinimumHeight(35)
        form_layout.addRow(geo_skip_rot_label, self.geo_skip_rot_input)

        geo_skip_pos_label = QLabel("Geo Skip Pos (m):")
        geo_skip_pos_label.setStyleSheet(label_style)
        self.geo_skip_pos_input = QLineEdit()
        self.geo_skip_pos_input.setText(str(config.geo_skip_pos_m) if config else "0.10")
        self.geo_skip_pos_input.setMinimumHeight(35)
        form_layout.addRow(geo_skip_pos_label, self.geo_skip_pos_input)

        # ── M2M Min Observation Count Filter ────────────────────────────────
        add_section("🔗 M2M Min Observation Filter")

        minobs_label = QLabel("Min Pair Obs:")
        minobs_label.setStyleSheet(label_style)
        self.min_pair_obs_input = QLineEdit()
        self.min_pair_obs_input.setText(str(getattr(config, 'min_pair_obs', 2)) if config else "2")
        self.min_pair_obs_input.setMinimumHeight(35)
        self.min_pair_obs_input.setToolTip(
            "Marker pairs seen fewer times than this are removed, so a pair needs\n"
            "at least this many co-observations to survive.\n"
            "\n"
            "For a removed pair the less-observed marker's detections are dropped\n"
            "from the cameras that saw the pair.\n"
            "\n"
            "Set to 1 to disable.")
        form_layout.addRow(minobs_label, self.min_pair_obs_input)

        layout.addLayout(form_layout)
        layout.addSpacing(8)

        # Info label — same look as the yellow box in Marker Settings; the
        # trailing stretch keeps it from absorbing leftover vertical space
        # inside the scroll area (it used to balloon into a tall empty box).
        info_label = QLabel("💡 Observation covariances are computed with MARSCOT "
                            "(Unscented Transform through PnP) and passed to GTSAM as-is.")
        info_label.setStyleSheet("font-size: 12px; color: #666; padding: 8px; background-color: #fff3cd; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        layout.addStretch(1)

        button_layout = _dialog_button_row(self.accept_and_save, self.reject)
        make_dialog_scrollable(self, layout, button_layout, preferred_size=(600, 420))

    def accept_and_save(self):
        """Save and accept"""
        try:
            self.config.geo_skip_rot_deg   = float(self.geo_skip_rot_input.text())
            self.config.geo_skip_pos_m     = float(self.geo_skip_pos_input.text())
            self.config.min_pair_obs        = int(self.min_pair_obs_input.text())

            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {str(e)}")


if __name__ == "__main__":
    # Suppress Qt font warnings (set before program start)
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false;qt.gui.font=false'

    app = QApplication(sys.argv)

    # Set default font (prevent font size warnings)
    default_font = QFont("Segoe UI", 10)  # Windows default font
    app.setFont(default_font)

    # Qt message handler - Completely ignore QFont warnings only
    def qt_message_handler(mode, context, message):
        # Ignore QFont-related warnings (no issues with actual rendering)
        if "QFont::setPointSize" in message or "Point size" in message:
            return  # Don't output anything

        # Output other warnings/errors normally
        if mode == 0:  # QtDebugMsg
            print(f"Debug: {message}")
        elif mode == 1:  # QtWarningMsg
            print(f"Warning: {message}")
        elif mode == 2:  # QtCriticalMsg
            print(f"Critical: {message}")
        elif mode == 3:  # QtFatalMsg
            print(f"Fatal: {message}")

    from PySide6.QtCore import qInstallMessageHandler
    qInstallMessageHandler(qt_message_handler)

    window = PathSelector()
    window.show()
    exit_code = app.exec()

    # Leave without unwinding: see PathSelector.closeEvent for why interpreter
    # teardown crashes once a calibration has run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
