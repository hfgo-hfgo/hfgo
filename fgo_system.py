
import glob
import math
import os
import pickle
import threading
from collections import defaultdict

import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure as MplFigure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial.transform import Rotation as Ro

from pose_covariance_calculator import calculate_pose_covariance_april

# Arial for every plot this module builds, so the figures look the same whether
# the GUI imported it or a script calls it directly.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
# Journals reject Type 3 fonts (matplotlib's PDF/EPS default): 42 embeds the
# TrueType face itself, and svg.fonttype='none' keeps SVG text as text.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

try:
    import pupil_apriltags
except Exception as _pupil_import_err:  # ImportError or DLL load failure
    pupil_apriltags = None
    _pupil_import_reason = str(_pupil_import_err)
else:
    _pupil_import_reason = None


g_current_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# constants (algorithm tuning - not configurable via GUI)
# ============================================================================
# Image preprocessing
BLUR_SIG = 1
DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi

# Smallest eigenvalue a covariance may have before it is nudged back to positive
# definite. Measured minimum on real detections is ~1e-8, so 1e-12 only ever fires
# on a genuinely degenerate block rather than on ordinary well-conditioned ones.
_COV_MIN_EIGENVALUE = 1e-12


def create_noise_model_from_uncertainty(uncertainty_matrix):
    """
    Build a GTSAM noise model from a MARSCOT 6x6 pose covariance.

    The covariance is handed to GTSAM untouched (off-diagonal correlation terms
    included) after reordering it from the calculator's [pos, rot] layout to the
    [rot, pos] layout GTSAM's Pose3 tangent space uses.

    Args:
        uncertainty_matrix: 6x6 pose covariance matrix, [pos(0-2), rot(3-5)].

    Returns:
        gtsam.noiseModel.Gaussian: full covariance noise model.
    """
    idx = [3, 4, 5, 0, 1, 2]
    cov = np.array(uncertainty_matrix, dtype=float)[idx][:, idx]

    # Gaussian.Covariance needs a symmetric positive-definite matrix, so
    # symmetrise (the calculator leaves float asymmetry behind) and nudge a
    # rank-deficient diagonal — otherwise GTSAM raises and the caller's broad
    # except drops the image.
    cov_sym = (cov + cov.T) / 2.0
    min_eig = float(np.linalg.eigvalsh(cov_sym).min())
    if min_eig <= _COV_MIN_EIGENVALUE:
        jitter = _COV_MIN_EIGENVALUE - min_eig
        cov_sym = cov_sym + np.eye(6) * jitter
        print(f"  [covariance] non-PD block (min eig {min_eig:.3e}); "
              f"added {jitter:.3e} to the diagonal to keep it invertible")
    return gtsam.noiseModel.Gaussian.Covariance(cov_sym)


# ============================================================================
# Pose-scene rendering: markers as oriented quads, cameras as view frustums
#
# Every camera contributes 8 frustum edges and one image-plane quad. Drawing
# those with one plot3D call each, the way _draw_camera_fov_pyramids does, costs
# a dozen artists per camera, and Matplotlib reprojects and depth-sorts every
# artist on each mouse move - a few hundred camera poses is enough to make the
# view unusable. Batching the whole scene into one Line3DCollection and one
# Poly3DCollection keeps the artist count flat no matter how many poses there
# are, which is what makes these views interactive.
# ============================================================================

# Marker quads are red for environment boards and blue for wheel boards, matching
# the existing stage plots. Cameras get their own hues so a frustum can never be
# mistaken for a board: teal for environment, orange for wheel, grey when a
# camera group is only there for context.
SCENE_COLORS = {
    'env_marker': '#d62728',
    'wheel_marker': '#1f77b4',
    'env_camera': '#17becf',
    'wheel_camera': '#ff7f0e',
    'context_camera': '#90a4ae',
    'vehicle_camera': '#8e44ad',
}


def fov_from_intrinsics(camera_params, image_size=None):
    """Horizontal and vertical field of view in degrees from [fx, fy, cx, cy].

    Without an explicit image size the sensor is assumed centred, so the principal
    point sits at half the width and height. That is only an approximation for a
    decentred calibration, but the frustum is a schematic of where the camera
    looks rather than a measurement, so a degree or two of error is invisible.
    """
    fx, fy, cx, cy = (float(v) for v in camera_params[:4])
    width, height = image_size if image_size else (cx * 2.0, cy * 2.0)
    if fx <= 0 or fy <= 0 or width <= 0 or height <= 0:
        return 60.0, 45.0
    return (float(np.degrees(2 * np.arctan(width / (2 * fx)))),
            float(np.degrees(2 * np.arctan(height / (2 * fy)))))


def camera_frustum_vertices(position, rotation, fov_h_deg, fov_v_deg, depth):
    """Frustum corners in world coordinates: apex first, then the image plane.

    The camera looks down +Z with +X right and +Y down, the OpenCV convention the
    rest of the pipeline solves PnP in, so the returned rectangle is what the
    camera actually frames at `depth` metres in front of it.
    """
    half_w = depth * np.tan(np.radians(fov_h_deg) / 2.0)
    half_h = depth * np.tan(np.radians(fov_v_deg) / 2.0)
    local = np.array([
        [0.0, 0.0, 0.0],
        [-half_w, -half_h, depth],
        [half_w, -half_h, depth],
        [half_w, half_h, depth],
        [-half_w, half_h, depth],
    ])
    return local @ np.asarray(rotation, dtype=float).T + np.asarray(position, dtype=float)


def frustum_edges_and_faces(camera_poses, fov_h_deg, fov_v_deg, depth):
    """Flatten many camera poses into segment and polygon lists ready for batching.

    Args:
        camera_poses: iterable of (position, 3x3 rotation) in world coordinates.

    Returns:
        (segments, faces) where segments is an (8N, 2, 3) array of frustum edges
        and faces is a list of N image-plane quads.
    """
    segments, faces = [], []
    for position, rotation in camera_poses:
        v = camera_frustum_vertices(position, rotation, fov_h_deg, fov_v_deg, depth)
        # Apex to each image-plane corner, then the image-plane rectangle. The
        # rectangle is not square unless the FOVs match, so it also shows roll.
        for i in range(1, 5):
            segments.append([v[0], v[i]])
        for i in range(1, 5):
            segments.append([v[i], v[i + 1 if i < 4 else 1]])
        faces.append(v[1:5])
    if not segments:
        return np.empty((0, 2, 3)), []
    return np.asarray(segments, dtype=float), faces


def add_camera_frustums(ax, camera_poses, fov_h_deg, fov_v_deg, depth,
                        color, alpha=0.9, face_alpha=0.18, linewidth=1.0, label=None):
    """Draw every camera as a view frustum using exactly two artists.

    Returns the number of cameras drawn.
    """
    segments, faces = frustum_edges_and_faces(camera_poses, fov_h_deg, fov_v_deg, depth)
    if not len(segments):
        return 0

    ax.add_collection3d(Line3DCollection(
        segments, colors=color, linewidths=linewidth, alpha=alpha, label=label))
    image_planes = Poly3DCollection(faces, facecolors=color, edgecolors='none',
                                    alpha=face_alpha)
    # Poly3DCollection ignores the alpha kwarg once facecolors are resolved on some
    # Matplotlib versions, so set it again on the resolved RGBA.
    image_planes.set_alpha(face_alpha)
    ax.add_collection3d(image_planes)
    return len(faces)


def observation_edges(graphs):
    """(camera_id, marker_id) pairs of every camera→marker factor in the graphs.

    Read from the optimized graphs rather than the detections, so observations
    dropped by min_pair_obs / geo_skip never show up as a link.
    """
    edges = set()
    for graph in graphs or []:
        if graph is None:
            continue
        for i in range(graph.size()):
            factor = graph.at(i)
            if not isinstance(factor, gtsam.BetweenFactorPose3):
                continue
            symbols = [gtsam.Symbol(k) for k in factor.keys()]
            chars = [chr(s.chr()) for s in symbols]
            if sorted(chars) != ['C', 'M']:
                continue
            camera = symbols[chars.index('C')].index()
            marker = symbols[chars.index('M')].index()
            edges.add((camera, marker))
    return edges


def add_observation_edges(ax, segments, color, alpha=0.2, linewidth=0.4):
    """Draw camera→marker observation links as one faint, thin collection."""
    if not len(segments):
        return 0
    lines = Line3DCollection(np.asarray(segments, dtype=float), colors=color,
                             linewidths=linewidth, alpha=alpha)
    # Behind the frustums and boards, which stay the thing the eye lands on.
    lines.set_zorder(0)
    ax.add_collection3d(lines)
    return len(segments)


def add_marker_quads(ax, quads, color, alpha=0.55, edge_alpha=0.95,
                     normals=None, label=None):
    """Draw board outlines as filled oriented quads using one or two artists.

    Args:
        quads: list of (4, 3) world-coordinate board outlines.
        normals: optional list of (2, 3) segments showing each board's +Z normal,
            which is what tells a board facing the wall from one facing the floor
            when the quad is seen edge-on.
    """
    if not quads:
        return 0

    faces = Poly3DCollection(quads, facecolors=color, edgecolors=color,
                             linewidths=1.2, label=label)
    faces.set_alpha(alpha)
    ax.add_collection3d(faces)

    if normals is not None and len(normals):
        ax.add_collection3d(Line3DCollection(
            np.asarray(normals, dtype=float), colors=color,
            linewidths=1.4, alpha=edge_alpha))
    return len(quads)


def scene_frustum_depth(points, fraction=0.04, minimum=0.02):
    """Pick a frustum size that stays legible at any scene scale.

    A fixed depth in metres is either invisible in a warehouse or swamps a
    desktop rig, so scale it to the diagonal of everything being drawn.
    """
    points = np.asarray([p for p in points if p is not None], dtype=float).reshape(-1, 3)
    if points.size == 0:
        return 0.3
    span = points.max(axis=0) - points.min(axis=0)
    diagonal = float(np.linalg.norm(span))
    return max(minimum, fraction * diagonal) if diagonal > 0 else 0.3


def level_display_frame(points, x_toward=None):
    """Rotation and origin that lay the best-fit plane of `points` horizontal.

    Display-only. The optimisation frame is anchored to marker M10's board, so
    the raw map appears tilted by that board's mounting angle and is hard to
    read in a 3-D plot. This fits a plane through the given points (marker
    centres) and returns (R, c) with

        p_display = R @ (p - c)

    +Z is the plane normal, +X points toward `x_toward` projected into the
    plane (falls back to the first principal direction). Estimates are NOT
    changed - only what the figures draw.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return np.eye(3), np.zeros(3)
    c = pts.mean(axis=0)
    if len(pts) < 3:
        return np.eye(3), c
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    n = Vt[-1]
    if n[int(np.argmax(np.abs(n)))] < 0:
        n = -n

    x = (np.asarray(x_toward, dtype=float) - c) if x_toward is not None else Vt[0]
    x = x - (x @ n) * n
    if np.linalg.norm(x) < 1e-9:
        x = Vt[0] - (Vt[0] @ n) * n
    if np.linalg.norm(x) < 1e-9:
        return np.eye(3), c
    x = x / np.linalg.norm(x)
    y = np.cross(n, x)
    return np.vstack([x, y, n]), c


def median_pose3(poses):
    """
    Compute the median Pose3 from a list of candidates on the SE(3) manifold.

    Translation: element-wise median of xyz (no wrap issue).
    Rotation: SO(3) Lie algebra median to avoid RPY ±180° wrap artifacts.
      - All rotations are mapped into the tangent space of the first pose via
        Rot3.Logmap(R_ref^{-1} * R_i), giving a 3-vector per pose.
      - The component-wise median is taken in that tangent space.
      - The result is mapped back with Rot3.Expmap and composed with R_ref.

    This is robust to outliers (median) and free of Euler-angle singularities.

    Args:
        poses: list of gtsam.Pose3

    Returns:
        gtsam.Pose3 – median pose
    """
    if len(poses) == 1:
        return poses[0]

    # --- translation: plain median (no wrap issue) ---
    t_arr = np.array([p.translation() for p in poses])   # (N, 3)
    t_med = np.median(t_arr, axis=0)

    # --- rotation: Lie algebra median on SO(3) ---
    R_ref = poses[0].rotation()
    logs  = np.array([
        gtsam.Rot3.Logmap(R_ref.inverse().compose(p.rotation()))
        for p in poses
    ])                                                     # (N, 3)
    log_med = np.median(logs, axis=0)                     # (3,)
    R_med   = R_ref.compose(gtsam.Rot3.Expmap(log_med))

    return gtsam.Pose3(R_med, gtsam.Point3(t_med))


def filter_low_count_pairs(marker_pair_poses, marker_pair_cam_ids,
                           marker_observations, camera_to_observations,
                           camera_ids, marker_ids,
                           min_obs=5):
    """
    Remove marker pairs that have fewer than min_obs observations.

    For each removed pair (Mi, Mj):
      - The less-observed marker's C→M detection is removed from every camera
        that co-observed the pair.  (The camera and other markers are kept.)
      - Cascade: markers that become fully isolated (no remaining pairs) have
        all their C→M observations removed entirely.

    All dicts/sets are modified in place.
    """
    if min_obs <= 1:
        return          # removing pairs with <1 observation is a no-op

    # Total observation count per marker across all cameras
    total_obs = defaultdict(int)
    for obs_list in camera_to_observations.values():
        for mid, *_ in obs_list:
            total_obs[mid] += 1

    removed_pairs = set()
    for pair in list(marker_pair_poses.keys()):
        if len(marker_pair_poses[pair]) >= min_obs:
            continue

        mi, mj = pair
        # Remove the less-observed marker. On a tie the LOWER id loses (pairs
        # are canonical (lo, hi), so `<=` picks mi) — an arbitrary but
        # long-standing choice kept for reproducibility. Counts are a
        # pre-filter snapshot, not updated between removals.
        marker_to_remove = mi if total_obs[mi] <= total_obs[mj] else mj

        pair_cam_set = set(marker_pair_cam_ids.get(pair, []))

        # Remove C→marker_to_remove from each pair camera
        for cam_id in pair_cam_set:
            if cam_id in camera_to_observations:
                camera_to_observations[cam_id] = [
                    (m, T, nm) for m, T, nm in camera_to_observations[cam_id]
                    if m != marker_to_remove
                ]

        # Reflect removal in marker_observations
        if marker_to_remove in marker_observations:
            marker_observations[marker_to_remove] = [
                (cid, T, nm) for cid, T, nm in marker_observations[marker_to_remove]
                if cid not in pair_cam_set
            ]

        del marker_pair_poses[pair]
        marker_pair_cam_ids.pop(pair, None)
        removed_pairs.add(pair)

    if removed_pairs:
        print(f"  [min_pair_obs] removed {len(removed_pairs)} pairs "
              f"(<{min_obs} obs): {sorted(removed_pairs)}")

    # Cascade: remove markers that are now fully isolated (no remaining pairs)
    changed = True
    removed_markers = set()
    while changed:
        changed = False
        all_in_pairs = set()
        for mi, mj in marker_pair_poses:
            all_in_pairs.add(mi)
            all_in_pairs.add(mj)

        for mid in list(marker_ids):
            if mid in all_in_pairs:
                continue
            removed_markers.add(mid)
            marker_ids.discard(mid)
            marker_observations.pop(mid, None)

            for cam_id in list(camera_to_observations.keys()):
                camera_to_observations[cam_id] = [
                    (m, T, nm) for m, T, nm in camera_to_observations[cam_id]
                    if m != mid
                ]
                if not camera_to_observations[cam_id]:
                    del camera_to_observations[cam_id]
                    camera_ids.discard(cam_id)

            for pair in list(marker_pair_poses.keys()):
                if mid in pair:
                    del marker_pair_poses[pair]
                    marker_pair_cam_ids.pop(pair, None)
            changed = True

    if removed_markers:
        print(f"  [min_pair_obs] cascade-removed {len(removed_markers)} "
              f"isolated markers: {sorted(removed_markers)}")


def remove_disconnected_nodes(graph, initial_estimate, new_camera_ids=None):
    """
    Remove variables from initial_estimate that have no factor in the graph.

    When the outlier filter removes observations AFTER marker_pair_median is
    built, Phase-2 BFS can initialize markers whose every observation was
    subsequently filtered.  Those markers end up in initial_estimate with zero
    factors, making GTSAM's ordering inconsistent (RuntimeError).

    Args:
        graph               : gtsam.NonlinearFactorGraph
        initial_estimate    : gtsam.Values  (modified in place via rebuild)
        new_camera_ids      : optional set of camera IDs to sync (modified in place)

    Returns:
        cleaned gtsam.Values (a new object if anything was removed, else the
        original object unchanged)
    """
    graph_keys: set = set()
    for i in range(graph.size()):
        factor = graph.at(i)
        if factor is not None:
            for k in factor.keys():
                graph_keys.add(k)

    disconnected = [k for k in initial_estimate.keys() if k not in graph_keys]
    if not disconnected:
        return initial_estimate

    disconnected_names = []
    removed_cam_ids: set = set()
    for k in disconnected:
        sym = gtsam.Symbol(k)
        ch  = chr(sym.chr())
        disconnected_names.append(f"{ch}{sym.index()}")
        if ch == 'C':
            removed_cam_ids.add(sym.index())

    clean_estimate = gtsam.Values()
    for k in initial_estimate.keys():
        if k in graph_keys:
            clean_estimate.insert(k, initial_estimate.atPose3(k))

    if new_camera_ids is not None and removed_cam_ids:
        new_camera_ids -= removed_cam_ids

    preview = disconnected_names[:10]
    suffix  = f" ... (+{len(disconnected_names) - 10} more)" if len(disconnected_names) > 10 else ""
    print(f"  ⚠ [graph-cleanup] {len(disconnected)} disconnected variable(s) removed"
          f" before optimization: {preview}{suffix}")
    return clean_estimate


def carry_over_stage_poses(new_result, base_result, chars=('C', 'V'), label=''):
    """Copy an earlier stage's camera poses into the newer stage's result.

    Every stage rebuilds its graph from the previous result's *markers* only and
    re-anchors them with Constrained priors, so the cameras solved in the earlier
    stage never enter the new graph and vanish from its Values. They are still
    valid in the same frame - the markers they were solved against did not move -
    and the callers treat `result` as the one place every stage's poses live
    (C < 1000 environment, 10000s wheel, 20000s vehicle), so put them back.

    Modifies `new_result` in place and returns it.
    """
    if new_result is None or base_result is None:
        return new_result

    wanted = {ord(c) for c in chars}
    restored = []
    for key in base_result.keys():
        symbol = gtsam.Symbol(key)
        if symbol.chr() not in wanted or new_result.exists(key):
            continue
        try:
            new_result.insert(key, base_result.atPose3(key))
        except Exception:
            # Non-Pose3 values (nothing stores any today) are simply skipped.
            continue
        restored.append(f"{chr(symbol.chr())}{symbol.index()}")

    if restored:
        preview = ', '.join(restored[:8])
        suffix = f" ... (+{len(restored) - 8} more)" if len(restored) > 8 else ""
        print(f"  {label}carried over {len(restored)} pose(s) from the previous stage: "
              f"{preview}{suffix}")
    return new_result


def get_marker_size(marker_id, marker_size_default, marker_size2_default,
                   vehicle_markers, marker_size_mapping=None):
    """
    Return marker half-size for a given marker ID.

    Args:
        marker_id: marker ID to look up
        marker_size_default: default half-size for environment markers
        marker_size2_default: half-size for vehicle/wheel markers
        vehicle_markers: list of vehicle corner marker IDs
        marker_size_mapping: optional {marker_id: size} override dict
    Returns:
        float: marker half-size in meters
    """
    if marker_size_mapping and marker_id in marker_size_mapping:
        return marker_size_mapping[marker_id]
    if marker_id in vehicle_markers:
        return marker_size2_default
    return marker_size_default


# ============================================================================
# AprilTag / AprilGrid configuration
# ============================================================================

# Cell layout of every tag family shipped with AprilTag 3, taken from
# libapriltag's family definitions. (36h10 exists in OpenCV's dictionary list
# but was dropped from AprilTag 3, so it is absent here.)
#
# width_at_border is the square the detector reports as the tag corners, and
# total_width is the full printed pattern including the outer ring of cells.
# The two differ per family, so the physical length a marker size refers to
# depends on which of them was measured. Families with a reversed border draw
# that square white-on-black, which puts the measurable square at the centre of
# the pattern rather than around it.
APRILTAG_FAMILY_GEOMETRY = {
    'tagStandard41h12': {'width_at_border': 5, 'total_width': 9,  'reversed_border': True},
    'tagStandard52h13': {'width_at_border': 6, 'total_width': 10, 'reversed_border': True},
    'tag36h11':         {'width_at_border': 8, 'total_width': 10, 'reversed_border': False},
    'tag25h9':          {'width_at_border': 7, 'total_width': 9,  'reversed_border': False},
    'tag16h5':          {'width_at_border': 6, 'total_width': 8,  'reversed_border': False},
    'tagCircle21h7':    {'width_at_border': 5, 'total_width': 9,  'reversed_border': True},
    'tagCircle49h12':   {'width_at_border': 5, 'total_width': 11, 'reversed_border': True},
    'tagCustom48h12':   {'width_at_border': 6, 'total_width': 10, 'reversed_border': True},
}

APRILTAG_FAMILIES = tuple(APRILTAG_FAMILY_GEOMETRY)

# AprilTag 3 lists corners counter-clockwise in the tag's own frame, while the
# object-point template follows OpenCV's clockwise order starting at the corner
# drawn top-left. Indexing the reported corners with this permutation converts
# between the two, for tags printed from the official AprilTag 3 images.
#
# Note this is not the permutation that matches cv2.aruco: OpenCV's
# DICT_APRILTAG_* images are a 180 degree rotation of the official images of the
# same ID, so the corner that sits top-left differs. tagStandard41h12 only exists
# in the official images, and using OpenCV's order there would rotate every
# marker frame by 180 degrees and break multi-tag boards, whose corner order
# would then contradict the grid slot layout.
APRILTAG3_TO_OPENCV_CORNERS = [3, 2, 1, 0]

DEFAULT_APRILTAG_CONFIG = {
    'family': 'tagStandard41h12',
    # Detector tuning. quad_decimate downsamples only the quad search; corners
    # are always refined at full resolution, so raising it costs little accuracy
    # while making large tags on slightly bowed boards detectable (their edges
    # deviate from a straight line by more pixels at full resolution).
    'quad_decimate': 2.0,
    'quad_sigma': 0.0,
    'refine_edges': 1,
    'decode_sharpening': 0.25,
    'nthreads': 0,  # 0 = one thread per CPU core
    # Board layout -----------------------------------------------------------
    # Env: board_id = min(tag_ids_on_board)
    #          = first_tag_id + ((tag_id - first_tag_id) // (rows*cols)) * (rows*cols)
    # e.g. env_first_tag_id=10, 2×2 grid:
    #   tags [10,11,12,13] → board_id=10   tags [14,15,16,17] → board_id=14
    #
    # Wheel: board_id is looked up directly from `vehicle_markers` (the 4
    # known wheel board_ids), so no separate "wheel_first_tag_id" anchor
    # exists — see board_layout_for_tag().
    # Grid defaults MUST match CalibrationConfig's — these are the getattr
    # fallbacks for a config file that predates the grid attributes, and a
    # mismatch would silently regroup boards (e.g. 1×1 with anchor 0).
    'env_grid_rows': 2,
    'env_grid_cols': 2,
    'env_tag_gap': 0.246914,
    'env_first_tag_id': 10,   # tag_id of the very first slot on the first env board
    'wheel_grid_rows': 2,
    'wheel_grid_cols': 2,
    'wheel_tag_gap': 0.154321,
}


def build_apriltag_config(cfg):
    """Collect the AprilTag/AprilGrid detector settings from a CalibrationConfig."""
    return {
        key: getattr(cfg, 'apriltag_family' if key == 'family' else key, default)
        for key, default in DEFAULT_APRILTAG_CONFIG.items()
    }



def normalize_apriltag_family(family):
    """Normalize a family name to its AprilTag 3 spelling ('36h11' -> 'tag36h11')."""
    requested = str(family).strip()
    key = requested.lower().replace('dict_apriltag_', '')
    if not key.startswith('tag'):
        key = 'tag' + key

    for known in APRILTAG_FAMILIES:
        if known.lower() == key:
            return known

    if key == 'tag36h10':
        raise ValueError(
            "AprilTag family '36h10' was dropped in AprilTag 3 and cannot be detected. "
            f"Re-print the boards using one of: {', '.join(APRILTAG_FAMILIES)}")
    raise ValueError(f"Unknown AprilTag family '{requested}'. "
                     f"Supported families: {', '.join(APRILTAG_FAMILIES)}")


def resolve_apriltag_config(apriltag_config):
    """Fill in defaults and validate an AprilTag/AprilGrid config dict."""
    resolved = dict(DEFAULT_APRILTAG_CONFIG)
    if apriltag_config:
        resolved.update({k: v for k, v in apriltag_config.items() if v is not None})

    resolved['family'] = normalize_apriltag_family(resolved['family'])

    for key in ('env_grid_rows', 'env_grid_cols', 'wheel_grid_rows', 'wheel_grid_cols'):
        value = int(resolved[key])
        if value < 1:
            raise ValueError(f"AprilGrid '{key}' must be 1 or greater (got {value}).")
        resolved[key] = value

    for key in ('env_tag_gap', 'wheel_tag_gap'):
        value = float(resolved[key])
        if value < 0.0:
            raise ValueError(f"AprilGrid '{key}' must be 0 or greater (got {value}).")
        resolved[key] = value

    value = int(resolved['env_first_tag_id'])
    if value < 0:
        raise ValueError(f"AprilGrid 'env_first_tag_id' must be 0 or greater (got {value}).")
    resolved['env_first_tag_id'] = value

    quad_decimate = float(resolved['quad_decimate'])
    if quad_decimate < 1.0:
        raise ValueError(f"'quad_decimate' must be 1.0 or greater (got {quad_decimate}).")
    resolved['quad_decimate'] = quad_decimate

    for key in ('quad_sigma', 'decode_sharpening'):
        resolved[key] = float(resolved[key])
    resolved['refine_edges'] = int(bool(resolved['refine_edges']))

    nthreads = int(resolved['nthreads'])
    if nthreads < 0:
        raise ValueError(f"'nthreads' must be 0 (auto) or greater (got {nthreads}).")
    resolved['nthreads'] = nthreads

    return resolved


# Detectors already built, keyed by the settings they were built from.
#
# Constructing one is not free of lasting consequence on Windows: for every
# successful construction pupil_apriltags leaks an os.add_dll_directory entry,
# because it closes that cookie only on the path where the DLL failed to load
# and breaks out of the loop before the close on the path where it succeeded.
# The entries accumulate for the life of the process and eventually LoadLibrary
# gives up with WinError 206 ("filename or extension too long") once detectors
# have been built repeatedly over a long session.
#
# Keyed per thread as well as per settings: the native detector carries mutable
# state through a detect() call and is not safe to drive from two threads at
# once, and the GUI thread (noise preview) uses one alongside the calibration
# worker thread.
#
# The cache also keeps every detector alive for the life of the process, which
# it must: pupil_apriltags.Detector.__del__ destroys the tag family and then the
# detector, but the detector already owns the family it was given, so the family
# is freed twice down in libapriltag and the double free segfaults the whole
# process.
_detector_cache = threading.local()


def create_apriltag_detector(apriltag_config):
    """Return this thread's AprilTag 3 detector for a resolved config dict.

    Detectors are reused, so repeated calls with equal settings hand back the
    same object rather than constructing another one.
    """
    if pupil_apriltags is None:
        reason = f"\n(import failed: {_pupil_import_reason})" if _pupil_import_reason else ""
        raise ImportError(
            "AprilTag detection requires the AprilTag 3 bindings, which are not installed.\n"
            f"Install them with:  pip install pupil-apriltags{reason}")

    settings = {
        'families': apriltag_config['family'],
        'nthreads': apriltag_config['nthreads'] or (os.cpu_count() or 1),
        'quad_decimate': apriltag_config['quad_decimate'],
        'quad_sigma': apriltag_config['quad_sigma'],
        'refine_edges': apriltag_config['refine_edges'],
        'decode_sharpening': apriltag_config['decode_sharpening'],
    }

    cache = getattr(_detector_cache, 'by_settings', None)
    if cache is None:
        cache = _detector_cache.by_settings = {}

    key = tuple(sorted(settings.items()))
    detector = cache.get(key)
    if detector is None:
        detector = cache[key] = pupil_apriltags.Detector(**settings)
    return detector


def build_tag_object_points(local_index, half_tag_size, rows, cols, tag_gap):
    """
    Return the four board-frame 3D corners of one tag slot on an AprilGrid board.

    Tag IDs fill the board row-major starting at the top-left slot. The board frame
    is centred on the board and keeps the single-tag corner convention used
    throughout the system: +X points image-left, +Y points image-up, Z = 0 on the
    board plane, and corners are ordered BR, BL, TL, TR.

    Args:
        local_index: slot index within the board (0 .. rows*cols-1)
        half_tag_size: half the side of the square the detector reports, in meters
            (see APRILTAG_FAMILY_GEOMETRY for what that square is per family)
        rows, cols: board grid dimensions
        tag_gap: gap between the detected squares of neighbouring tags, in meters

    Returns:
        np.array shape (4, 3) - board-frame corner coordinates
    """
    pitch = 2.0 * half_tag_size + tag_gap
    row, col = divmod(int(local_index), int(cols))

    center_x = -(col - (cols - 1) / 2.0) * pitch
    center_y = -(row - (rows - 1) / 2.0) * pitch

    s = half_tag_size
    return np.array([
        [center_x + s, center_y + s, 0.0],   # Bottom right
        [center_x - s, center_y + s, 0.0],   # Bottom left
        [center_x - s, center_y - s, 0.0],   # Top left
        [center_x + s, center_y - s, 0.0],   # Top right
    ])


def resolve_board_layout(tag_id, apriltag_config, vehicle_markers):
    """Resolve which physical board an AprilTag ID belongs to.

    Module-level so callers that only hold a config (e.g. the GUI mapping a
    typed tag ID onto its board) do not have to build a detection system.
    UncertaintyFactorGraph.board_layout_for_tag() delegates here, so both
    paths always apply the same grouping rule.

    Returns: (board_id, local_index, rows, cols, tag_gap) with
             board_id = min(tag_ids_on_board).
    """
    cfg = apriltag_config
    vehicle_markers = vehicle_markers or []

    wheel_rows = cfg['wheel_grid_rows']
    wheel_cols = cfg['wheel_grid_cols']
    wheel_n_m  = wheel_rows * wheel_cols

    # Wheel first: does tag_id fall inside a known wheel board's range?
    for wheel_board_id in vehicle_markers:
        local_index = tag_id - wheel_board_id
        if 0 <= local_index < wheel_n_m:
            return wheel_board_id, local_index, wheel_rows, wheel_cols, cfg['wheel_tag_gap']

    env_rows  = cfg['env_grid_rows']
    env_cols  = cfg['env_grid_cols']
    env_n_m   = env_rows * env_cols
    env_first = cfg.get('env_first_tag_id', 10)

    rel_env = tag_id - env_first
    if rel_env >= 0:
        env_group    = rel_env // env_n_m
        env_board_id = env_first + env_group * env_n_m   # = min_tag_id of env board
        local_index  = rel_env % env_n_m
        return env_board_id, local_index, env_rows, env_cols, cfg['env_tag_gap']

    # Fallback: tag_id below env_first_tag_id
    board_id, local_index = divmod(tag_id, env_n_m)
    return board_id, local_index, env_rows, env_cols, cfg['env_tag_gap']


def build_board_object_points(half_tag_size, rows, cols, tag_gap):
    """Return the board-frame corners of every tag slot on a board, shape (4*rows*cols, 3)."""
    return np.vstack([
        build_tag_object_points(idx, half_tag_size, rows, cols, tag_gap)
        for idx in range(int(rows) * int(cols))
    ])


def create_uncertainty_system(camera_params, apriltag_config=None,
                            marker_size=0.4, marker_size2=0.2,
                            image_directory='tmp_img/', wheel_directory='tmp_img/wheel2/',
                            dist_coeffs=None, wheel_camera_params=None, wheel_dist_coeffs=None,
                            marker_size_mapping=None, first_marker_id=None,
                            vehicle_markers=None, image_extensions=None,
                            vehicle_camera_configs=None,
                            min_pair_obs=2,
                            excluded_apriltag_ids=None,
                            enable_visualization=True):
    """
    Factory function to create a UncertaintyFactorGraph with all config values
    explicitly provided (no global variable fallbacks).

    geo_skip_* is deliberately absent: it only acts on the Stage-3 vehicle
    branch, whose system is built by
    run_incremental_optimization_with_multiple_cameras.
    """
    return UncertaintyFactorGraph(
        camera_params=camera_params,
        apriltag_config=apriltag_config,
        marker_size=marker_size,
        marker_size2=marker_size2,
        image_directory=image_directory,
        wheel_directory=wheel_directory,
        dist_coeffs=dist_coeffs,
        wheel_camera_params=wheel_camera_params,
        wheel_dist_coeffs=wheel_dist_coeffs,
        marker_size_mapping=marker_size_mapping,
        first_marker_id=first_marker_id,
        vehicle_markers=vehicle_markers,
        image_extensions=image_extensions,
        vehicle_camera_configs=vehicle_camera_configs,
        min_pair_obs=min_pair_obs,
        excluded_apriltag_ids=excluded_apriltag_ids,
        enable_visualization=enable_visualization,
    )


class UncertaintyFactorGraph:
    def __init__(self, camera_params, apriltag_config=None,
                 marker_size=0.4, marker_size2=0.2,
                 image_directory='tmp_img/', wheel_directory='tmp_img/wheel2/',
                 dist_coeffs=None, wheel_camera_params=None, wheel_dist_coeffs=None,
                 marker_size_mapping=None, first_marker_id=None,
                 vehicle_markers=None, image_extensions=None,
                 vehicle_camera_configs=None,
                 geo_skip_rot_deg=4.0, geo_skip_pos_m=0.10,
                 min_pair_obs=2,
                 excluded_apriltag_ids=None,
                 enable_visualization=True):
        """
        Initialize Uncertainty-based Factor Graph optimization system.

        Args:
            camera_params: [fx, fy, cx, cy] environment camera intrinsic (required)
            apriltag_config: AprilTag/AprilGrid settings dict (see DEFAULT_APRILTAG_CONFIG)
            marker_size: environment marker half-size in meters (half of a single tag)
            marker_size2: vehicle/wheel marker half-size in meters (half of a single tag)
            image_directory: environment image directory path
            wheel_directory: wheel image directory path
            dist_coeffs: distortion coefficients for environment camera
            wheel_camera_params: [fx, fy, cx, cy] for wheel camera
            wheel_dist_coeffs: distortion coefficients for wheel camera
            marker_size_mapping: {marker_id: size} per-marker size override dict
            first_marker_id: reference marker ID used as optimization origin (None = auto)
            vehicle_markers: list of vehicle corner marker IDs [FL, RL, RR, FR]
            image_extensions: list of glob patterns for image files
            vehicle_camera_configs: list of vehicle camera config dicts
            geo_skip_rot_deg/geo_skip_pos_m: Stage-3 geometric consistency
                thresholds (vehicle observations further from the frozen marker
                map than this are excluded from the graph)
            min_pair_obs: marker pairs co-observed fewer times than this are
                removed (M2M minimum observation filter)
            enable_visualization: render the per-stage PNGs and pose tables
        """
        self.camera_params = camera_params
        self.apriltag_config = resolve_apriltag_config(apriltag_config)
        self._apriltag_detector = None  # built lazily, reused across images
        self.marker_size = marker_size
        self.marker_size2 = marker_size2
        self.image_directory = image_directory
        self.wheel_directory = wheel_directory
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.wheel_camera_params = wheel_camera_params if wheel_camera_params is not None else camera_params
        self.wheel_dist_coeffs = wheel_dist_coeffs if wheel_dist_coeffs is not None else np.zeros(5)
        self.marker_size_mapping = marker_size_mapping if marker_size_mapping is not None else {}
        self.first_marker_id = first_marker_id
        self.vehicle_markers = vehicle_markers if vehicle_markers is not None else []
        self.image_extensions = image_extensions if image_extensions is not None else ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        self.vehicle_camera_configs = vehicle_camera_configs if vehicle_camera_configs is not None else []

        self.geo_skip_rot_deg  = geo_skip_rot_deg
        self.geo_skip_pos_m    = geo_skip_pos_m
        self.min_pair_obs        = int(min_pair_obs)
        self.excluded_apriltag_ids = set(excluded_apriltag_ids) if excluded_apriltag_ids else set()

        self.enable_visualization = bool(enable_visualization)

        # Running total of observations dropped for having no computable covariance,
        # so a rare occurrence stays distinguishable from a systematic failure.
        self._cov_failed_detections = 0

        # Pose tables prepared during optimization for the GUI to display.
        # This module must not create windows itself: in GUI mode the stages run
        # inside a Qt worker thread, and window toolkits are main-thread-only.
        # Each entry: {'title', 'pretty_text', 'values_tsv'}.
        self.pending_pose_tables = []

        # Per-image detection errors are swallowed (one bad image must not
        # abort the run) but counted, so detection loops can distinguish
        # "no markers in image" from "image failed with an error" in their
        # summaries.
        self._detect_error_count = 0

        self.camera_matrix = np.array([
            [self.camera_params[0], 0, self.camera_params[2]],
            [0, self.camera_params[1], self.camera_params[3]],
            [0, 0, 1]
        ])

        # vehicle coordinate system cache (computed once, reused)
        self.cached_vehicle_frame = None


    def _draw_camera_fov_pyramids(self, ax, camera_id_to_pos, camera_id_to_rot, camera_ids,
                                 fov_h=None, fov_v=None, depth=0.3):
        """
        drawing camera fov pyramids for visualization

        Args:
            ax: matplotlib 3D axes
            camera_id_to_pos: camera id to position dictionary
            camera_id_to_rot: camera id to rotation dictionary
            camera_ids: camera id list
            fov_h: horizontal FOV (degrees, None means automatic calculation from intrinsic)
            fov_v: vertical FOV (degrees, None means automatic calculation from intrinsic)
            depth: pyramid depth (meters)
        """

        # if FOV is not specified, calculate from camera parameters
        if fov_h is None or fov_v is None:
            fx, fy, cx, cy = self.camera_params
            # assume image size is 1920x1200 (or 2 times cx, cy)
            img_width = cx * 2
            img_height = cy * 2

            fov_h_calculated = 2 * np.degrees(np.arctan(img_width / (2 * fx)))
            fov_v_calculated = 2 * np.degrees(np.arctan(img_height / (2 * fy)))

            if fov_h is None:
                fov_h = fov_h_calculated
            if fov_v is None:
                fov_v = fov_v_calculated

        for camera_id in camera_ids:
            if camera_id not in camera_id_to_pos or camera_id not in camera_id_to_rot:
                continue

            pos = np.array(camera_id_to_pos[camera_id])
            rot = camera_id_to_rot[camera_id]

            # set color for camera type
            if camera_id <1000:
                color = 'blue'  # original camera
                alpha = 0.6
            else:
                color = 'darkorange'  # new camera
                alpha = 0.6

            # convert GTSAM Rot3 to numpy rotation matrix
            R = rot.matrix()

            # define pyramid vertices in camera coordinate system
            # camera looks towards +Z direction
            fov_h_rad = np.radians(fov_h / 2)
            fov_v_rad = np.radians(fov_v / 2)

            # pyramid four corners (camera coordinate system)
            corners_cam = np.array([
                [0, 0, 0],  #  center of camera
                [-depth * np.tan(fov_h_rad), -depth * np.tan(fov_v_rad), depth],  # bottom-left
                [depth * np.tan(fov_h_rad), -depth * np.tan(fov_v_rad), depth],   # bottom-right
                [depth * np.tan(fov_h_rad), depth * np.tan(fov_v_rad), depth],    # top-right
                [-depth * np.tan(fov_h_rad), depth * np.tan(fov_v_rad), depth]    # top-left
            ])

            # convert to world coordinate system
            corners_world = []
            for corner in corners_cam:
                corner_world = R @ corner + pos
                corners_world.append(corner_world)
            corners_world = np.array(corners_world)


            axis_length = depth * 0.5  # half of pyramid depth

            # camera coordinate system axes
            x_axis_cam = np.array([axis_length, 0, 0])  # red (right)
            y_axis_cam = np.array([0, axis_length, 0])  # green (down)
            z_axis_cam = np.array([0, 0, axis_length])  # blue (front)

            # convert to world coordinate system
            x_axis_world = R @ x_axis_cam + pos
            y_axis_world = R @ y_axis_cam + pos
            z_axis_world = R @ z_axis_cam + pos

            # draw RGB axes
            ax.plot3D([pos[0], x_axis_world[0]], [pos[1], x_axis_world[1]], [pos[2], x_axis_world[2]],
                     'r-', linewidth=3, alpha=0.8)  # X axis: red
            ax.plot3D([pos[0], y_axis_world[0]], [pos[1], y_axis_world[1]], [pos[2], y_axis_world[2]],
                     'g-', linewidth=3, alpha=0.8)  # Y axis: green
            ax.plot3D([pos[0], z_axis_world[0]], [pos[1], z_axis_world[1]], [pos[2], z_axis_world[2]],
                     'b-', linewidth=3, alpha=0.8)  # Z axis: blue

            # draw pyramid edges
            # from center to four corners
            for i in range(1, 5):
                ax.plot3D([corners_world[0][0], corners_world[i][0]],
                         [corners_world[0][1], corners_world[i][1]],
                         [corners_world[0][2], corners_world[i][2]],
                         color=color, alpha=alpha, linewidth=1.5)

            # draw pyramid base rectangle
            for i in range(1, 5):
                next_i = i + 1 if i < 4 else 1
                ax.plot3D([corners_world[i][0], corners_world[next_i][0]],
                         [corners_world[i][1], corners_world[next_i][1]],
                         [corners_world[i][2], corners_world[next_i][2]],
                         color=color, alpha=alpha, linewidth=1.5)

            # fill pyramid faces with semi-transparent color
            faces = [
                [corners_world[0], corners_world[1], corners_world[2]],  # bottom face
                [corners_world[0], corners_world[2], corners_world[3]],  # right face
                [corners_world[0], corners_world[3], corners_world[4]],  # top face
                [corners_world[0], corners_world[4], corners_world[1]],  # left face
                [corners_world[1], corners_world[2], corners_world[3], corners_world[4]]  # bottom face
            ]

            poly3d = Poly3DCollection(faces, alpha=0*0.3, edgecolor=color)
            ax.add_collection3d(poly3d)

            # add dummy plot for legend (only for first camera)
            if camera_id == camera_ids[0]:  # only for first camera
                # External Camera legend
                external_cameras = [cid for cid in camera_ids if cid < 1000]
                if external_cameras:
                    ax.plot3D([], [], [], color=color, linewidth=2, alpha=0.6, label='External Camera FoV')
                    ax.plot3D([], [], [], color='green', linewidth=0.3, alpha=0.6, label='camera 2 marker pose')

                else:
                # Vehicle Camera legend
                    ax.plot3D([], [], [], color=color, linewidth=2, alpha=0.6, label='Vehicle camera')

            # add camera id label (direction-based label)
            label_offset = np.array([0, 0, 0.5])  # label offset

            # mapping camera id label (cam0, cam1, cam2, cam3...)
            if camera_id >= 20000:  # vehicle camera is 20000-series
                vehicle_camera_index = camera_id - 20000

                label_offset = np.array([0, 0, 0.3])  # top

                if vehicle_camera_index < len(self.vehicle_camera_configs):
                    label_text = f'(cam{vehicle_camera_index})'
                else:
                    label_text = ''
            elif camera_id >= 10000:  # Wheel camera is 10000-series
                label_text = ''
                label_offset = np.array([0, 0, 0.3])
            else:
                # show only external camera id
                label_text = ''
                label_offset = np.array([0, 0, 0.5])
            ax.text(pos[0] + label_offset[0],
                   pos[1] + label_offset[1],
                   pos[2] + label_offset[2],
                   label_text,
                   fontsize=16,
                   color=color,
                   fontweight='bold',
                   ha='center',  # horizontal alignment
                   va='bottom')  # vertical alignment

    def _draw_vehicle_coordinate_system(self, ax, marker_id_to_pos):
        """
        drawing vehicle coordinate system
        - M30, M32: rear (left, right)
        - M29, M31: front (left, right)
        - X axis: front, Y axis: left, Z axis: up (right-hand rule)
        """
        # check if all required markers are present
        required_markers = self.vehicle_markers

        if not all(mid in marker_id_to_pos for mid in required_markers):
            print("[viz] ⚠ vehicle coordinate system: required markers (M29, M30, M31, M32) are not present")
            return

        # extract marker positions
        pos_29 = np.array(marker_id_to_pos[self.vehicle_markers[0]])  # front left
        pos_30 = np.array(marker_id_to_pos[self.vehicle_markers[1]])  # rear left
        pos_31 = np.array(marker_id_to_pos[self.vehicle_markers[3]])  # front right
        pos_32 = np.array(marker_id_to_pos[self.vehicle_markers[2]])  # rear right

        # calculate vehicle center axis
        rear_center = (pos_30 + pos_32) / 2  # rear center
        front_center = (pos_29 + pos_31) / 2  # front center
        vehicle_center = rear_center  # set vehicle coordinate system origin to rear center (between M30 and M32)

        # calculate vehicle coordinate system axes
        # X axis: front direction (rear center → front center)
        x_axis = front_center - rear_center
        # Y axis: left direction (rear right → rear left)
        y_axis = pos_30 - pos_32
        # Z axis: up direction (right-hand rule: X × Y)
        z_axis = np.cross(x_axis, y_axis)
        norms = (np.linalg.norm(x_axis), np.linalg.norm(y_axis), np.linalg.norm(z_axis))
        if min(norms) < 1e-9:
            print("  ⚠ [viz] vehicle axes not drawn: wheel markers are "
                  "coincident or collinear")
            return
        x_axis = x_axis / norms[0]
        y_axis = y_axis / norms[1]
        z_axis = z_axis / norms[2]

        # axis length
        axis_length = 0.5  # 0.5 meters


        ax.plot3D([vehicle_center[0], vehicle_center[0] + axis_length*2 * x_axis[0]],
                 [vehicle_center[1], vehicle_center[1] + axis_length*2 * x_axis[1]],
                 [vehicle_center[2], vehicle_center[2] + axis_length*2 * x_axis[2]],
                 'r-', linewidth=4, alpha=0.8, label='Vehicle X (Front)')

        ax.plot3D([vehicle_center[0], vehicle_center[0] + axis_length * y_axis[0]],
                 [vehicle_center[1], vehicle_center[1] + axis_length * y_axis[1]],
                 [vehicle_center[2], vehicle_center[2] + axis_length * y_axis[2]],
                 'g-', linewidth=4, alpha=0.8, label='Vehicle Y (Left)')

        ax.plot3D([vehicle_center[0], vehicle_center[0] + axis_length * z_axis[0]],
                 [vehicle_center[1], vehicle_center[1] + axis_length * z_axis[1]],
                 [vehicle_center[2], vehicle_center[2] + axis_length * z_axis[2]],
                 'b-', linewidth=4, alpha=0.8, label='Vehicle Z (Up)')

        # show vehicle center point
        ax.scatter(vehicle_center[0], vehicle_center[1], vehicle_center[2],
                  c='black', s=100, marker='o', alpha=0.8, label='Vehicle Center')

        # show axis labels
        label_offset = axis_length * 1.1
        ax.text(vehicle_center[0] + label_offset * x_axis[0],
               vehicle_center[1] + label_offset * x_axis[1],
               vehicle_center[2] + label_offset * x_axis[2],
               'X', fontsize=12, color='red', fontweight='bold')

        ax.text(vehicle_center[0] + label_offset * y_axis[0],
               vehicle_center[1] + label_offset * y_axis[1],
               vehicle_center[2] + label_offset * y_axis[2],
               'Y', fontsize=12, color='green', fontweight='bold')

        ax.text(vehicle_center[0] + label_offset * z_axis[0],
               vehicle_center[1] + label_offset * z_axis[1],
               vehicle_center[2] + label_offset * z_axis[2],
               'Z', fontsize=12, color='blue', fontweight='bold')

        print(f"[viz] ✅ vehicle coordinate system displayed - center: [{vehicle_center[0]:.3f}, {vehicle_center[1]:.3f}, {vehicle_center[2]:.3f}]")

    def add_vehicle_coordinate_system_post_optimization(self, optimized_result):
        """
        Add the vehicle coordinate system to the results after optimization completes.

        Args:
            optimized_result: GTSAM optimization result (Values)

        The vehicle frame is stored in self.cached_vehicle_frame; nothing is returned.
        (Previously a result copy with a V node inserted was returned, but no caller
        ever used it.)
        """
        # extract optimized marker positions
        marker_id_to_pos = {}
        for key in optimized_result.keys():
            symbol = gtsam.Symbol(key)
            if chr(symbol.chr()) == 'M':  # marker node
                marker_id = symbol.index()
                pose = optimized_result.atPose3(key)
                marker_id_to_pos[marker_id] = pose.translation()

        # check that all required markers are present
        required_markers = self.vehicle_markers
        if not all(mid in marker_id_to_pos for mid in required_markers):
            print(f"[viz] ⚠ vehicle frame: required markers ({required_markers}) missing from the optimization result")
            return

        # extract marker positions
        pos_29 = np.array(marker_id_to_pos[self.vehicle_markers[0]])  # front axle, left
        pos_30 = np.array(marker_id_to_pos[self.vehicle_markers[1]])  # rear axle, left
        pos_31 = np.array(marker_id_to_pos[self.vehicle_markers[3]])  # front axle, right
        pos_32 = np.array(marker_id_to_pos[self.vehicle_markers[2]])  # rear axle, right

        # calculate vehicle center axis
        rear_center = (pos_30 + pos_32) / 2  # rear axle center
        front_center = (pos_29 + pos_31) / 2  # front axle center
        vehicle_center = rear_center  # vehicle frame origin sits at the rear axle center

        # Vehicle frame axis vectors (adjusted to match GT; GT carries yaw=90°,
        # so the axis definitions were re-derived accordingly).
        # Gram-Schmidt orthonormalisation:
        # step 1: define and normalise the X axis (forward direction - most stable)
        forward_vec = front_center - rear_center
        forward_norm = np.linalg.norm(forward_vec)
        if forward_norm < 1e-9:
            print("  ⚠ [stage3] vehicle frame skipped: front/rear wheel "
                  "centres coincide (degenerate marker layout)")
            return
        forward_vec = forward_vec / forward_norm

        # step 2: Y axis (Gram-Schmidt: remove the projection onto X)
        left_vec_raw = pos_30 - pos_32  # raw leftward vector
        left_vec = left_vec_raw - np.dot(left_vec_raw, forward_vec) * forward_vec  # strip X component
        left_norm = np.linalg.norm(left_vec)
        if left_norm < 1e-9:
            print("  ⚠ [stage3] vehicle frame skipped: wheel markers are "
                  "collinear (left axis undefined)")
            return
        left_vec = left_vec / left_norm

        # step 3: Z axis (cross product of X and Y, orthogonal by construction)
        up_vec = np.cross(forward_vec, left_vec)
        up_vec = up_vec / np.linalg.norm(up_vec)  # normalize (already unit length, kept for safety)

        # standard vehicle frame definition
        x_axis = forward_vec  # X axis: forward
        y_axis = left_vec     # Y axis: left
        z_axis = up_vec       # Z axis: up

        # rotation matrix (vehicle frame → world frame)
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # cache the vehicle frame for later reuse
        R_vehicle_to_world = rotation_matrix
        R_world_to_vehicle = R_vehicle_to_world.T
        self.cached_vehicle_frame = {
            'center': vehicle_center,
            'R_vehicle_to_world': R_vehicle_to_world,
            'R_world_to_vehicle': R_world_to_vehicle,
            'axes': {
                'x_axis': x_axis,
                'y_axis': y_axis,
                'z_axis': z_axis
            }
        }
        print(f"[viz] ✅ vehicle frame cached - center: [{vehicle_center[0]:.3f}, {vehicle_center[1]:.3f}, {vehicle_center[2]:.3f}]")

        return

    def board_layout_for_tag(self, tag_id):
        """
        Resolve which physical marker board an AprilTag ID belongs to and
        return (board_id, local_index, rows, cols, tag_gap).

        Wheel boards are looked up directly against `vehicle_markers`
        --------------------------------------------------------------
        `vehicle_markers` already lists the exact board_id (= min tag_id) of
        each of the 4 wheel boards, so a tag is a wheel tag whenever it falls
        inside one of those boards' tag ranges:
            board_id <= tag_id < board_id + (wheel_rows * wheel_cols)
        No extra "wheel_first_tag_id" anchor is needed (or possible to get
        out of sync with vehicle_markers).

        Env boards use arithmetic grouping from env_first_tag_id (Method B)
        ---------------------------------------------------------------------
        board_id  =  min(tag_ids_on_board)
                  =  env_first_tag_id + group_index * (rows * cols)

        where  group_index = (tag_id - env_first_tag_id) // (rows * cols)

        Example — env_first_tag_id=10, 2×2 grid (n*m=4):
            tags [10,11,12,13] → board_id = 10   (group_index 0)
            tags [14,15,16,17] → board_id = 14   (group_index 1)

        Returns:
            tuple: (board_id, local_index, rows, cols, tag_gap)
                   board_id = min_tag_id_on_board
        """
        return resolve_board_layout(tag_id, self.apriltag_config, self.vehicle_markers)

    def board_object_points(self, board_id):
        """Return the board-frame corners of every tag slot on the given board."""
        cfg = self.apriltag_config
        if board_id in self.vehicle_markers:
            rows, cols, tag_gap = cfg['wheel_grid_rows'], cfg['wheel_grid_cols'], cfg['wheel_tag_gap']
        else:
            rows, cols, tag_gap = cfg['env_grid_rows'], cfg['env_grid_cols'], cfg['env_tag_gap']

        half_tag_size = get_marker_size(
            board_id,
            self.marker_size,
            self.marker_size2,
            self.vehicle_markers,
            self.marker_size_mapping
        )
        return build_board_object_points(half_tag_size, rows, cols, tag_gap)

    def get_apriltag_detector(self):
        """Return this system's AprilTag 3 detector, building it on first use."""
        if self._apriltag_detector is None:
            self._apriltag_detector = create_apriltag_detector(self.apriltag_config)
        return self._apriltag_detector

    def detect_april(self, image):
        """
        Detect AprilTags and merge the tags of each physical board into one detection.

        Board ID definition (Method B)
        --------------------------------
        board_id  =  min(tag_ids_on_board)
                  =  first_tag_id + group_index * (rows * cols)

        board_layout_for_tag() returns this board_id directly, so no
        post-hoc division is needed.  Each detected board is identified by
        the smallest AprilTag ID printed on it.

        Example — env_first_tag_id=10, 2×2 grid:
            tags [10,11,12,13] → board_id = 10
            tags [14,15,16,17] → board_id = 14

        Downstream, a single solvePnP over all detected corners of the board
        yields the board pose, which remains valid even when only a subset of
        the board's tags is visible (occlusion tolerance).
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        results = self.get_apriltag_detector().detect(np.ascontiguousarray(gray))

        # board_id → list of (local_index, corners, obj_pts, tag_id)
        # board_id = min_tag_id_on_board, returned directly by board_layout_for_tag
        raw_groups = {}
        for result in results:
            # Reject error-corrected decodes: a misread ID would attach the
            # observation to the wrong board and corrupt the pose graph.
            if result.hamming > 0:
                continue

            tag_id = int(result.tag_id)
            corner_set = np.asarray(result.corners, dtype=np.float64)[APRILTAG3_TO_OPENCV_CORNERS]

            board_id, local_index, rows, cols, tag_gap = self.board_layout_for_tag(tag_id)

            half_tag_size = get_marker_size(
                board_id,
                self.marker_size,
                self.marker_size2,
                self.vehicle_markers,
                self.marker_size_mapping
            )

            raw_groups.setdefault(board_id, []).append((
                local_index,
                corner_set,
                build_tag_object_points(local_index, half_tag_size, rows, cols, tag_gap),
                tag_id,
            ))

        detections = []
        for board_id, tags in sorted(raw_groups.items()):
            # Sort by slot index so the merged corner array has a stable layout
            # regardless of the order the detector reported the tags.
            tags.sort(key=lambda t: t[0])
            detections.append({
                'marker_id': board_id,          # = min_tag_id_on_board
                'corners': np.vstack([t[1] for t in tags]),
                'object_points': np.vstack([t[2] for t in tags]),
                'tag_ids': sorted(t[3] for t in tags),
                'tag_slots': [t[0] for t in tags],
            })
        return detections

    def get_image_files(self):
        """Return list of environment image files from the configured image_directory."""
        if not self.image_directory:
            print("[io] ❌ image_directory is not set. Please select the environment image folder in the GUI.")
            return []
        if os.path.isabs(self.image_directory):
            directory = self.image_directory
        else:
            directory = os.path.normpath(os.path.join(g_current_dir, self.image_directory))
        image_files = []
        for extension in self.image_extensions:
            image_files.extend(glob.glob(os.path.join(directory, extension.lower())))
            image_files.extend(glob.glob(os.path.join(directory, extension.upper())))
        return sorted(set(image_files))

    def get_wheel_image_files(self):
        """Return list of wheel image files from the configured wheel_directory."""
        if not self.wheel_directory:
            print("[io] ❌ wheel_directory is not set. Please select the wheel image folder in the GUI.")
            return []
        if os.path.isabs(self.wheel_directory):
            directory = self.wheel_directory
        else:
            directory = os.path.normpath(os.path.join(g_current_dir, self.wheel_directory))
        image_files = []
        for extension in self.image_extensions:
            found = glob.glob(os.path.join(directory, extension.lower())) + \
                    glob.glob(os.path.join(directory, extension.upper()))
            image_files.extend(found)
        image_files = sorted(set(image_files))
        print(f"[io] wheel images found: {len(image_files)} in '{directory}'")
        return image_files

    def process_wheel_images_with_base_result(self, base_result):
        """Process wheel images against the base result and run the additional FGO."""
        print("\n[stage2] processing wheel images...")

        _dbg = os.path.join(g_current_dir, "debug_wheel.log")
        def _log(msg):
            with open(_dbg, 'a', encoding='utf-8') as _f:
                _f.write(msg + '\n')
        _log("="*10)
        _log("\n process_wheel_images_with_base_result called")
        _log(f"  wheel_directory      : {self.wheel_directory}")
        _log(f"  apriltag_config      : {self.apriltag_config}")
        _log(f"  marker_size          : {self.marker_size}  marker_size2: {self.marker_size2}")
        _log(f"  vehicle_markers      : {self.vehicle_markers}")
        _log(f"  wheel_camera_params  : {self.wheel_camera_params}")
        _log(f"  wheel_dist_coeffs    : {self.wheel_dist_coeffs}")

        print("[stage2] wheel observation covariance: MARSCOT (same as environment)")

        # 1. detect markers in wheel images
        wheel_image_files = self.get_wheel_image_files()
        print(f"[stage2] wheel images found: {len(wheel_image_files)}")
        if not wheel_image_files:
            print(f"[stage2] ⚠ no wheel images in: '{self.wheel_directory}'")
        wheel_detections = {}

        print(f"\n[stage2] processing {len(wheel_image_files)} wheel images...")

        for i, image_path in enumerate(wheel_image_files):
            filename = f"wheel_{os.path.basename(image_path)}"
            detections = self.detect_markers_with_uncertainty(image_path, use_wheel_params=True)
            if detections:
                wheel_detections[filename] = detections
                board_ids = [d['marker_id'] for d in detections]
                tag_ids   = [tid for d in detections for tid in d.get('tag_ids', [])]
                print(f"  [{i+1}/{len(wheel_image_files)}] {os.path.basename(image_path)}: "
                      f"{len(detections)} board(s) {board_ids}  (tag IDs: {tag_ids})")
            else:
                print(f"  [{i+1}/{len(wheel_image_files)}] {os.path.basename(image_path)}: no markers detected")

        # Wheel detection summary (same format as env detection)
        all_wheel_board_ids = sorted({d['marker_id'] for dets in wheel_detections.values() for d in dets})
        all_wheel_tag_ids   = sorted({tid for dets in wheel_detections.values() for d in dets for tid in d.get('tag_ids', [])})
        cfg = self.apriltag_config
        w_n_m = cfg['wheel_grid_rows'] * cfg['wheel_grid_cols']
        print("\n[stage2] detection summary (wheel):")
        print(f"  Images with detections : {len(wheel_detections)} / {len(wheel_image_files)}")
        if self._detect_error_count:
            print(f"  ⚠ Images that FAILED with an error (not just markerless): "
                  f"{self._detect_error_count} — see 'marker detect fail' lines above")
            self._detect_error_count = 0
        print(f"  Unique board IDs found : {all_wheel_board_ids}")
        print(f"  Underlying tag IDs     : {all_wheel_tag_ids}")
        print(f"  Grid (wheel)           : {cfg['wheel_grid_rows']}×{cfg['wheel_grid_cols']} "
              f"({w_n_m} tags/board) → board_id looked up directly from "
              f"vehicle_markers={self.vehicle_markers}")

        if not wheel_detections:
            print("[stage2] no markers detected in wheel images!")
            return None, None, None, []

        # 2. build the factor graph combining the base result with the wheel images
        print("\n[stage2] creating combined factor graph...")

        wheel_graph, wheel_initial_estimate, wheel_camera_ids = self.create_combined_factor_graph_with_base_result(
            base_result, wheel_detections, camera_type='wheel'
        )

        print(f"  combined graph size: {wheel_graph.size()} factors")
        print(f"  combined variables: {wheel_initial_estimate.size()}")

        # log which marker/camera IDs are in initial estimate
        _log(f"  combined graph: {wheel_graph.size()} factors, {wheel_initial_estimate.size()} variables")
        marker_keys_in_estimate = []
        camera_keys_in_estimate = []
        for key in wheel_initial_estimate.keys():
            sym = gtsam.Symbol(key)
            ch = chr(sym.chr())
            if ch == 'M':
                marker_keys_in_estimate.append(sym.index())
            elif ch == 'C':
                camera_keys_in_estimate.append(sym.index())
        _log(f"  marker IDs in initial_estimate : {sorted(marker_keys_in_estimate)}")
        _log(f"  camera IDs in initial_estimate : {sorted(camera_keys_in_estimate)}")
        _log(f"  vehicle_markers required       : {self.vehicle_markers}")
        missing = [m for m in self.vehicle_markers if m not in marker_keys_in_estimate]
        if missing:
            _log(f"  !! MISSING wheel markers in estimate: {missing}")
            _log(f"     → Wheel camera likely could not be initialized (no overlap with Stage-1 markers)")
        else:
            _log(f"  ✓ All vehicle_markers present in initial_estimate")

        # 3. run optimization
        print("[stage2] running optimization...")

        try:
            wheel_result = self.optimize_poses(wheel_graph, wheel_initial_estimate)
            _log(f"  optimization SUCCESS  result size: {wheel_result.size()}")
            print("[stage2] optimization completed.")

            # Stage 1's environment cameras are not variables of this graph, so
            # bring their optimized poses across; the views expect one result
            # holding every stage's cameras.
            carry_over_stage_poses(wheel_result, base_result, label='[stage2] ')

            # ── Stage 2 marker pose table window ──────────────────────────
            if self.enable_visualization:
                try:
                    _marker_ids_s2 = sorted(
                        gtsam.Symbol(_k).index()
                        for _k in wheel_result.keys()
                        if chr(gtsam.Symbol(_k).chr()) == 'M'
                    )
                    self.show_marker_poses_table(
                        wheel_result, _marker_ids_s2,
                        stage_label="Stage 2 – Wheel Optimization"
                    )
                except Exception as _e:
                    print(f"  [pose-table] stage 2 failed: {_e}")

            return wheel_graph, wheel_result, wheel_detections, wheel_camera_ids

        except Exception as e:
            import traceback
            _log(f"  optimization FAILED: {e}\n{traceback.format_exc()}")
            print(f"[stage2] combined optimization failed: {e}")
            return None, None, None, []

    def create_combined_factor_graph_with_base_result(self, base_result, new_detections, camera_start_id=None, camera_type='wheel', filename_to_camera_idx=None):
        """
        create factor graph with fixed markers and new camera observations

        Args:
            base_result: previous optimization result
            new_detections: new camera observations
            camera_start_id: new camera start ID (None means automatic allocation)
            camera_type: 'wheel' or 'vehicle' - camera type
            filename_to_camera_idx: filename -> camera index mapping (used for vehicle type)

        Returns:
            tuple: (graph, initial_estimate, new_camera_ids)
        """
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        print("\n" + "=" * 60)
        print(f"{'Wheel camera' if camera_type == 'wheel' else 'Vehicle camera'} Factor Graph created")
        print("=" * 60)

        # extract marker IDs from the previous result
        base_marker_ids = []

        for key in base_result.keys():
            symbol = gtsam.Symbol(key)
            if chr(symbol.chr()) == 'M':
                base_marker_ids.append(symbol.index())

        # 1. set initial values for only the previous markers and fix them strongly with prior factors
        for marker_id in base_marker_ids:
            marker_key = gtsam.symbol('M', marker_id)
            marker_pose = base_result.atPose3(marker_key)
            initial_estimate.insert(marker_key, marker_pose)

            # Constrained.All(6) fixes all 6 dimensions of Pose3 completely
            prior_noise = gtsam.noiseModel.Constrained.All(6)  # 6 dimensions (Pose3)
            graph.addPriorPose3(marker_key, marker_pose, prior_noise)

        # 2. assign
        if camera_start_id is not None:
            new_camera_start_id = camera_start_id
        else:
            # default: wheel starts from 10000, vehicle starts from 20000
            if camera_type == 'wheel':
                new_camera_start_id = 10000
            elif camera_type == 'vehicle':
                new_camera_start_id = 20000
            else:
                new_camera_start_id = 10000
        print(f"new camera start ID: {new_camera_start_id} ({camera_type.upper()})")
        filename_to_camera_id = {}

        # assign camera IDs: use filename_to_camera_idx if provided, otherwise sequentially
        if filename_to_camera_idx:
            # use provided mapping (cam_idx -> camera_id)
            print("  using provided filename_to_camera_idx mapping")
            for filename, cam_idx in filename_to_camera_idx.items():
                camera_id = new_camera_start_id + cam_idx
                filename_to_camera_id[filename] = camera_id
            print(f"  filename to camera ID mapping: {filename_to_camera_id}")
        else:
            # Legacy: sequentially assign
            for idx, filename in enumerate(sorted(new_detections.keys())):
                camera_id = new_camera_start_id + idx
                filename_to_camera_id[filename] = camera_id

        # 3. organize new image observations
        marker_observations = defaultdict(list)
        new_camera_ids = set()
        new_marker_ids = set()

        # sort by filename for reproducibility
        for filename, detections in sorted(new_detections.items()):
            if filename in filename_to_camera_id:
                camera_id = filename_to_camera_id[filename]
                new_camera_ids.add(camera_id)

                for detection in detections:
                    marker_id = detection['marker_id']

                    # track new marker IDs
                    if marker_id not in base_marker_ids:
                        new_marker_ids.add(marker_id)

                # calculate SE(3) transformation and noise model
                    T_camera_to_marker = detection['pose_se3']
                    noise_model = detection['noise_model']

                    marker_observations[marker_id].append((camera_id, T_camera_to_marker, noise_model))

        if new_marker_ids:
            print(f"  new detected markers: {sorted(list(new_marker_ids))}")
        else:
            print("  no new markers (all markers already exist)")

        # create reverse mapping from camera_id to filename (for warning messages)
        camera_id_to_filename = {v: k for k, v in filename_to_camera_id.items()}

        # Filter: for wheel cameras, remove cameras that observe fewer than 2 markers total.
        # Build per-camera observation count from marker_observations.
        if camera_type == 'wheel':
            cam_marker_count = defaultdict(set)
            for marker_id, obs_list in marker_observations.items():
                for cam_id, _, _ in obs_list:
                    if cam_id in new_camera_ids:
                        cam_marker_count[cam_id].add(marker_id)
            single_marker_cams = {cid for cid in new_camera_ids if len(cam_marker_count[cid]) < 2}
            if single_marker_cams:
                print(f"⚠ [stage2-wheel] removing {len(single_marker_cams)} cameras with <2 marker observations")
                new_camera_ids -= single_marker_cams
                for marker_id in list(marker_observations.keys()):
                    marker_observations[marker_id] = [
                        (cid, T, nm) for cid, T, nm in marker_observations[marker_id]
                        if cid not in single_marker_cams
                    ]

        # ── PHASE 1 (wheel) ───────────────────────────────────────────────
        #   Collect all marker-pair relative poses from ALL wheel images and
        #   compute median T_Mi_to_Mj for each pair.
        #   Pairs include env→wheel, wheel→env, wheel→wheel.
        if camera_type == 'wheel':
            # build per-camera observation list for wheel cameras only
            wheel_cam_to_obs = defaultdict(list)
            for marker_id, obs_list in marker_observations.items():
                for cam_id, T, nm in obs_list:
                    if cam_id in new_camera_ids:
                        wheel_cam_to_obs[cam_id].append((marker_id, T, nm))

            # PHASE 1 (wheel): cam2mar → mar2mar conversion
            marker_pair_poses_w   = defaultdict(list)
            marker_pair_cam_ids_w = defaultdict(list)
            for cam_id, obs_list in wheel_cam_to_obs.items():
                for i in range(len(obs_list)):
                    mi, T_cam_mi, _ = obs_list[i]
                    for j in range(i + 1, len(obs_list)):
                        mj, T_cam_mj, _ = obs_list[j]
                        T_mi_to_mj = T_cam_mi.inverse().compose(T_cam_mj)
                        lo, hi = (mi, mj) if mi < mj else (mj, mi)
                        if mi < mj:
                            marker_pair_poses_w[(lo, hi)].append(T_mi_to_mj)
                        else:
                            marker_pair_poses_w[(lo, hi)].append(T_mi_to_mj.inverse())
                        marker_pair_cam_ids_w[(lo, hi)].append(cam_id)

            print(f"  [phase1-wheel] {len(marker_pair_poses_w)} unique marker pairs collected")

            # ── Low-count pair filter (wheel) ─────────────────────────────
            if self.min_pair_obs > 0:
                wheel_marker_ids_set = set(new_marker_ids)
                filter_low_count_pairs(
                    marker_pair_poses_w, marker_pair_cam_ids_w,
                    marker_observations, wheel_cam_to_obs,
                    new_camera_ids, wheel_marker_ids_set,
                    min_obs=self.min_pair_obs,
                )
                new_marker_ids = list(wheel_marker_ids_set)

            # PHASE 1.5 (wheel): compute representative pose per pair using median
            marker_pair_median_w = {}
            for pair, poses in marker_pair_poses_w.items():
                rep_pose = median_pose3(poses)
                if rep_pose is not None:
                    marker_pair_median_w[pair] = rep_pose
            print(f"  [phase1-wheel] {len(marker_pair_median_w)} pairs")

            # ── PHASE 2 (wheel) ───────────────────────────────────────────
            #   BFS on marker graph.
            #   env markers are already initialized (fixed from Stage 1).
            #   Use them as anchors to propagate into wheel markers.
            initialized_markers = set(base_marker_ids)
            marker_init_source  = {}

            for _pass in range(len(new_marker_ids) + 1):
                newly_initialized = 0
                for marker_id in sorted(new_marker_ids):
                    if marker_id in initialized_markers:
                        continue
                    marker_key = gtsam.symbol('M', marker_id)

                    T_init = None
                    src_marker = None
                    for (mi, mj), T_mi_to_mj in marker_pair_median_w.items():
                        if mi == marker_id and mj in initialized_markers:
                            T_world_to_mi = initial_estimate.atPose3(gtsam.symbol('M', mj))
                            T_init = T_world_to_mi.compose(T_mi_to_mj.inverse())
                            src_marker = mj
                            break
                        elif mj == marker_id and mi in initialized_markers:
                            T_world_to_mi = initial_estimate.atPose3(gtsam.symbol('M', mi))
                            T_init = T_world_to_mi.compose(T_mi_to_mj)
                            src_marker = mi
                            break

                    if T_init is not None:
                        initial_estimate.insert(marker_key, T_init)
                        initialized_markers.add(marker_id)
                        marker_init_source[marker_id] = src_marker
                        newly_initialized += 1

                if newly_initialized == 0:
                    break

            print(f"  [phase2-wheel] {len(initialized_markers) - len(base_marker_ids)}/{len(new_marker_ids)} wheel markers initialized")

            # ── PHASE 3 (wheel) ───────────────────────────────────────────
            #   Initialize cameras from fully-initialized markers.
            initialized_cameras = set()
            camera_init_source  = {}

            for camera_id in sorted(new_camera_ids):
                camera_key = gtsam.symbol('C', camera_id)
                candidates        = []
                noise_models_list = []
                used_markers_list = []

                for marker_id, T_cam_to_marker, noise_model in wheel_cam_to_obs[camera_id]:
                    if marker_id not in initialized_markers:
                        continue
                    marker_key_m = gtsam.symbol('M', marker_id)
                    if initial_estimate.exists(marker_key_m):
                        T_world_to_marker = initial_estimate.atPose3(marker_key_m)
                        T_world_to_camera = T_world_to_marker.compose(T_cam_to_marker.inverse())
                        candidates.append(T_world_to_camera)
                        noise_models_list.append(noise_model)
                        used_markers_list.append(marker_id)

                if candidates:
                    # A zero-information model has a NaN covariance trace and
                    # np.argmin would pick it; treat it as worst, not best.
                    traces   = [np.trace(nm.covariance()) for nm in noise_models_list]
                    traces   = [t if np.isfinite(t) else np.inf for t in traces]
                    best_idx = int(np.argmin(traces))
                    initial_estimate.insert(camera_key, candidates[best_idx])
                    initialized_cameras.add(camera_id)
                    camera_init_source[camera_id] = used_markers_list[best_idx]

            print(f"  [phase3-wheel] {len(initialized_cameras)}/{len(new_camera_ids)} cameras initialized")

        else:
            # vehicle camera: keep original iterative initialization
            initialized_cameras = set()
            initialized_markers = set(base_marker_ids)
            camera_init_source  = {}
            marker_init_source  = {}

            for iteration in range(5):
                newly_initialized = 0

                for camera_id in sorted(new_camera_ids):
                    if camera_id in initialized_cameras:
                        continue
                    camera_key = gtsam.symbol('C', camera_id)
                    if initial_estimate.exists(camera_key):
                        continue

                    camera_pose_candidates = []
                    noise_models_list      = []
                    used_markers           = []
                    T_cam_to_markers_list  = []

                    for marker_id, observations in sorted(marker_observations.items()):
                        if marker_id not in initialized_markers:
                            continue
                        for obs_camera_id, T_cam_to_marker, noise_model in observations:
                            if obs_camera_id == camera_id:
                                marker_key_m = gtsam.symbol('M', marker_id)
                                if initial_estimate.exists(marker_key_m):
                                    T_world_to_marker = initial_estimate.atPose3(marker_key_m)
                                    T_world_to_camera = T_world_to_marker.compose(T_cam_to_marker.inverse())
                                    camera_pose_candidates.append(T_world_to_camera)
                                    noise_models_list.append(noise_model)
                                    used_markers.append(marker_id)
                                    T_cam_to_markers_list.append(T_cam_to_marker)
                                break

                    if len(camera_pose_candidates) > 0:
                        if len(camera_pose_candidates) >= 2:
                            # vehicle camera: for each candidate pose (derived from marker Mi),
                            # project ALL other markers Mj through that pose and compare against
                            # Stage-1 GT. Pick the candidate with the smallest total residual.
                            total_residuals = []
                            for i, cand_pose in enumerate(camera_pose_candidates):
                                total_rot = 0.0
                                total_pos = 0.0
                                n_others  = 0
                                for j, (other_mid, T_cam_other) in enumerate(
                                        zip(used_markers, T_cam_to_markers_list)):
                                    if j == i:
                                        continue
                                    T_world_mj_meas = cand_pose.compose(T_cam_other)
                                    T_world_mj_gt   = initial_estimate.atPose3(
                                        gtsam.symbol('M', other_mid))
                                    delta   = T_world_mj_gt.inverse().compose(T_world_mj_meas)
                                    lv      = gtsam.Pose3.Logmap(delta)
                                    total_rot += float(np.linalg.norm(lv[:3]))
                                    total_pos += float(np.linalg.norm(lv[3:]))
                                    n_others  += 1
                                mean_rot = total_rot / n_others if n_others else 0.0
                                mean_pos = total_pos / n_others if n_others else 0.0
                                total_residuals.append((mean_rot, mean_pos))

                            best_idx = min(range(len(total_residuals)),
                                           key=lambda i: (total_residuals[i][0]
                                                          + total_residuals[i][1]))
                            camera_pose = camera_pose_candidates[best_idx]
                            src_marker  = used_markers[best_idx]
                            br, bp = total_residuals[best_idx]
                            print(f"  [vehicle C{camera_id}] geo-residual init: "
                                  f"selected M{src_marker} "
                                  f"(rot={np.degrees(br):.2f}° pos={bp*100:.1f}cm) "
                                  f"from {[f'M{m} r={np.degrees(r):.2f}° p={p*100:.1f}cm' for m, (r,p) in zip(used_markers, total_residuals)]}")
                        else:
                            best_idx   = 0
                            camera_pose = camera_pose_candidates[best_idx]
                            src_marker  = used_markers[best_idx]

                        initial_estimate.insert(camera_key, camera_pose)
                        initialized_cameras.add(camera_id)
                        camera_init_source[camera_id] = src_marker
                        newly_initialized += 1

                # initialize new markers from already-initialized cameras
                for new_marker_id in sorted(new_marker_ids):
                    if new_marker_id in initialized_markers:
                        continue
                    marker_key = gtsam.symbol('M', new_marker_id)
                    if initial_estimate.exists(marker_key):
                        continue

                    marker_pose_candidates    = []
                    observation_uncertainties = []
                    used_cameras_v            = []

                    for obs_camera_id, T_cam_to_marker, noise_model in marker_observations[new_marker_id]:
                        if obs_camera_id not in initialized_cameras:
                            continue
                        cam_key_v = gtsam.symbol('C', obs_camera_id)
                        if initial_estimate.exists(cam_key_v):
                            T_world_to_camera = initial_estimate.atPose3(cam_key_v)
                            T_world_to_marker = T_world_to_camera.compose(T_cam_to_marker)
                            marker_pose_candidates.append(T_world_to_marker)
                            observation_uncertainties.append(
                                np.trace(noise_model.covariance())
                            )
                            used_cameras_v.append(obs_camera_id)

                    if marker_pose_candidates:
                        best_cam_idx = int(np.argmin(observation_uncertainties))
                        initial_estimate.insert(marker_key, median_pose3(marker_pose_candidates))
                        initialized_markers.add(new_marker_id)
                        marker_init_source[new_marker_id] = used_cameras_v[best_cam_idx]
                        newly_initialized += 1

                if newly_initialized == 0:
                    break

        # check uninitialized cameras/markers
        uninitialized_cameras = [cid for cid in new_camera_ids if cid not in initialized_cameras]
        uninitialized_new_markers = [mid for mid in new_marker_ids if mid not in initialized_markers]

        if uninitialized_cameras:
            print(f"  ⚠ warning: {len(uninitialized_cameras)} cameras not initialized:")
            for cam_id in uninitialized_cameras:
                filename = camera_id_to_filename.get(cam_id, f"Unknown(C{cam_id})")
                print(f"    - '{filename}' (C{cam_id}) - no overlap with Stage-1 markers")

        if uninitialized_new_markers:
            print(f"  ⚠ warning: {len(uninitialized_new_markers)} new markers not initialized: {uninitialized_new_markers}")
            print("    wheel camera must see at least one Stage-1 marker to be anchored")

        print(f"  [init-summary] cameras initialized: {len(initialized_cameras)}/{len(new_camera_ids)}, "
              f"new markers initialized: {len(initialized_markers)-len(base_marker_ids)}/{len(new_marker_ids)}")

        # 6. add new camera observation factors
        print("\nadding observation factors...")

        skip_rot_rad = self.geo_skip_rot_deg * DEG2RAD
        skip_pos_m   = self.geo_skip_pos_m
        skipped_vehicle = 0
        added_vehicle   = 0

        for marker_id, observations in marker_observations.items():
            marker_key = gtsam.symbol('M', marker_id)

            if not initial_estimate.exists(marker_key):
                continue

            for camera_id, T_cam_to_marker, noise_model in observations:
                camera_key = gtsam.symbol('C', camera_id)

                if not initial_estimate.exists(camera_key):
                    continue

                if camera_type == 'vehicle':
                    # vehicle camera: geometric consistency filter.
                    #
                    # The camera was initialized from its reference marker (camera_init_source).
                    # For every other marker Mk, we compute the marker pose via the camera:
                    #   T_world_Mk_measured = T_world_cam  ×  T_cam_Mk
                    # and compare it against the Stage-1 ground-truth pose:
                    #   T_world_Mk_GT = initial_estimate (fixed by Constrained prior)
                    #
                    # delta = T_world_Mk_GT^{-1}  ×  T_world_Mk_measured
                    # rot_err = ||Logmap(delta)[:3]||,  pos_err = ||Logmap(delta)[3:]||
                    #
                    # If either exceeds the geo_skip threshold (GUI) → discard observation.
                    # Reference marker always passes (error ≈ 0 by construction).
                    T_world_cam      = initial_estimate.atPose3(camera_key)
                    T_world_mk_meas  = T_world_cam.compose(T_cam_to_marker)
                    T_world_mk_gt    = initial_estimate.atPose3(marker_key)

                    delta   = T_world_mk_gt.inverse().compose(T_world_mk_meas)
                    log_vec = gtsam.Pose3.Logmap(delta)
                    rot_err = float(np.linalg.norm(log_vec[:3]))
                    pos_err = float(np.linalg.norm(log_vec[3:]))

                    if rot_err > skip_rot_rad or pos_err > skip_pos_m:
                        ref_mid = camera_init_source.get(camera_id, '?')
                        print(f"  [vehicle C{camera_id}] skip M{marker_id} "
                              f"(ref=M{ref_mid}): "
                              f"rot={np.degrees(rot_err):.2f}° "
                              f"(limit {self.geo_skip_rot_deg:.1f}°), "
                              f"pos={pos_err*100:.1f}cm "
                              f"(limit {skip_pos_m*100:.1f}cm)")
                        skipped_vehicle += 1
                        continue

                    graph.add(gtsam.BetweenFactorPose3(camera_key, marker_key,
                                                        T_cam_to_marker, noise_model))
                    added_vehicle += 1
                else:
                    graph.add(gtsam.BetweenFactorPose3(camera_key, marker_key,
                                                        T_cam_to_marker, noise_model))

        if camera_type == 'vehicle':
            print(f"  [vehicle] factors added: {added_vehicle}, skipped: {skipped_vehicle} "
                  f"(geo_skip rot={self.geo_skip_rot_deg}° pos={skip_pos_m*100:.1f}cm)")

        initial_estimate = remove_disconnected_nodes(graph, initial_estimate, new_camera_ids)
        return graph, initial_estimate, sorted(list(new_camera_ids))

    def visualize_poses_3d_optimized_wheel_only(self, optimized_result, marker_ids=None, save_path=None):
        """Visualize only the markers and wheel cameras (existing cameras excluded)."""
        fig = MplFigure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # extract and visualize marker positions
        marker_positions_opt = []
        marker_id_to_pos_opt = {}

        if marker_ids:
            for marker_id in marker_ids:
                key = gtsam.symbol('M', marker_id)
                if optimized_result.exists(key):
                    pose = optimized_result.atPose3(key)
                    pos = pose.translation()
                    position = [pos[0], pos[1], pos[2]]
                    marker_positions_opt.append(position)
                    marker_id_to_pos_opt[marker_id] = position

        # marker visualization (wheel markers blue, env markers red)
        if marker_positions_opt:
            wheel_set = set(self.vehicle_markers)
            env_ids_viz   = [mid for mid in marker_ids if mid in marker_id_to_pos_opt and mid not in wheel_set]
            wheel_ids_viz = [mid for mid in marker_ids if mid in marker_id_to_pos_opt and mid in wheel_set]

            if env_ids_viz:
                env_pos = np.array([marker_id_to_pos_opt[mid] for mid in env_ids_viz])
                ax.scatter(env_pos[:, 0], env_pos[:, 1], env_pos[:, 2],
                           c='red', s=120, marker='s', label='Env markers', alpha=0.8)

            if wheel_ids_viz:
                wheel_pos = np.array([marker_id_to_pos_opt[mid] for mid in wheel_ids_viz])
                ax.scatter(wheel_pos[:, 0], wheel_pos[:, 1], wheel_pos[:, 2],
                           c='blue', s=150, marker='s', label='Wheel markers', alpha=0.9)

            for marker_id in marker_ids:
                if marker_id in marker_id_to_pos_opt:
                    pos = marker_id_to_pos_opt[marker_id]
                    color = 'blue' if marker_id in wheel_set else 'red'
                    ax.text(pos[0], pos[1], pos[2], f'M{marker_id}',
                            fontsize=16, fontweight='bold', color='black')

        # ── Marker-Marker (M2M) drawing M2M connections ───────────────────
        if hasattr(self, 'all_detections_for_viz'):
            unique_pairs = set()
            for filename, detections in self.all_detections_for_viz.items():
                if len(detections) > 1:
                    ids = [d['marker_id'] for d in detections]
                    for i in range(len(ids)):
                        for j in range(i + 1, len(ids)):
                            pair = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                            unique_pairs.add(pair)
            m2m_count = 0
            for idx, (m1, m2) in enumerate(unique_pairs):
                if m1 in marker_id_to_pos_opt and m2 in marker_id_to_pos_opt:
                    pos1, pos2 = marker_id_to_pos_opt[m1], marker_id_to_pos_opt[m2]
                    ax.plot3D([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                              'r-', alpha=0.7, linewidth=1,
                              label='Marker-Marker Connections' if idx == 0 else "")
                    m2m_count += 1
            print(f"[viz] drawn M2M connections: {m2m_count} (unique pairs)")

        ax.set_xlabel('X (m)', fontsize=16)
        ax.set_ylabel('Y (m)', fontsize=16)
        ax.set_zlabel('Z (m)', fontsize=16)
        ax.legend(fontsize=16, markerscale=1.3)
        ax.set_title('Stage 2: Env. markers and wheel markers graph', fontsize=20, fontweight='bold')

        if save_path:
            # fig is not pyplot-managed, so plt.savefig would write an empty
            # "current figure" instead — save through the figure itself.
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  [viz] Stage 2 visualization saved to: {save_path}")

        return fig

    # ── Pose scenes: optimized boards and the cameras that observed them ──

    def wheel_reference_marker_id(self):
        """Board id of the rear-left vehicle marker, or None if unconfigured.

        `vehicle_markers` is ordered [FL, RL, RR, FR], so index 1 is the marker
        on the left rear of the vehicle. Used as the fallback origin for the
        wheel-camera scene when the full vehicle frame cannot be built.
        """
        markers = self.vehicle_markers or []
        return markers[1] if len(markers) > 1 else None

    def vehicle_frame_pose(self, optimized_result):
        """The vehicle frame as a Pose3 in the optimizer's world frame, or None.

        X forward, Y left, Z up, origin at the rear axle centre — the same frame
        Export Poses writes camera extrinsics in, so a pose read off the plot
        matches the exported numbers.
        """
        frame = getattr(self, 'cached_vehicle_frame', None)
        if frame is None:
            self.add_vehicle_coordinate_system_post_optimization(optimized_result)
            frame = getattr(self, 'cached_vehicle_frame', None)
        if frame is None:
            return None
        return gtsam.Pose3(
            gtsam.Rot3(np.asarray(frame['R_vehicle_to_world'], dtype=float)),
            np.asarray(frame['center'], dtype=float).reshape(3))

    def _reference_frame_inverse(self, optimized_result, marker_ids,
                                 ref_pose=None, ref_marker_id=None):
        """Inverse of the pose everything is drawn relative to.

        The optimizer's world frame is whichever marker happened to anchor the
        graph, so expressing the scene in the configured first_marker_id's frame
        keeps the origin fixed across stages and reruns.

        A view can override that origin: `ref_pose` is an explicit frame in world
        coordinates (the vehicle frame, for the wheel-camera scene), `ref_marker_id`
        names a marker to use instead. Whichever is given is tried first, with the
        default anchor as fallback.
        """
        if ref_pose is not None:
            return ref_pose.inverse()

        candidates = []
        if ref_marker_id is not None:
            candidates.append(ref_marker_id)
        if not marker_ids:
            if not candidates:
                return gtsam.Pose3()
        else:
            candidates.append(self.first_marker_id
                              if (self.first_marker_id is not None
                                  and self.first_marker_id in marker_ids)
                              else min(marker_ids))

        for ref_id in candidates:
            if ref_id is None:
                continue
            ref_key = gtsam.symbol('M', ref_id)
            if optimized_result.exists(ref_key):
                return optimized_result.atPose3(ref_key).inverse()
        return gtsam.Pose3()

    def vehicle_scene_reference(self, optimized_result):
        """(ref_pose, ref_marker_id, label) for the vehicle-anchored pose scenes.

        Prefers the vehicle frame and degrades to the rear-left vehicle marker
        when the four wheel boards are not all in the result, since that is the
        only case where the vehicle axes cannot be derived.
        """
        ref_pose = self.vehicle_frame_pose(optimized_result)
        if ref_pose is not None:
            return ref_pose, None, None

        ref_marker_id = self.wheel_reference_marker_id()
        if ref_marker_id is not None:
            return None, ref_marker_id, f'Rear-left vehicle marker M{ref_marker_id}'
        return None, None, None

    def board_outline_local(self, board_id):
        """The board's four outer corners in its own frame, shape (4, 3).

        board_object_points returns every tag slot's corners; the printed board is
        their bounding rectangle on the z = 0 plane, which is what should be drawn
        as the board.
        """
        points = np.asarray(self.board_object_points(board_id), dtype=float).reshape(-1, 3)
        if points.shape[0] < 4:
            return None
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()
        return np.array([
            [x_min, y_min, 0.0],
            [x_max, y_min, 0.0],
            [x_max, y_max, 0.0],
            [x_min, y_max, 0.0],
        ])

    def collect_pose_scene(self, optimized_result, marker_ids=None, camera_groups=None,
                           ref_pose=None, ref_marker_id=None, edges=None):
        """Gather optimized boards and camera poses in the reference-marker frame.

        Shared by the Matplotlib and Open3D renderers so both draw exactly the
        same geometry.

        Args:
            camera_groups: list of dicts with keys 'name', 'ids' and 'camera_params'.
            ref_pose: Pose3 in world coordinates whose frame becomes the origin.
            ref_marker_id: board id whose frame becomes the origin, overriding the
                default first_marker_id anchor.
            edges: optional (camera_id, marker_id) pairs (see observation_edges);
                links to boards outside marker_ids are not drawn.

        Returns:
            dict with 'quads', 'normals' and 'labels' keyed by 'env'/'wheel', a
            'cameras' dict of group name to [(position, rotation)], an 'edges'
            dict of group name to [(camera position, board centre)], and the
            stacked 'points' used for scaling the view.
        """
        T_ref_inv = self._reference_frame_inverse(optimized_result, marker_ids or [],
                                                  ref_pose=ref_pose,
                                                  ref_marker_id=ref_marker_id)
        wheel_set = set(self.vehicle_markers)

        scene = {
            'quads': {'env': [], 'wheel': []},
            'normals': {'env': [], 'wheel': []},
            'labels': [],
            'cameras': {},
            'edges': {},
            'points': [],
        }
        marker_centres = {}
        markers_seen_by = defaultdict(list)
        for camera_id, marker_id in (edges or ()):
            markers_seen_by[camera_id].append(marker_id)

        for marker_id in (marker_ids or []):
            key = gtsam.symbol('M', marker_id)
            if not optimized_result.exists(key):
                continue
            outline = self.board_outline_local(marker_id)
            if outline is None:
                continue

            pose = T_ref_inv.compose(optimized_result.atPose3(key))
            rotation = pose.rotation().matrix()
            translation = np.asarray(pose.translation(), dtype=float).reshape(3)
            quad = outline @ rotation.T + translation

            group = 'wheel' if marker_id in wheel_set else 'env'
            scene['quads'][group].append(quad)

            centre = quad.mean(axis=0)
            marker_centres[marker_id] = centre
            # The +Z stub is what distinguishes a board's front from its back once
            # the quad is seen edge-on.
            stub = float(np.linalg.norm(outline[1] - outline[0])) * 0.6
            scene['normals'][group].append([centre, centre + rotation[:, 2] * stub])
            scene['labels'].append((centre, marker_id, group == 'wheel'))
            scene['points'].append(quad)

        for group in (camera_groups or []):
            poses = []
            links = []
            for camera_id in group.get('ids') or []:
                key = gtsam.symbol('C', camera_id)
                if not optimized_result.exists(key):
                    continue
                pose = T_ref_inv.compose(optimized_result.atPose3(key))
                position = np.asarray(pose.translation(), dtype=float).reshape(3)
                poses.append((position, pose.rotation().matrix()))
                scene['points'].append(position.reshape(1, 3))
                for marker_id in markers_seen_by.get(camera_id, ()):
                    if marker_id in marker_centres:
                        links.append((position, marker_centres[marker_id]))
            scene['cameras'][group['name']] = poses
            scene['edges'][group['name']] = links

        scene['points'] = (np.vstack(scene['points']) if scene['points']
                           else np.zeros((0, 3)))
        return scene

    def visualize_markers_and_cameras_3d(self, optimized_result, marker_ids=None,
                                         camera_groups=None, title='Optimized poses',
                                         show_labels=True, elev=22, azim=-60,
                                         save_path=None, ref_pose=None,
                                         ref_marker_id=None, ref_label=None,
                                         edges=None):
        """Boards as oriented quads and cameras as view frustums, batch rendered.

        The whole scene is a handful of collections rather than one artist per
        edge, so the view stays interactive with hundreds of camera poses.
        edges (camera_id, marker_id) are drawn as faint thin camera→board links;
        per group 'edge_alpha' / 'edge_linewidth' tune them.
        """
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        camera_groups = camera_groups or []
        scene = self.collect_pose_scene(optimized_result, marker_ids, camera_groups,
                                        ref_pose=ref_pose, ref_marker_id=ref_marker_id,
                                        edges=edges)

        fig = MplFigure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        depth = scene_frustum_depth(scene['points'])
        handles = []

        for group_key, group_label in (('env', 'Env markers'), ('wheel', 'Wheel markers')):
            quads = scene['quads'][group_key]
            if not quads:
                continue
            color = SCENE_COLORS[f'{group_key}_marker']
            add_marker_quads(ax, quads, color, normals=scene['normals'][group_key])
            handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.55,
                                 label=f'{group_label} ({len(quads)})'))

        for group in camera_groups:
            poses = scene['cameras'].get(group['name'], [])
            if not poses:
                continue
            fov_h, fov_v = fov_from_intrinsics(group['camera_params'])
            color = group['color']
            edge_count = add_observation_edges(
                ax, scene['edges'].get(group['name'], []), color,
                alpha=group.get('edge_alpha', 0.2),
                linewidth=group.get('edge_linewidth', 0.4))
            drawn = add_camera_frustums(
                ax, poses, fov_h, fov_v, depth * group.get('depth_scale', 1.0),
                color, alpha=group.get('alpha', 0.9),
                face_alpha=group.get('face_alpha', 0.18),
                linewidth=group.get('linewidth', 1.0))
            handles.append(Patch(facecolor=color, edgecolor=color,
                                 alpha=group.get('alpha', 0.7),
                                 label=f"{group['name']} ({drawn})"))
            if edge_count:
                # The scene links are too faint to read at legend size, so the
                # swatch is thinner than the frustum entry but more opaque.
                handles.append(Line2D([0], [0], color=color, linewidth=1.0,
                                      alpha=max(group.get('edge_alpha', 0.5), 0.6),
                                      label=f"Camera→marker edges ({edge_count})"))

        # Origin triad: with the scene expressed in a marker's frame, the axes
        # only mean something if it is visible which marker that is.
        axis_length = depth * 3.0 if ref_pose is not None else depth * 2.0
        axis_names = (('X (fwd)', 'Y (left)', 'Z (up)') if ref_pose is not None
                      else ('X', 'Y', 'Z'))
        for axis, axis_color in ((0, 'red'), (1, 'green'), (2, 'blue')):
            end = np.zeros(3)
            end[axis] = axis_length
            ax.plot([0, end[0]], [0, end[1]], [0, end[2]],
                    color=axis_color, linewidth=2.5, alpha=0.9)
            ax.text(end[0], end[1], end[2], axis_names[axis],
                    fontsize=20, fontweight='bold', color=axis_color)
        # if ref_label:
        #     handles.append(Line2D([0], [0], color='black', linewidth=2,
        #                           label=f'Origin: {ref_label}'))

        if show_labels:
            for centre, marker_id, is_wheel in scene['labels']:
                ax.text(centre[0], centre[1], centre[2], f'M{marker_id}',
                        fontsize=20, fontweight='bold',
                        color=SCENE_COLORS['wheel_marker' if is_wheel else 'env_marker'])

        # Collections do not grow the data limits reliably, so set them from the
        # geometry and leave the GUI to equalise the aspect afterwards.
        if len(scene['points']):
            low = scene['points'].min(axis=0) - depth
            high = scene['points'].max(axis=0) + depth
            ax.set_xlim3d(low[0], high[0])
            ax.set_ylim3d(low[1], high[1])
            ax.set_zlim3d(low[2], high[2])

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title, fontsize=24, fontweight='bold')
        ax.grid(True)
        if handles:
            ax.legend(handles=handles, loc='upper left', fontsize=20)
        ax.view_init(elev=elev, azim=azim)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  [viz] pose scene saved to: {save_path}")
        return fig

    def stage1_camera_groups(self, camera_ids):
        """Environment cameras only - the boards-and-cameras set Stage 1 solves."""
        return [{
            'name': 'Auxiliary cameras',
            'ids': sorted(c for c in (camera_ids or []) if c < 1000),
            'camera_params': self.camera_params,
            'color': SCENE_COLORS['env_camera'],
            'depth_scale': 0.6,
            # Observation links: darker than the 0.2 / 0.4 default.
            'edge_alpha': 0.5,
            'edge_linewidth': 0.4,
        }]

    def stage2_camera_groups(self, wheel_camera_ids, env_camera_ids=None):
        """Wheel cameras in front, environment cameras behind them for context.

        Stage 2 only adds the wheel views, but the environment cameras are what
        pinned the boards they attach to, so drawing them faintly shows how the
        two stages relate instead of leaving the wheel poses floating.
        """
        return [
            # {
            #     'name': 'Env cameras (Stage 1)',
            #     'ids': sorted(c for c in (env_camera_ids or []) if c < 1000),
            #     'camera_params': self.camera_params,
            #     'color': SCENE_COLORS['context_camera'],
            #     # Context only: kept faint enough that the wheel frustums stay
            #     # the thing the eye lands on in a scene with far more env views.
            #     'alpha': 0.12,
            #     'face_alpha': 0.0,
            #     'linewidth': 0.5,
            #     'depth_scale': 0.6,
            # },
            {
                'name': 'Wheel cameras',
                'ids': sorted(wheel_camera_ids or []),
                'camera_params': self.wheel_camera_params,
                'color': SCENE_COLORS['wheel_camera'],
                'alpha': 1.0,
                'linewidth': 1.6,
                'face_alpha': 0.28,
                'depth_scale': 0.6,
                # Far fewer links than the env view, so they can be less faint.
                'edge_alpha': 0.4,
                'edge_linewidth': 0.6,

            },
        ]

    def stage4_camera_groups(self, vehicle_camera_ids):
        """Every vehicle camera in one group: one colour, one frustum shape.

        The frustum is a schematic of where a camera looks rather than a
        measurement, so the cones share the first configured vehicle intrinsics
        instead of varying per camera.
        """
        shared_params = None
        for config in self.vehicle_camera_configs or []:
            intrinsic = config.get('intrinsic') if isinstance(config, dict) else None
            if intrinsic:
                shared_params = intrinsic
                break

        return [{
            'name': 'Vehicle cameras',
            'ids': sorted(vehicle_camera_ids or []),
            'camera_params': shared_params or self.camera_params,
            'color': SCENE_COLORS['vehicle_camera'],
            'alpha': 1.0,
            'linewidth': 1.6,
            'face_alpha': 0.25,
            # A handful of cameras sit inside a room-sized scene, so the
            # scene-scaled default cone swamps the vehicle they belong to.
            'depth_scale': 0.8,
            # Only a couple of dozen links: visible, still behind the frustums.
            'edge_alpha': 0.5,
            'edge_linewidth': 0.7,
        }]

    def visualize_stage4_markers_and_cameras(self, optimized_result, marker_ids=None,
                                             vehicle_camera_ids=None, save_path=None,
                                             edges=None):
        """Stage 4: the Stage 3 result in the pose-scene style.

        Boards as oriented quads and vehicle cameras as frustums, drawn in the
        vehicle frame — the same rendering as the wheel-camera pose scene, which
        stays legible where Stage 3's edge bundle does not. Observation edges,
        when given, are drawn faint and thin behind the frustums.
        """
        ref_pose, ref_marker_id, ref_label = self.vehicle_scene_reference(optimized_result)
        title = 'Stage 2: optimized markers + vehicle cameras'
        if ref_label:
            title += f'\n(origin: {ref_label})'
        return self.visualize_markers_and_cameras_3d(
            optimized_result,
            marker_ids=marker_ids,
            camera_groups=self.stage4_camera_groups(vehicle_camera_ids),
            title=title,
            save_path=save_path,
            ref_pose=ref_pose,
            ref_marker_id=ref_marker_id,
            ref_label=ref_label,
            edges=edges,
        )

    def visualize_stage1_markers_and_cameras(self, optimized_result, marker_ids=None,
                                             camera_ids=None, save_path=None, edges=None):
        """Stage 1: environment boards with the cameras that photographed them."""
        return self.visualize_markers_and_cameras_3d(
            optimized_result,
            marker_ids=marker_ids,
            camera_groups=self.stage1_camera_groups(camera_ids),
            title='Stage 1: environment markers + auxiliary camera',
            save_path=save_path,
            edges=edges,
        )

    def visualize_stage2_markers_and_cameras(self, optimized_result, marker_ids=None,
                                             wheel_camera_ids=None, env_camera_ids=None,
                                             save_path=None, edges=None):
        """Stage 2: the same boards plus the wheel cameras added on top.

        Drawn in the vehicle frame: the wheel cameras are calibrated against the
        vehicle, so the numbers on the axes are only readable if the origin sits
        on the vehicle too.
        """
        ref_pose, ref_marker_id, ref_label = self.vehicle_scene_reference(optimized_result)
        title = 'Stage 2: optimized env. markers + auxiliary camera'
        if ref_label:
            title += f'\n(origin: {ref_label})'
        return self.visualize_markers_and_cameras_3d(
            optimized_result,
            marker_ids=marker_ids,
            camera_groups=self.stage2_camera_groups(wheel_camera_ids, env_camera_ids),
            title=title,
            save_path=save_path,
            ref_pose=ref_pose,
            ref_marker_id=ref_marker_id,
            ref_label=ref_label,
            edges=edges,
        )

    def build_pose_scene_open3d(self, optimized_result, marker_ids=None,
                                camera_groups=None, ref_pose=None, ref_marker_id=None,
                                edges=None):
        """Same scene as the Matplotlib views, as Open3D geometries.

        Open3D renders through OpenGL, so rotating a scene with thousands of
        frustum edges stays smooth where Matplotlib's software 3D cannot.
        """
        import open3d as o3d
        from matplotlib.colors import to_rgb

        camera_groups = camera_groups or []
        scene = self.collect_pose_scene(optimized_result, marker_ids, camera_groups,
                                        ref_pose=ref_pose, ref_marker_id=ref_marker_id,
                                        edges=edges)
        depth = scene_frustum_depth(scene['points'])
        geometries = []

        for group_key in ('env', 'wheel'):
            quads = scene['quads'][group_key]
            if not quads:
                continue
            colour = to_rgb(SCENE_COLORS[f'{group_key}_marker'])
            front = np.vstack(quads)
            triangles = np.vstack([[[i, i + 1, i + 2], [i, i + 2, i + 3]]
                                   for i in range(0, len(front), 4)])
            # Open3D culls back faces, so a board seen from behind vanishes. Give
            # each quad a second, reverse-wound copy on its own vertices; culling
            # keeps exactly one of the pair visible, so they cannot z-fight and
            # each side still gets a correctly oriented normal for shading.
            vertices = np.vstack([front, front])
            triangles = np.vstack([triangles,
                                   triangles[:, ::-1] + len(front)])
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices),
                o3d.utility.Vector3iVector(triangles))
            mesh.paint_uniform_color(colour)
            mesh.compute_vertex_normals()
            geometries.append(mesh)

            normals = scene['normals'][group_key]
            if normals:
                stub_points = np.asarray(normals, dtype=float).reshape(-1, 3)
                stubs = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(stub_points),
                    o3d.utility.Vector2iVector(
                        np.arange(len(stub_points)).reshape(-1, 2)))
                stubs.paint_uniform_color(colour)
                geometries.append(stubs)

        for group in camera_groups:
            poses = scene['cameras'].get(group['name'], [])
            if not poses:
                continue
            fov_h, fov_v = fov_from_intrinsics(group['camera_params'])
            segments, _ = frustum_edges_and_faces(
                poses, fov_h, fov_v, depth * group.get('depth_scale', 1.0))
            points = segments.reshape(-1, 3)
            lines = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(points),
                o3d.utility.Vector2iVector(np.arange(len(points)).reshape(-1, 2)))
            lines.paint_uniform_color(to_rgb(group['color']))
            geometries.append(lines)

            links = scene['edges'].get(group['name'], [])
            if links:
                # LineSet has no alpha: fade toward the white background instead.
                fade = group.get('edge_alpha', 0.2)
                faint = np.asarray(to_rgb(group['color'])) * fade + (1.0 - fade)
                link_points = np.asarray(links, dtype=float).reshape(-1, 3)
                link_set = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(link_points),
                    o3d.utility.Vector2iVector(np.arange(len(link_points)).reshape(-1, 2)))
                link_set.paint_uniform_color(faint)
                geometries.append(link_set)

        # A frame at the reference marker gives the otherwise axis-free Open3D
        # window something to orient against.
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(depth * 2.0, 0.1)))
        return geometries, scene

    def show_pose_scene_open3d(self, optimized_result, marker_ids=None,
                               camera_groups=None, window_name='Pose scene',
                               ref_pose=None, ref_marker_id=None, edges=None):
        """Open the pose scene in an interactive Open3D window."""
        import open3d as o3d

        geometries, scene = self.build_pose_scene_open3d(
            optimized_result, marker_ids, camera_groups,
            ref_pose=ref_pose, ref_marker_id=ref_marker_id, edges=edges)
        if not len(scene['points']):
            return scene
        o3d.visualization.draw_geometries(geometries, window_name=window_name,
                                          width=1280, height=860)
        return scene

    def detect_markers_with_uncertainty(self, image_path, camera_params=None, use_wheel_params=False, dist_coeffs=None, camera_model='pinhole'):
        """
        Detect markers and compute uncertainty simultaneously.
        """
        _dbg_path = os.path.join(g_current_dir, "debug_wheel.log") if use_wheel_params else None
        def _log(msg):
            if _dbg_path:
                with open(_dbg_path, 'a', encoding='utf-8') as _f:
                    _f.write(msg + '\n')

        try:
            # Decide BEFORE the block below, which assigns self.camera_params into
            # camera_params — after that every path has a non-None camera_params and
            # this test can no longer tell them apart.
            is_vehicle_camera = (camera_params is not None and not use_wheel_params)

            # Camera parameter selection (mutually exclusive by call-site design):
            #   vehicle camera : camera_params + dist_coeffs passed explicitly
            #   wheel camera   : use_wheel_params=True  → self.wheel_camera_params
            #   env camera     : default (both omitted)  → self.camera_params
            if camera_params is not None:
                # vehicle camera: intrinsic and dist_coeffs are passed explicitly
                camera_matrix = np.array([
                    [camera_params[0], 0, camera_params[2]],
                    [0, camera_params[1], camera_params[3]],
                    [0, 0, 1]
                ])
                if dist_coeffs is not None:
                    if len(dist_coeffs) == 2:
                        dist_coeffs = np.array([dist_coeffs[0], dist_coeffs[1], 0.0, 0.0, 0.0])
                    else:
                        dist_coeffs = np.array(dist_coeffs)
                else:
                    raise ValueError(
                        f"detect_markers_with_uncertainty: camera_params was provided but "
                        f"dist_coeffs is None. Vehicle camera must pass dist_coeffs explicitly."
                    )
            elif use_wheel_params:
                # wheel camera
                camera_params = self.wheel_camera_params
                camera_matrix = np.array([
                    [camera_params[0], 0, camera_params[2]],
                    [0, camera_params[1], camera_params[3]],
                    [0, 0, 1]
                ])
                dist_coeffs = self.wheel_dist_coeffs
            else:
                # environment camera
                camera_params = self.camera_params
                camera_matrix = self.camera_matrix
                dist_coeffs = self.dist_coeffs

            # camera type label for debug output
            if camera_params is not self.camera_params and camera_params is not self.wheel_camera_params:
                _cam_label = "vehicle"
            elif use_wheel_params:
                _cam_label = "wheel"
            else:
                _cam_label = "env"
            _log(f"  detect: {os.path.basename(image_path)}")
            _log(f"    camera_type        : {_cam_label}")
            _log(f"    use_wheel_params   : {use_wheel_params}")
            _log(f"    self.camera_params : {self.camera_params}")
            _log(f"    self.wheel_cam_params: {self.wheel_camera_params}")
            _log(f"    resolved cam_params: {camera_params}  ← actually used")
            _log(f"    resolved dist_coeffs: {dist_coeffs}  ← actually used")
            _log(f"    apriltag_config    : {self.apriltag_config}")
            _log(f"    vehicle_markers    : {self.vehicle_markers}")
            _log(f"    marker_size        : {self.marker_size}  marker_size2: {self.marker_size2}")

            image = cv2.imread(image_path)
            if image is None:
                print(f"[detect] ⚠ cannot read image: {image_path}")
                _log(f"    ERROR: cv2.imread returned None \n")
                return []

            trusted_marker_ids = None
            # No full-image undistortion for either model.
            # Pinhole: cv2.undistortPoints on corners before solvePnP
            # Fisheye: cv2.fisheye.undistortPoints on corners before solvePnP

            # image preprocessing (using original image; distortion is handled in PnP)
            blur = cv2.GaussianBlur(image, (0, 0), BLUR_SIG)
            gray_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

            # marker detection
            marker_detections = self.detect_april(gray_img)
            _log(f"    raw detections : {len(marker_detections)} boards → "
                 f"{[(d['marker_id'], d['tag_ids']) for d in marker_detections]}")

            # skip entire image if any excluded apriltag ID is detected
            if self.excluded_apriltag_ids:
                detected_ids = {d['marker_id'] for d in marker_detections}
                hit_ids = detected_ids & self.excluded_apriltag_ids
                if hit_ids:
                    print(f"  [excluded-tag] {os.path.basename(image_path)} skipped "
                          f"(excluded IDs detected: {sorted(hit_ids)})")
                    return []

            # filter: keep only markers confirmed by the undistorted pass
            if trusted_marker_ids is not None:
                before = [d['marker_id'] for d in marker_detections]
                marker_detections = [d for d in marker_detections
                                     if d['marker_id'] in trusted_marker_ids]
                after = [d['marker_id'] for d in marker_detections]
                removed = set(before) - set(after)
                if removed:
                    print(f"  [undistort-filter] removed markers not in undist candidates: {sorted(removed)}")
                _log(f"    after undistort filter: {after}")

            if not marker_detections:
                return []

            detections_with_uncertainty = []
            _pnp_param_printed = getattr(self, '_pnp_param_printed_env', False)
            if use_wheel_params:
                _pnp_param_printed = getattr(self, '_pnp_param_printed_wheel', False)

            if not _pnp_param_printed:
                label = "WHEEL" if use_wheel_params else "ENV"
                if is_vehicle_camera:
                    label = f"VEHICLE [{camera_model.upper()}]"
                print(f"  [solvePnP params - {label}] first image: {os.path.basename(image_path)}")
                print(f"    camera_matrix :\n{camera_matrix}")
                print(f"    dist_coeffs   : {dist_coeffs.flatten().tolist() if dist_coeffs is not None else None}")
                if is_vehicle_camera:
                    print(f"    camera_model  : {camera_model}  ← PnP will use {'fisheye.undistortPoints + eye(3)' if camera_model == 'fisheye' else 'pinhole solvePnP'}")
                if use_wheel_params:
                    self._pnp_param_printed_wheel = True
                else:
                    self._pnp_param_printed_env = True

            for detection in marker_detections:
                obj_pts = detection['object_points'].astype(np.float64)
                img_pts = detection['corners'].astype(np.float64)
                marker_id = detection['marker_id']

                try:
                    if is_vehicle_camera and camera_model == 'fisheye':
                        # Vehicle + Fisheye KB: undistort only corner points → solvePnP with K=I
                        img_pts_norm = cv2.fisheye.undistortPoints(
                            img_pts.reshape(-1, 1, 2),
                            camera_matrix,
                            dist_coeffs.reshape(-1, 1)
                        ).reshape(-1, 2)
                        print(f"  [pnp-model] marker {marker_id}: FISHEYE KB (fisheye.undistortPoints + eye(3))")
                        success, rvec, tvec = cv2.solvePnP(
                            obj_pts, img_pts_norm,
                            np.eye(3), None,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                    elif is_vehicle_camera:
                        # Vehicle + Pinhole: solvePnP with K, D directly (more accurate than undistortPoints + eye(3))
                        success, rvec, tvec = cv2.solvePnP(
                            obj_pts, img_pts.reshape(-1, 1, 2),
                            camera_matrix, dist_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                    else:
                        # Env / Wheel: original pinhole solvePnP (unchanged)
                        success, rvec, tvec = cv2.solvePnP(
                            obj_pts, img_pts, camera_matrix, dist_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                except cv2.error:
                    continue
                _log(f"    PnP marker {marker_id:3d}: success={success}")
                if not success:
                    continue

                rotation_matrix, _ = cv2.Rodrigues(rvec)

                tvec_m = tvec.reshape(3)
                se3_pose = gtsam.Pose3(gtsam.Rot3(rotation_matrix), gtsam.Point3(tvec_m))

                # **Core: Uncertainty calculation (MARSCOT)**
                cov_camera_model = ('fisheye'
                                    if (is_vehicle_camera and camera_model == 'fisheye')
                                    else 'pinhole')

                uncertainty_matrix = calculate_pose_covariance_april(  # MARSCOT
                    corners=detection['corners'],
                    preprocessed_image=gray_img,
                    camera_params=camera_params,
                    marker_id=marker_id,
                    marker_size=self.marker_size,
                    marker_size2=self.marker_size2,
                    dist_coeffs=dist_coeffs,
                    vehicle_markers=self.vehicle_markers,
                    object_points=obj_pts,
                    camera_model=cov_camera_model
                )

                # Drop the detection when MARSCOT cannot compute its covariance
                # (a fallback would grant unearned confidence).
                if uncertainty_matrix is None:
                    self._cov_failed_detections += 1
                    print(f"  [MARSCOT] {os.path.basename(image_path)} marker {marker_id}: "
                          f"covariance unavailable, observation dropped "
                          f"(total dropped so far: {self._cov_failed_detections})")
                    continue

                noise_model = create_noise_model_from_uncertainty(uncertainty_matrix)

                detections_with_uncertainty.append({
                    'marker_id': marker_id,
                    'tag_ids': detection.get('tag_ids', [marker_id]),
                    'corners': detection['corners'],
                    'corners_3d': detection['object_points'],
                    'pose_se3': se3_pose,
                    'uncertainty_matrix': uncertainty_matrix,
                    'noise_model': noise_model,
                    'rvec': rvec.reshape(3),
                    'tvec': tvec.reshape(3),
                })

            return detections_with_uncertainty

        except (ImportError, ValueError) as e:
            # Missing AprilTag 3 bindings or a bad detector setting would fail on
            # every image, so surface it instead of reporting a per-image miss.
            raise RuntimeError(f"AprilTag detection is misconfigured: {e}") from e
        except Exception as e:
            self._detect_error_count += 1
            print(f"[detect] marker detect fail: {os.path.basename(image_path)} ({type(e).__name__}: {e})")
            return []

    def create_uncertainty_factor_graph(self, all_detections, image_files):
        """
        factor graph creation based on uncertainty information

        Args:
            all_detections: detection results for each image (uncertainty included)
            image_files: list of image files

        Returns:
            tuple: (graph, initial_estimate)
        """
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        # assign camera ID based on file name order
        filename_to_camera_id = {}
        for idx, image_path in enumerate(image_files):
            filename = os.path.basename(image_path)
            filename_to_camera_id[filename] = 100 + idx

        # observation data organization
        marker_observations = defaultdict(list)
        camera_ids = set()
        marker_ids = set()

        for filename, detections in all_detections.items():
            camera_id = filename_to_camera_id[filename]
            camera_ids.add(camera_id)

            for detection in detections:
                marker_id = detection['marker_id']
                marker_ids.add(marker_id)

                # SE(3) transformation from camera to marker and uncertainty
                T_camera_to_marker = detection['pose_se3']
                noise_model = detection['noise_model']  # uncertainty-based noise model


                marker_observations[marker_id].append((camera_id, T_camera_to_marker, noise_model))

        # ── Variables setting ─────────────────────────────────────────────
        # if self.first_marker_id is specified, find the image with the corresponding marker first
        first_filename = None
        first_marker_id = None

        if self.first_marker_id is not None:
            # find the specified marker in all images
            for filename, detections_list in all_detections.items():
                markers_in_image = [d['marker_id'] for d in detections_list]
                if self.first_marker_id in markers_in_image:
                    first_filename = filename
                    first_marker_id = self.first_marker_id
                    print(f"[stage1] ✅ set the specified marker M{first_marker_id} as World origin (image: {filename})")
                    break

            if first_marker_id is None:
                # if the specified marker is not found in all images, warn and automatically select
                print(f"[stage1] ⚠ the specified marker M{self.first_marker_id} is not detected in all images!")
                # Auto-select the GLOBAL smallest board_id across ALL images —
                # not just whichever markers happen to be in the first file.
                all_marker_ids_global = {
                    d['marker_id'] for dets in all_detections.values() for d in dets
                }
                first_marker_id = min(all_marker_ids_global)
                first_filename = next(
                    fn for fn, dets in all_detections.items()
                    if any(d['marker_id'] == first_marker_id for d in dets)
                )
                print(f"[stage1] ✅ automatically select M{first_marker_id} as World origin (image: {first_filename})")
        else:
            # if self.first_marker_id is None, automatically select the
            # GLOBAL smallest board_id across ALL images.
            all_marker_ids_global = {
                d['marker_id'] for dets in all_detections.values() for d in dets
            }
            first_marker_id = min(all_marker_ids_global)
            first_filename = next(
                fn for fn, dets in all_detections.items()
                if any(d['marker_id'] == first_marker_id for d in dets)
            )
            print(f"[stage1] ✅ automatically select M{first_marker_id} as World origin (image: {first_filename})")

        first_marker_key = gtsam.symbol('M', first_marker_id)
        initial_estimate.insert(first_marker_key, gtsam.Pose3())

        # prior factor for the first marker — fix all 6 DOF at world origin with tight sigmas
        # Using Diagonal.Sigmas(1e-6) instead of Constrained.All(6) for numerical stability
        # with LevenbergMarquardtOptimizer (Constrained can cause ill-conditioned Hessian)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))

        graph.addPriorPose3(first_marker_key, gtsam.Pose3(), prior_noise)

        # build reverse lookup: camera_id -> [(marker_id, T_cam_to_marker, noise_model)]
        camera_to_observations = defaultdict(list)
        for marker_id, observations in marker_observations.items():
            for camera_id, T_cam_to_marker, noise_model in observations:
                camera_to_observations[camera_id].append((marker_id, T_cam_to_marker, noise_model))

        # ── Diagnostic: board count distribution per camera (before filtering) ──
        from collections import Counter as _Counter
        cam_board_counts = {cid: len(obs) for cid, obs in camera_to_observations.items()}
        dist = _Counter(cam_board_counts.values())
        print(f"\n  [diagnostic] total cameras: {len(cam_board_counts)}, "
              f"board count per image distribution: {dict(sorted(dist.items()))}")
        print(f"  [diagnostic] total unique boards detected: {sorted(marker_ids)}")
        if all(cnt == 1 for cnt in cam_board_counts.values()):
            print("  ⚠ NO image sees 2+ boards simultaneously!")
            print("    BFS cannot link boards without overlapping views.")
            print("    → Take images where 2+ boards are visible at once, OR")
            print("    → Set min_pair_obs to 1 in Uncertainty Settings to disable pair filtering.")

        # Filter: remove cameras that observe fewer than 2 markers
        single_marker_cameras = {cid for cid, obs in camera_to_observations.items() if len(obs) < 2}
        if single_marker_cameras:
            print(f"  ⚠ [stage1-bfs] removing {len(single_marker_cameras)}/{len(cam_board_counts)} "
                  f"cameras with <2 board observations")
            camera_ids -= single_marker_cameras
            for cid in single_marker_cameras:
                del camera_to_observations[cid]
            for marker_id in list(marker_observations.keys()):
                marker_observations[marker_id] = [
                    (cid, T, nm) for cid, T, nm in marker_observations[marker_id]
                    if cid in camera_ids
                ]
        print(f"  [stage1-bfs] {len(camera_ids)} cameras remain after 2-board filter")

        # ── PHASE 1 ───────────────────────────────────────────────────────
        #   Collect all marker-to-marker relative poses from ALL images
        #   (cam2mar → mar2mar conversion).
        #   Canonical direction: smaller-id → larger-id.
        marker_pair_poses   = defaultdict(list)
        marker_pair_cam_ids = defaultdict(list)
        for cam_id, obs_list in camera_to_observations.items():
            for i in range(len(obs_list)):
                mi, T_cam_mi, _ = obs_list[i]
                for j in range(i + 1, len(obs_list)):
                    mj, T_cam_mj, _ = obs_list[j]
                    T_mi_to_mj = T_cam_mi.inverse().compose(T_cam_mj)
                    lo, hi = (mi, mj) if mi < mj else (mj, mi)
                    if mi < mj:
                        marker_pair_poses[(lo, hi)].append(T_mi_to_mj)
                    else:
                        marker_pair_poses[(lo, hi)].append(T_mi_to_mj.inverse())
                    marker_pair_cam_ids[(lo, hi)].append(cam_id)

        pair_obs_counts = {pair: len(poses) for pair, poses in marker_pair_poses.items()}
        print(f"  [phase1] {len(marker_pair_poses)} unique marker pairs collected")
        if pair_obs_counts:
            print("    pairs & observation counts: "
                  + ", ".join(f"(M{a}↔M{b}):{cnt}"
                               for (a, b), cnt in sorted(pair_obs_counts.items())))
        else:
            print("  ⚠ [phase1] No marker pairs found → BFS cannot propagate beyond M"
                  f"{first_marker_id}.")
            print("    Possible causes:")
            print("    1) No image shows 2+ boards at once (see distribution above)")
            print(f"    2) min_pair_obs={self.min_pair_obs} too high for your dataset size")

        # ── Low-count pair filter: remove pairs with < min_pair_obs observations ──
        if self.min_pair_obs > 0:
            filter_low_count_pairs(
                marker_pair_poses, marker_pair_cam_ids,
                marker_observations, camera_to_observations,
                camera_ids, marker_ids,
                min_obs=self.min_pair_obs,
            )
            if first_marker_id not in marker_ids:
                print(f"  ⚠ [filter] world-origin marker M{first_marker_id} was "
                      f"removed by min_pair_obs — BFS cannot initialise anything "
                      f"from it; check the dataset or lower min_pair_obs")

        # ── PHASE 1.5 ─────────────────────────────────────────────────────
        #   Compute representative pose per pair using median.
        marker_pair_median = {}
        for pair, poses in marker_pair_poses.items():
            rep_pose = median_pose3(poses)
            if rep_pose is not None:
                marker_pair_median[pair] = rep_pose
        print(f"  [phase1] {len(marker_pair_median)} pairs with representative poses")

        # ── PHASE 2 ───────────────────────────────────────────────────────
        #   Propagate marker poses from origin (first_marker_id).
        #   T_world_Mj = T_world_Mi · T_Mi_to_Mj
        #   Use the first available initialized neighbor.
        initialized_markers = {first_marker_id}
        marker_init_source = {}  # {marker_id: src_marker_id}

        # Iterate until no new markers can be added
        for _pass in range(len(marker_ids)):
            newly_initialized = 0
            for marker_id in sorted(marker_ids):
                if marker_id in initialized_markers:
                    continue
                marker_key = gtsam.symbol('M', marker_id)

                # Use the first available initialized neighbor
                T_init = None
                src_marker = None
                for (mi, mj), T_mi_to_mj in marker_pair_median.items():
                    if mi == marker_id and mj in initialized_markers:
                        T_world_to_mi = initial_estimate.atPose3(gtsam.symbol('M', mj))
                        T_init = T_world_to_mi.compose(T_mi_to_mj.inverse())
                        src_marker = mj
                        break
                    elif mj == marker_id and mi in initialized_markers:
                        T_world_to_mi = initial_estimate.atPose3(gtsam.symbol('M', mi))
                        T_init = T_world_to_mi.compose(T_mi_to_mj)
                        src_marker = mi
                        break

                if T_init is not None:
                    initial_estimate.insert(marker_key, T_init)
                    initialized_markers.add(marker_id)
                    marker_init_source[marker_id] = src_marker
                    newly_initialized += 1

            if newly_initialized == 0:
                break

        print(f"  [phase2] {len(initialized_markers)}/{len(marker_ids)} markers initialized")

        # ── PHASE 3 ───────────────────────────────────────────────────────
        #   Initialize cameras from fully-initialized markers.
        #   Pick the marker with min trace(Σ) for each camera.
        initialized_cameras = set()
        camera_init_source  = {}  # {camera_id: marker_id}

        for camera_id in sorted(camera_ids):
            camera_key = gtsam.symbol('C', camera_id)
            candidates       = []
            noise_models_list = []
            used_markers_list = []

            for marker_id, T_cam_to_marker, noise_model in camera_to_observations[camera_id]:
                if marker_id not in initialized_markers:
                    continue
                marker_key_m = gtsam.symbol('M', marker_id)
                if initial_estimate.exists(marker_key_m):
                    T_world_to_marker = initial_estimate.atPose3(marker_key_m)
                    T_world_to_camera = T_world_to_marker.compose(T_cam_to_marker.inverse())
                    candidates.append(T_world_to_camera)
                    noise_models_list.append(noise_model)
                    used_markers_list.append(marker_id)

            if candidates:
                # A zero-information model has a NaN covariance trace and
                # np.argmin would pick it; treat it as worst, not best.
                traces    = [np.trace(nm.covariance()) for nm in noise_models_list]
                traces    = [t if np.isfinite(t) else np.inf for t in traces]
                best_idx  = int(np.argmin(traces))
                initial_estimate.insert(camera_key, candidates[best_idx])
                initialized_cameras.add(camera_id)
                camera_init_source[camera_id] = used_markers_list[best_idx]

        print(f"  [phase3] {len(initialized_cameras)}/{len(camera_ids)} cameras initialized")

        # check uninitialized nodes and warn
        uninitialized_markers = []
        for marker_id in marker_ids:
            marker_key = gtsam.symbol('M', marker_id)
            if not initial_estimate.exists(marker_key):
                uninitialized_markers.append(marker_id)

        uninitialized_cameras = []
        for camera_id in camera_ids:
            camera_key = gtsam.symbol('C', camera_id)
            if not initial_estimate.exists(camera_key):
                uninitialized_cameras.append(camera_id)

        if uninitialized_markers:
            print(f"⚠ warning: {len(uninitialized_markers)} markers are not initialized (connection issues): {uninitialized_markers}")
            print("  → factors for these markers are not added")
        if uninitialized_cameras:
            print(f"⚠ warning: {len(uninitialized_cameras)} cameras are not initialized (connection issues):")
            print("  → factors for these cameras are not added")

        # ── add factors (uncertainty-based) ───────────────────────────────
        for marker_id, observations in marker_observations.items():
            marker_key = gtsam.symbol('M', marker_id)

            if not initial_estimate.exists(marker_key):
                continue

            for camera_id, T_cam_to_marker, noise_model in observations:
                camera_key = gtsam.symbol('C', camera_id)

                if not initial_estimate.exists(camera_key):
                    continue

                # **core: use uncertainty-based BetweenFactor**
                graph.add(gtsam.BetweenFactorPose3(camera_key, marker_key, T_cam_to_marker, noise_model))

        initial_estimate = remove_disconnected_nodes(graph, initial_estimate)
        return graph, initial_estimate


    def optimize_poses(self, graph, initial_estimate):
        """Running uncertainty-based optimization"""
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity('ERROR')
        params.setMaxIterations(500)  # sufficient number of iterations
        params.setRelativeErrorTol(1e-9)
        params.setAbsoluteErrorTol(1e-9)  # absolute error tolerance

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()

        return result


    def show_marker_poses_table(self, result, marker_ids,
                                stage_label="Stage", reference_marker_id=None):
        """Print the 6DOF pose table (relative to the reference marker) and queue it for the GUI.

        The table data is appended to self.pending_pose_tables; the GUI
        (calibration_tool) opens it in a window on the main thread after the
        worker finishes. In headless runs only the stdout output remains.

        Args:
            result             : gtsam optimization result (Values)
            marker_ids         : list of marker IDs to display
            stage_label        : window title (e.g. "Stage 1 – Env Optimization")
            reference_marker_id: reference marker ID (None → first_marker_id or the minimum)
        """
        # ── determine reference marker ────────────────────────────────────
        if reference_marker_id is None:
            if self.first_marker_id is not None and self.first_marker_id in marker_ids:
                reference_marker_id = self.first_marker_id
            else:
                reference_marker_id = min(marker_ids) if marker_ids else None

        if reference_marker_id is None:
            print("  [pose-table] no reference marker available.")
            return

        ref_key = gtsam.symbol('M', reference_marker_id)
        if not result.exists(ref_key):
            print(f"  [pose-table] reference M{reference_marker_id} not in result.")
            return

        ref_pose_inv = result.atPose3(ref_key).inverse()

        # ── collect data ──────────────────────────────────────────────────
        rows_env   = []
        rows_wheel = []

        for mid in sorted(marker_ids):
            mkey = gtsam.symbol('M', mid)
            if not result.exists(mkey):
                continue
            rel = ref_pose_inv.compose(result.atPose3(mkey))
            t   = rel.translation()
            x, y, z = float(t[0]), float(t[1]), float(t[2])
            yaw, pitch, roll = Ro.from_matrix(
                rel.rotation().matrix()).as_euler('ZYX', degrees=True)
            entry = (mid, x, y, z, float(roll), float(pitch), float(yaw))
            if mid in self.vehicle_markers:
                rows_wheel.append(entry)
            else:
                rows_env.append(entry)

        title_str = f"=== {stage_label}  |  Reference: M{reference_marker_id} ==="

        # ── stdout output (also visible in the GUI log) ───────────────────
        COL = (f"{'ID':<10}{'TYPE':<8}{'X(m)':>10}{'Y(m)':>10}{'Z(m)':>10}"
               f"{'Roll(deg)':>12}{'Pitch(deg)':>12}{'Yaw(deg)':>12}")
        SEP = "-" * len(COL)

        def _fmt(entry, tag):
            mid, x, y, z, roll, pitch, yaw = entry
            t = tag + ("*" if mid == reference_marker_id else "")
            return (f"M{mid:<9}{t:<8}"
                    f"{x:>10.4f}{y:>10.4f}{z:>10.4f}"
                    f"{roll:>12.2f}{pitch:>12.2f}{yaw:>12.2f}")

        lines = [COL, SEP]
        for e in rows_env:
            lines.append(_fmt(e, "ENV"))
        if rows_wheel:
            lines.append(SEP)
            for e in rows_wheel:
                lines.append(_fmt(e, "WHEEL"))
        lines.append(SEP)

        print("\n" + "=" * len(title_str))
        print(title_str)
        print("=" * len(title_str))
        print("\n".join(lines))
        print()

        # ── queue data for GUI display ────────────────────────────────────
        # The window itself is created by calibration_tool on the Qt main
        # thread after the worker finishes. Previously a tkinter window was
        # spawned right here on a background thread, which window toolkits do
        # not support.
        val_lines = ["x(m)\ty(m)\tz(m)\troll(deg)\tpitch(deg)\tyaw(deg)"]
        for e in rows_env + ([None] if rows_wheel else []) + rows_wheel:
            if e is None:
                val_lines.append("")
                continue
            mid, x, y, z, roll, pitch, yaw = e
            val_lines.append(f"{x:.4f}\t{y:.4f}\t{z:.4f}\t"
                             f"{roll:.2f}\t{pitch:.2f}\t{yaw:.2f}")

        self.pending_pose_tables.append({
            'title': f"Marker Poses  –  {stage_label}",
            'pretty_text': "\n".join([title_str] + lines),
            'values_tsv': "\n".join(val_lines),
        })



    def visualize_poses_3d_optimized(self, optimized_result, marker_ids=None, save_path=None,
                                     plot_type=0):
        """Visualize the optimized poses in 3D space."""
        fig = MplFigure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # ── reference marker frame ────────────────────────────────────────
        # Express every pose in the reference marker frame via the inverse of
        # first_marker_id (or the smallest of marker_ids).
        ref_id = self.first_marker_id if (self.first_marker_id is not None and marker_ids and self.first_marker_id in marker_ids) \
                 else (min(marker_ids) if marker_ids else None)
        T_ref_inv = gtsam.Pose3()   # default = identity
        if ref_id is not None:
            ref_key = gtsam.symbol('M', ref_id)
            if optimized_result.exists(ref_key):
                T_ref_inv = optimized_result.atPose3(ref_key).inverse()

        # extract poses (transformed into the reference marker frame)
        marker_positions_opt = []
        marker_id_to_pos_opt = {}

        if marker_ids:
            for marker_id in marker_ids:
                key = gtsam.symbol('M', marker_id)
                if optimized_result.exists(key):
                    pose_world = optimized_result.atPose3(key)
                    pose_ref   = T_ref_inv.compose(pose_world)   # pose in the reference marker frame
                    pos        = pose_ref.translation()
                    position   = [pos[0], pos[1], pos[2]]
                    marker_positions_opt.append(position)
                    marker_id_to_pos_opt[marker_id] = position

        # plot
        if marker_positions_opt:
            marker_pos = np.array(marker_positions_opt)
            ax.scatter(marker_pos[:, 0], marker_pos[:, 1], marker_pos[:, 2],
                      c='red', s=120, marker='s', label='Markers', alpha=0.8)

            for marker_id in marker_ids:
                if marker_id in marker_id_to_pos_opt:
                    pos = marker_id_to_pos_opt[marker_id]
                    ax.text(pos[0], pos[1], pos[2], f'M{marker_id}', fontsize=16, fontweight='bold')

        # M2M connection lines (visualization only):
        # lines between markers observed in the same image (unique pairs only)
        if hasattr(self, 'all_detections_for_viz'):
            unique_pairs = set()
            for filename, detections in self.all_detections_for_viz.items():
                if len(detections) > 1:
                    ids = [d['marker_id'] for d in detections]
                    for i in range(len(ids)):
                        for j in range(i + 1, len(ids)):
                            pair = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                            unique_pairs.add(pair)
            for m1, m2 in unique_pairs:
                if m1 in marker_id_to_pos_opt and m2 in marker_id_to_pos_opt:
                    pos1, pos2 = marker_id_to_pos_opt[m1], marker_id_to_pos_opt[m2]
                    ax.plot3D([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                              'r-', alpha=0.7, linewidth=1)

        # Draw the vehicle coordinate system (only for plot_type=1).
        # (The plot_type=2 'V'-node visualization was removed — no code inserts
        #  V nodes, so it was an unreachable dead branch. The vehicle frame uses
        #  cached_vehicle_frame.)
        if plot_type == 1:
            self._draw_vehicle_coordinate_system(ax, marker_id_to_pos_opt)

        ax.set_xlabel('X (m)', fontsize=16)
        ax.set_ylabel('Y (m)', fontsize=16)
        ax.set_zlabel('Z (m)', fontsize=16)
        if plot_type==0:
            ax.set_title('Stage 1: optimized environment markers graph', fontsize=20, fontweight='bold')
        elif plot_type==1:
            ax.set_title('Optimized poses with vehicle camera', fontsize=20, fontweight='bold')
        ax.legend(fontsize=16, markerscale=1.3)
        ax.grid(True)

        # Front view of the marker frame.
        # Marker frame: X=right, Y=up, Z=camera direction (out of the wall).
        # elev=90, azim=-90 → looking straight down at the XY plane (marker wall) from +Z;
        # on screen X points right and Y points up.
        ax.view_init(elev=90, azim=-90)

        if save_path:
            # fig is not pyplot-managed — save through the figure itself.
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[viz] optimized 3D poses saved to: {save_path}")

        return fig

    def visualize_poses_3d_vehicle_centric(self, optimized_result, marker_ids=None, camera_ids=None, save_path=None):
        """
        Draw every marker node, camera position and edge relative to the vehicle frame.
        - Visualizes all connections around the vehicle frame (based on M29, M30, M31, M32)
        - Transforms every position into the vehicle frame for display
        - Shows Camera-Marker connections

        Returns:
            fig: matplotlib figure
            ax: matplotlib 3D axis
            vehicle_frame_data: dict - vehicle frame transform data
                'marker_positions': {marker_id: [x, y, z]} - marker positions in the vehicle frame
                'marker_rotations': {marker_id: gtsam.Rot3} - marker rotations in the vehicle frame
                'camera_positions': {camera_id: [x, y, z]} - camera positions in the vehicle frame
                'camera_rotations': {camera_id: gtsam.Rot3} - camera rotations in the vehicle frame
                'vehicle_center': np.array - vehicle frame origin (in world coordinates)
                'R_world_to_vehicle': np.array - world → vehicle transform matrix (3x3)
                'R_vehicle_to_world': np.array - vehicle → world transform matrix (3x3)
                'vehicle_axes': dict - vehicle frame axis vectors {'x_axis', 'y_axis', 'z_axis'}
        """
        print("\n[viz] vehicle-centric visualization")
        print(f"  received camera_ids: {camera_ids}")
        print(f"  number of cameras: {len(camera_ids) if camera_ids else 0}")

        fig = MplFigure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # ── extract vehicle frame ─────────────────────────────────────────
        # use the cached vehicle frame (precomputed by add_vehicle_coordinate_system_post_optimization)
        if self.cached_vehicle_frame is None:
            # no cache (e.g. viz-only load) — recompute the vehicle frame from the result
            self.add_vehicle_coordinate_system_post_optimization(optimized_result)
            if self.cached_vehicle_frame is None:
                # vehicle_markers missing from the result — fall back to the generic view
                fallback_fig = self.visualize_poses_3d_optimized(optimized_result, marker_ids, save_path, plot_type=1)
                return fallback_fig, None, None  # return 3 values for GUI compatibility

        print("[viz] ✅ using cached vehicle frame")
        vehicle_center = self.cached_vehicle_frame['center']
        R_vehicle_to_world = self.cached_vehicle_frame['R_vehicle_to_world']
        R_world_to_vehicle = self.cached_vehicle_frame['R_world_to_vehicle']
        x_axis = self.cached_vehicle_frame['axes']['x_axis']
        y_axis = self.cached_vehicle_frame['axes']['y_axis']
        z_axis = self.cached_vehicle_frame['axes']['z_axis']

        def transform_to_vehicle_frame(world_pos):
            """Transform world coordinates into the vehicle frame."""
            world_pos = np.array(world_pos)
            # translate to the origin, then rotate
            relative_pos = world_pos - vehicle_center
            vehicle_pos = R_world_to_vehicle @ relative_pos
            return vehicle_pos

        # extract and transform pose data
        marker_positions_opt = []
        marker_id_to_pos_opt = {}
        marker_id_to_rot_opt = {}
        camera_id_to_pos_opt = {}
        camera_id_to_rot_opt = {}

        # extract and transform marker data
        if marker_ids:
            for marker_id in marker_ids:
                key = gtsam.symbol('M', marker_id)
                if optimized_result.exists(key):
                    pose = optimized_result.atPose3(key)
                    world_pos = pose.translation()
                    world_rot = pose.rotation()
                    world_position = [world_pos[0], world_pos[1], world_pos[2]]

                    # transform into the vehicle frame
                    vehicle_position = transform_to_vehicle_frame(world_position)

                    # transform the rotation into the vehicle frame as well
                    world_rot_matrix = world_rot.matrix()
                    world_x_axis = world_rot_matrix[:, 0]
                    world_y_axis = world_rot_matrix[:, 1]
                    world_z_axis = world_rot_matrix[:, 2]

                    # transform each axis into the vehicle frame
                    vehicle_x_axis = R_world_to_vehicle @ world_x_axis
                    vehicle_y_axis = R_world_to_vehicle @ world_y_axis
                    vehicle_z_axis = R_world_to_vehicle @ world_z_axis

                    # assemble the new rotation matrix
                    vehicle_rot_matrix = np.column_stack([vehicle_x_axis, vehicle_y_axis, vehicle_z_axis])
                    vehicle_rot = gtsam.Rot3(vehicle_rot_matrix)

                    marker_positions_opt.append(vehicle_position)
                    marker_id_to_pos_opt[marker_id] = vehicle_position
                    marker_id_to_rot_opt[marker_id] = vehicle_rot

        # ── extract and transform vehicle camera data ─────────────────────
        if camera_ids:
            for camera_id in camera_ids:
                key = gtsam.symbol('C', camera_id)
                if optimized_result.exists(key):
                    pose = optimized_result.atPose3(key)
                    world_pos = pose.translation()
                    world_rot = pose.rotation()
                    world_position = [world_pos[0], world_pos[1], world_pos[2]]

                    vehicle_position = transform_to_vehicle_frame(world_position)

                    world_rot_matrix = world_rot.matrix()
                    vehicle_x_axis = R_world_to_vehicle @ world_rot_matrix[:, 0]
                    vehicle_y_axis = R_world_to_vehicle @ world_rot_matrix[:, 1]
                    vehicle_z_axis = R_world_to_vehicle @ world_rot_matrix[:, 2]
                    vehicle_rot_matrix = np.column_stack([vehicle_x_axis, vehicle_y_axis, vehicle_z_axis])
                    vehicle_rot = gtsam.Rot3(vehicle_rot_matrix)

                    camera_id_to_pos_opt[camera_id] = vehicle_position
                    camera_id_to_rot_opt[camera_id] = vehicle_rot

        # ── draw vehicle camera FOV pyramids ──────────────────────────────
        if camera_id_to_pos_opt:
            self._draw_camera_fov_pyramids(ax, camera_id_to_pos_opt, camera_id_to_rot_opt,
                                           list(camera_id_to_pos_opt.keys()))

        # ── draw the vehicle frame axes (at the origin) ───────────────────
        axis_length = 0.5
        # X axis (front, red)
        ax.plot3D([0, axis_length], [0, 0], [0, 0], 'r-', linewidth=4, alpha=0.8)
        ax.text(axis_length + 0.1, 0, 0, 'X (Front)', fontsize=12, fontweight='bold', color='red')

        # Y axis (left, green)
        ax.plot3D([0, 0], [0, axis_length], [0, 0], 'g-', linewidth=4, alpha=0.8)
        ax.text(0, axis_length + 0.1, 0, 'Y (Left)', fontsize=12, fontweight='bold', color='green')

        # Z axis (up, blue)
        ax.plot3D([0, 0], [0, 0], [0, axis_length], 'b-', linewidth=4, alpha=0.8)
        ax.text(0, 0, axis_length + 0.1, 'Z (Up)', fontsize=12, fontweight='bold', color='blue')

        # ── draw marker nodes (wheel markers blue, env markers red) ───────
        if marker_positions_opt:
            wheel_set = set(self.vehicle_markers)
            env_ids_viz   = [mid for mid in marker_ids if mid in marker_id_to_pos_opt and mid not in wheel_set]
            wheel_ids_viz = [mid for mid in marker_ids if mid in marker_id_to_pos_opt and mid in wheel_set]

            if env_ids_viz:
                env_pos = np.array([marker_id_to_pos_opt[mid] for mid in env_ids_viz])
                ax.scatter(env_pos[:, 0], env_pos[:, 1], env_pos[:, 2],
                           c='red', s=120, marker='s', label='Env Markers', alpha=0.8)

            if wheel_ids_viz:
                wheel_pos = np.array([marker_id_to_pos_opt[mid] for mid in wheel_ids_viz])
                ax.scatter(wheel_pos[:, 0], wheel_pos[:, 1], wheel_pos[:, 2],
                           c='blue', s=150, marker='s', label='Wheel Markers', alpha=0.9)

            # add marker labels
            for marker_id in marker_ids:
                if marker_id in marker_id_to_pos_opt:
                    pos = marker_id_to_pos_opt[marker_id]
                    color = 'blue' if marker_id in wheel_set else 'red'
                    ax.text(pos[0], pos[1], pos[2], f'M{marker_id}',
                            fontsize=16, fontweight='bold', color=color)

        # ── Vehicle camera → Marker edges (optimized positions) ───────────
        if camera_id_to_pos_opt and hasattr(self, 'all_detections_for_viz'):
            cam_marker_legend_added = False
            # Collect unique (camera_id, marker_id) pairs from detections
            cam_marker_pairs = set()
            for filename, detections in self.all_detections_for_viz.items():
                for detection in detections:
                    m_id = detection.get('marker_id')
                    c_id = detection.get('camera_id')
                    if c_id is not None and m_id is not None:
                        cam_marker_pairs.add((c_id, m_id))

            for (c_id, m_id) in cam_marker_pairs:
                if c_id not in camera_id_to_pos_opt or m_id not in marker_id_to_pos_opt:
                    continue
                cp = camera_id_to_pos_opt[c_id]
                mp = marker_id_to_pos_opt[m_id]
                lbl = 'Camera-marker edge' if not cam_marker_legend_added else ""
                ax.plot3D([cp[0], mp[0]], [cp[1], mp[1]], [cp[2], mp[2]],
                          color='salmon', alpha=0.7, linewidth=1.0, label=lbl)
                if lbl:
                    cam_marker_legend_added = True

        # ── draw vehicle structure connections (highlighted in the vehicle frame) ──
        # connect the vehicle's 4 corner markers (near the origin in the vehicle frame)
        # front-left, rear-left, front-right, rear-right
        if all(mid in marker_id_to_pos_opt for mid in self.vehicle_markers):
            # draw the vehicle outline
            vehicle_edges = [
                (self.vehicle_markers[0], self.vehicle_markers[3]),  # front axle: left→right
                (self.vehicle_markers[1], self.vehicle_markers[2]),  # rear axle: left→right
                (self.vehicle_markers[0], self.vehicle_markers[1]),  # left side: front→rear
                (self.vehicle_markers[2], self.vehicle_markers[3]),  # right side: front→rear
            ]

            for mid1, mid2 in vehicle_edges:
                pos1 = marker_id_to_pos_opt[mid1]
                pos2 = marker_id_to_pos_opt[mid2]
                ax.plot3D([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                         'black', alpha=0.8, linewidth=1,
                         label='Vehicle body Frame' if mid1 == self.vehicle_markers[0] and mid2 == self.vehicle_markers[2] else "")

        # ── plot settings ─────────────────────────────────────────────────
        ax.set_xlabel('X (Front) [m]', fontsize=16)
        ax.set_ylabel('Y (Left) [m]', fontsize=16)
        ax.set_zlabel('Z (Up) [m]', fontsize=16)
        ax.set_title('Vehicle-centric vehicle camera pose',
                    fontsize=20, fontweight='bold')

        # legend
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=14, markerscale=1.3)
        ax.grid(True, alpha=0.3)

        # view angle (looking forward from behind the vehicle)
        ax.view_init(elev=15, azim=0)

        # adjust axis ranges (centered on the vehicle)
        all_positions = []
        for pos in marker_id_to_pos_opt.values():
            all_positions.append(pos)
        for pos in camera_id_to_pos_opt.values():
            all_positions.append(pos)

        if all_positions:
            all_positions = np.array(all_positions)
            margin = 1.0  # one meter of headroom

            x_min, x_max = np.min(all_positions[:, 0]) - margin, np.max(all_positions[:, 0]) + margin
            y_min, y_max = np.min(all_positions[:, 1]) - margin, np.max(all_positions[:, 1]) + margin
            z_min, z_max = np.min(all_positions[:, 2]) - margin, np.max(all_positions[:, 2]) + margin

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

        if save_path:
            # fig is not pyplot-managed — save through the figure itself.
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[viz] vehicle-centric visualization saved to: {save_path}")

        # relative positions of the vehicle markers (vehicle frame)
        for mid in self.vehicle_markers:
            key = gtsam.symbol('M', mid)
            if optimized_result.exists(key):
                pose = optimized_result.atPose3(key)
                world_pos = pose.translation()
                world_position = [world_pos[0], world_pos[1], world_pos[2]]

                # transform into the vehicle frame
                vehicle_position = transform_to_vehicle_frame(world_position)
                print(f"  M{mid}: [{vehicle_position[0]:.3f}, {vehicle_position[1]:.3f}, {vehicle_position[2]:.3f}]")

        # debug: print camera positions and orientations
        if camera_id_to_pos_opt:
            print("\n" + "=" * 60)
            print("camera poses in the vehicle frame")
            print("  axes: X=Front, Y=Left, Z=Up")
            print("=" * 60)
            for cid in sorted(camera_id_to_pos_opt.keys()):
                pos = camera_id_to_pos_opt[cid]
                if cid in camera_id_to_rot_opt:
                    rot_matrix = camera_id_to_rot_opt[cid].matrix()

                    # Euler angles (ZYX convention)
                    r = Ro.from_matrix(rot_matrix)
                    yaw, pitch, roll = r.as_euler('ZYX', degrees=True)

                    # camera viewing direction (Z axis vector)
                    camera_forward = rot_matrix[:, 2]

                    print(f"\nC{cid}:")
                    print(f"  position (m): X={pos[0]:7.4f} (front), Y={pos[1]:7.4f} (left), Z={pos[2]:7.4f} (up)")
                    print(f"  attitude (deg): Roll={roll:7.2f}°, Pitch={pitch:7.2f}°, Yaw={yaw:7.2f}°")
                    print(f"  forward vector: [{camera_forward[0]:.4f}, {camera_forward[1]:.4f}, {camera_forward[2]:.4f}]")
                else:
                    print(f"\nC{cid}:")
                    print(f"  position (m): X={pos[0]:7.4f} (front), Y={pos[1]:7.4f} (left), Z={pos[2]:7.4f} (up)")
                    print("  attitude: N/A")
            print("=" * 60 + "\n")

        # build the vehicle-frame transform result dictionary
        vehicle_frame_data = {
            'marker_positions': marker_id_to_pos_opt,      # {marker_id: [x, y, z]}
            'marker_rotations': marker_id_to_rot_opt,      # {marker_id: gtsam.Rot3}
            'camera_positions': camera_id_to_pos_opt,      # {camera_id: [x, y, z]}
            'camera_rotations': camera_id_to_rot_opt,      # {camera_id: gtsam.Rot3}
            'vehicle_center': vehicle_center,              # vehicle frame origin (world coordinates)
            'R_world_to_vehicle': R_world_to_vehicle,      # world → vehicle transform matrix
            'R_vehicle_to_world': R_vehicle_to_world,      # vehicle → world transform matrix
            'vehicle_axes': {                              # vehicle frame axis vectors
                'x_axis': x_axis,  # front
                'y_axis': y_axis,  # left
                'z_axis': z_axis   # up
            }
        }

        return fig, ax, vehicle_frame_data


    def run_uncertainty_optimization(self):
        """
        Run the full uncertainty-based optimization pipeline.

        Returns:
            tuple: (graph, result, all_detections, uncertainty_info)
        """
        print("=== Uncertainty-based Factor Graph Optimization ===")

        # 1. detect markers and compute uncertainty in every image
        image_files = self.get_image_files()
        all_detections = {}
        uncertainty_info = {}

        print(f"\n[stage1] processing {len(image_files)} images...")

        for i, image_path in enumerate(image_files):
            filename = os.path.basename(image_path)

            detections = self.detect_markers_with_uncertainty(image_path, dist_coeffs=None)
            if detections:
                all_detections[filename] = detections

                # store uncertainty info
                for detection in detections:
                    marker_id = detection['marker_id']
                    if marker_id not in uncertainty_info:
                        uncertainty_info[marker_id] = []

                    if detection['uncertainty_matrix'] is not None:
                        # compute uncertainty statistics
                        cov_matrix = detection['uncertainty_matrix']
                        pos_uncertainty = np.sqrt(np.diag(cov_matrix[:3, :3]))  # [σx, σy, σz]
                        rot_uncertainty = np.sqrt(np.diag(cov_matrix[3:, 3:]))  # [σroll, σpitch, σyaw]

                        uncertainty_info[marker_id].append({
                            'filename': filename,
                            'covariance_matrix': cov_matrix,
                            'position_std': pos_uncertainty,
                            'rotation_std': rot_uncertainty
                        })

                marker_count = len(detections)
                board_ids = [d['marker_id'] for d in detections]
                tag_ids = [tid for d in detections for tid in d.get('tag_ids', [])]
                print(f"  [{i+1}/{len(image_files)}] {filename}: {marker_count} board(s) {board_ids}  (tag IDs: {tag_ids})")
            else:
                print(f"  [{i+1}/{len(image_files)}] {filename}: no markers detected")

        # Detection summary
        all_board_ids = sorted({d['marker_id'] for dets in all_detections.values() for d in dets})
        all_tag_ids   = sorted({tid for dets in all_detections.values() for d in dets for tid in d.get('tag_ids', [])})
        print("\n[stage1] detection summary:")
        print(f"  Images with detections : {len(all_detections)} / {len(image_files)}")
        if self._detect_error_count:
            print(f"  ⚠ Images that FAILED with an error (not just markerless): "
                  f"{self._detect_error_count} — see 'marker detect fail' lines above")
            self._detect_error_count = 0
        print(f"  Unique board IDs found : {all_board_ids}")
        print(f"  Underlying tag IDs     : {all_tag_ids}")
        cfg = self.apriltag_config
        env_n_m   = cfg['env_grid_rows'] * cfg['env_grid_cols']
        env_first = cfg.get('env_first_tag_id', 10)
        print(f"  Grid (env)             : {cfg['env_grid_rows']}×{cfg['env_grid_cols']} "
              f"({env_n_m} tags/board, first_tag_id={env_first})"
              f" → board_id = min(tag_ids) = {env_first}, {env_first+env_n_m}, {env_first+2*env_n_m}, ...")

        if not all_detections:
            print("\n[stage1] ⚠ no markers detected in any image!")
            print(f"  Image directory: '{self.image_directory}'")
            image_files_check = self.get_image_files()
            print(f"  Images found: {len(image_files_check)}")
            if image_files_check:
                print(f"  First image: {image_files_check[0]}")
            return None, None, None, None, None

        kept = sum(len(v) for v in all_detections.values())

        if self._cov_failed_detections:
            total = kept + self._cov_failed_detections
            print(f"  ⚠ {self._cov_failed_detections}/{total} observations "
                  f"({100.0 * self._cov_failed_detections / total:.2f}%) dropped: MARSCOT "
                  f"could not compute their covariance. A few is expected; a large "
                  f"share points at abnormally large corner covariances upstream.")

        # 3. Factor graph creation
        print("\n[stage1] creating factor graph (BFS)...")
        graph, initial_estimate = self.create_uncertainty_factor_graph(all_detections, image_files)

        print(f"  graph size: {graph.size()} factors")
        print(f"  variables: {initial_estimate.size()}")

        # 4. run optimization
        print("\n[stage1] running optimization...")
        try:
            result = self.optimize_poses(graph, initial_estimate)
        except Exception as e:
            import traceback
            print(f"[stage1] optimization failed: {e}")
            print(traceback.format_exc())
            return None, None, None, None, None

        print("[stage1] optimization completed successfully!")

        # 5. extract results
        all_marker_ids_set = set()

        for detections in all_detections.values():
            for detection in detections:
                all_marker_ids_set.add(detection['marker_id'])

        all_marker_ids = list(all_marker_ids_set)

        # save data for M2M visualization
        self.all_detections_for_viz = all_detections

        # 6. Creating Stage 1 Visualizations (Base Dataset: tmp_img)
        if self.enable_visualization:
            print("\n[stage1] creating Stage 1 visualizations (base dataset)...")
            viz_dir = os.path.join(g_current_dir, "visualization_results")

            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
                print(f"  created directory: {viz_dir}")

            # Stage 1: Creating Base Dataset Visualizations (Markers + Camera FOV)
            try:
                self.visualize_poses_3d_optimized(
                    result, all_marker_ids,
                    save_path=os.path.join(viz_dir, "stage1_base_dataset.png")
                )
                print("[stage1] visualization completed!")
            except Exception as e:
                print(f"[stage1] visualization failed (non-critical): {e}")

        print("\n" + "=" * 60)
        print("  [stage1] optimization summary")
        print(f"  - result size: {result.size()} variables")
        print("=" * 60 + "\n")

        # ── Stage 1 marker pose table window ──────────────────────────────
        if self.enable_visualization:
            try:
                self.show_marker_poses_table(
                    result, all_marker_ids,
                    stage_label="Stage 1 – Env Marker Optimization"
                )
            except Exception as _e:
                print(f"  [pose-table] stage 1 failed: {_e}")

        return graph, initial_estimate, result, all_detections, uncertainty_info



    def create_marker_corner_pointcloud_from_result(self, result, marker_ids):
        """
        Build the corner point cloud of every marker (relative to M1) from an FGO result.

        Args:
            result: GTSAM optimization result
            marker_ids: list of marker IDs

        Returns:
            dict: {
                'corner_points': np.array,  # all corner points (N x 3)
                'marker_info': list,        # marker info for each corner point
                'marker_corners': dict,     # full per-board corners {marker_id: corners_(4*tags)x3}
                'marker_poses': dict        # per-board poses {marker_id: gtsam.Pose3}
            }
        """
        all_corner_points = []
        corner_marker_info = []  # which marker and which corner each point belongs to
        marker_corners_dict = {}
        marker_poses_dict = {}

        print(f"Creating corner point cloud for {len(marker_ids)} markers...")

        for marker_id in marker_ids:
            marker_key = gtsam.symbol('M', marker_id)
            if not result.exists(marker_key):
                print(f"⚠ Marker {marker_id} not found in result, skipping...")
                continue

            # extract the marker pose (world frame, M1-based)
            marker_pose = result.atPose3(marker_key)
            marker_rotation = marker_pose.rotation().matrix()
            marker_translation = marker_pose.translation()

            current_marker_size = get_marker_size(
                marker_id,
                self.marker_size,
                self.marker_size2,
                self.vehicle_markers,
                self.marker_size_mapping
            )

            # Corner points in the board's local frame (z=0 plane). When several
            # tags are printed on the board, corners for every tag slot are
            # generated relative to the board center.
            corners_local = self.board_object_points(marker_id)

            # transform into the world frame (M1-based)
            corners_world = []
            for corner_idx, corner_local in enumerate(corners_local):
                # rotate, then translate: P_world = R * P_local + t
                corner_world = marker_rotation @ corner_local + marker_translation
                corners_world.append(corner_world)
                all_corner_points.append(corner_world)

                # store info for each corner point
                corner_marker_info.append({
                    'marker_id': marker_id,
                    'tag_index': corner_idx // 4,       # tag slot number on the board
                    'corner_index': corner_idx % 4,     # 0: BR, 1: BL, 2: TL, 3: TR
                    'marker_size': current_marker_size,
                    'local_coords': corner_local.copy(),
                    'world_coords': corner_world.copy()
                })

            # store per-marker corners
            marker_corners_dict[marker_id] = np.array(corners_world)
            marker_poses_dict[marker_id] = marker_pose

        # stack all corner points into one numpy array
        all_corner_points = np.array(all_corner_points)

        return {
            'corner_points': all_corner_points,
            'marker_info': corner_marker_info,
            'marker_corners': marker_corners_dict,
            'marker_poses': marker_poses_dict
        }

    def verify_corner_order_consistency(self, detected_corners, projected_corners):
        """
        Check whether detected and projected corners are in the same order (debugging aid).

        Args:
            detected_corners: detected marker corner coordinates (Nx2)
            projected_corners: projected marker corner coordinates (Nx2)
        """
        # geometric consistency check
        det_center = np.mean(detected_corners, axis=0)
        proj_center = np.mean(projected_corners, axis=0)

        # compare each corner's relative direction
        consistency_checks = []
        for i in range(len(detected_corners)):
            det_rel = detected_corners[i] - det_center
            proj_rel = projected_corners[i] - proj_center

            # angle difference between direction vectors
            det_angle = np.arctan2(det_rel[1], det_rel[0])
            proj_angle = np.arctan2(proj_rel[1], proj_rel[0])
            angle_diff = abs(det_angle - proj_angle)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # minimal angle difference

            consistency_checks.append(angle_diff < np.pi/8)  # within 22.5 degrees

        return all(consistency_checks)

    def compute_reprojection_errors_with_corner_pointcloud(self, incremental_result, incremental_detections,
                                                         corner_pointcloud, camera_params_mapping):
        """
        Reproject the GT corner point cloud into the new cameras (incremental_result) and compute errors.
        Corners are matched by exact order before comparison.

        Args:
            incremental_result: incremental optimization result (includes new camera poses)
            incremental_detections: marker detections of the new cameras {filename: detections} (uses the 'corners' key)
            corner_pointcloud: result of create_marker_corner_pointcloud_from_result() (GT)
            camera_params_mapping: per-camera-ID parameter mapping {camera_id: {'intrinsic': [fx,fy,cx,cy], 'dist_coeffs': [k1,k2]}}

        Returns:
            dict: reprojection error info (including per-corner details)
        """
        camera_reprojection_errors = {}
        total_errors = []
        gt_marker_corners = corner_pointcloud['marker_corners']
        gt_marker_poses = corner_pointcloud.get('marker_poses', {})

        print("Computing reprojection errors using GT corner pointcloud...")
        print(f"GT markers available: {list(gt_marker_corners.keys())}")
        print(f"Available detection files: {list(incremental_detections.keys())}")

        # build camera_id -> filename mapping
        camera_id_to_filename = {}
        for filename, detections in incremental_detections.items():
            if detections and 'camera_id' in detections[0]:
                camera_id = detections[0]['camera_id']
                camera_id_to_filename[camera_id] = filename
                print(f"  ✓ Mapping from detection: Camera {camera_id} → {filename}")
            else:
                print(f"  ⚠ No camera_id in detection for {filename}")

        # only process cameras present in camera_params_mapping (vehicle cameras)
        available_camera_ids = sorted(camera_params_mapping.keys())

        if len(camera_id_to_filename) != len(available_camera_ids):
            print("⚠ Camera-Filename mapping incomplete!")
            print(f"  Available camera IDs in params_mapping: {available_camera_ids}")
            print(f"  Mapped filenames: {camera_id_to_filename}")

        # process each new camera
        for camera_id in available_camera_ids:
            if camera_id not in camera_id_to_filename:
                print(f"⚠ No detection file for camera {camera_id}, skipping...")
                continue

            filename = camera_id_to_filename[camera_id]
            detections = incremental_detections[filename]

            # extract the new camera pose (from incremental_result)
            camera_key = gtsam.symbol('C', camera_id)
            if not incremental_result.exists(camera_key):
                print(f"⚠ Camera {camera_id} not found in incremental_result, skipping...")
                continue

            # camera pose in world coordinates (= camera-to-world)
            camera_pose = incremental_result.atPose3(camera_key)

            # cv2.projectPoints needs world-to-camera, so invert
            camera_pose_inv = camera_pose.inverse()  # world-to-camera
            R_cam = camera_pose_inv.rotation().matrix()
            t_cam = camera_pose_inv.translation()

            camera_errors = []
            detection_results = []

            for detection in detections:
                marker_id = detection['marker_id']

                # extract corner data (unified key)
                detected_corners = np.array(detection['corners'])  # detected corners [Nx2]

                if marker_id not in gt_marker_corners:
                    print(f"  ⚠ Marker {marker_id} not found in GT corners, skipping...")
                    continue

                # GT 3D corners (world frame, M1-based).
                # When only part of a board's tags is visible, the full board's
                # corner count differs from the detected count, so map the board
                # coordinates matching the detected corners through the optimized
                # board pose to pair them 1:1.
                board_corners_3d = detection.get('corners_3d', None)
                board_pose = gt_marker_poses.get(marker_id, None)
                if board_corners_3d is not None and board_pose is not None:
                    R_board = board_pose.rotation().matrix()
                    t_board = board_pose.translation()
                    gt_corners_3d = np.array(board_corners_3d) @ R_board.T + t_board
                else:
                    gt_corners_3d = gt_marker_corners[marker_id]

                if len(gt_corners_3d) != len(detected_corners):
                    print(f"  ⚠ Marker {marker_id} corner count mismatch "
                          f"(detected {len(detected_corners)}, GT {len(gt_corners_3d)}), skipping...")
                    continue

                # fetch this camera's parameters
                if camera_id in camera_params_mapping:
                    camera_config = camera_params_mapping[camera_id]
                    current_camera_params = camera_config['intrinsic']
                    current_dist_coeffs   = camera_config['dist_coeffs']
                    current_camera_model  = camera_config.get('camera_model', 'pinhole')
                else:
                    print(f"⚠ Camera {camera_id} not found in camera_params_mapping, skipping...")
                    continue

                # build the camera matrix
                camera_matrix = np.array([
                    [current_camera_params[0], 0, current_camera_params[2]],
                    [0, current_camera_params[1], current_camera_params[3]],
                    [0, 0, 1]
                ])

                # convert distortion coefficients
                if current_camera_model == 'fisheye':
                    # Fisheye KB: 4 values [k1,k2,k3,k4], reshape(-1,1) required
                    dist_coeffs = np.array(current_dist_coeffs, dtype=np.float64).reshape(-1, 1)
                elif len(current_dist_coeffs) == 2:
                    dist_coeffs = np.array([current_dist_coeffs[0], current_dist_coeffs[1], 0.0, 0.0, 0.0])
                else:
                    dist_coeffs = np.array(current_dist_coeffs)

                # rvec, tvec for cv2.projectPoints (world-to-camera)
                rvec, _ = cv2.Rodrigues(R_cam)
                tvec = t_cam.reshape(3, 1)

                # project the GT 3D corners into image coordinates (branch by model)
                if current_camera_model == 'fisheye':
                    # Fisheye KB: use cv2.fisheye.projectPoints
                    projected_cv2, _ = cv2.fisheye.projectPoints(
                        gt_corners_3d.reshape(-1, 1, 3).astype(np.float64),
                        rvec, tvec, camera_matrix, dist_coeffs
                    )
                else:
                    # Pinhole: use cv2.projectPoints
                    projected_cv2, _ = cv2.projectPoints(
                        gt_corners_3d.reshape(-1, 1, 3).astype(np.float64),
                        rvec, tvec, camera_matrix, dist_coeffs
                    )
                projected_corners_dist = projected_cv2.reshape(-1, 2)

                # depth validity check (filter points behind the camera)
                valid_projections = []
                for corner_world in gt_corners_3d:
                    corner_cam = R_cam @ corner_world + t_cam
                    valid_projections.append(corner_cam[2] > 0.1)

                # corner order check (first marker only) - uses distorted coordinates
                if marker_id == list(gt_marker_corners.keys())[0]:
                    self.verify_corner_order_consistency(detected_corners, projected_corners_dist)

                # per-corner reprojection errors (in distorted image coordinates)
                corner_errors = []
                corner_details = []

                for corner_idx in range(len(gt_corners_3d)):
                    if valid_projections[corner_idx]:
                        # compare distorted image coordinates directly
                        detected_corner = detected_corners[corner_idx]
                        projected_corner = projected_corners_dist[corner_idx]

                        error = np.linalg.norm(detected_corner - projected_corner)
                        corner_errors.append(error)

                        corner_details.append({
                            'corner_index': corner_idx % 4,
                            'tag_index': corner_idx // 4,
                            'detected': detected_corner.copy(),    # original distorted coordinates
                            'projected': projected_corner.copy(),  # projected coordinates with distortion applied
                            'error': error,
                            'gt_3d': gt_corners_3d[corner_idx].copy(),
                            'valid': True
                        })

                        # extra debugging for large errors
                        if error > 20:
                            corner_cam = R_cam @ gt_corners_3d[corner_idx] + t_cam
                            print("      ⚠ Large error detected! Checking possible causes:")
                            print(f"        GT 3D (world): {gt_corners_3d[corner_idx]}")
                            print(f"        Camera coordinates: {corner_cam}")
                            print(f"        Detected vs Projected difference: "
                                  f"[{detected_corner[0]-projected_corner[0]:.4f}, "
                                  f"{detected_corner[1]-projected_corner[1]:.4f}]")
                    else:
                        corner_cam = R_cam @ gt_corners_3d[corner_idx] + t_cam
                        print(f"    ⚠ Corner {corner_idx} behind camera (z={corner_cam[2]:.3f})")
                        corner_details.append({
                            'corner_index': corner_idx % 4,
                            'tag_index': corner_idx // 4,
                            'detected': detected_corners[corner_idx].copy(),
                            'projected': [0, 0],
                            'error': float('inf'),
                            'gt_3d': gt_corners_3d[corner_idx].copy(),
                            'valid': False
                        })

                if corner_errors:  # only when valid corners exist
                    mean_error = np.mean(corner_errors)
                    max_error = np.max(corner_errors)

                    camera_errors.append(mean_error)
                    total_errors.append(mean_error)

                    detection_results.append({
                        'marker_id': marker_id,
                        'mean_error': mean_error,
                        'max_error': max_error,
                        'corner_details': corner_details,
                        'valid_corners': sum(valid_projections)
                    })
                else:
                    print(f"    → No valid projections for marker {marker_id}")

            camera_reprojection_errors[camera_id] = {
                'filename': filename,
                'detections': detection_results,
                'camera_mean_error': np.mean(camera_errors) if camera_errors else 0,
                'camera_max_error': np.max(camera_errors) if camera_errors else 0,
                'camera_pose': {'R': R_cam, 't': t_cam}
            }

        # overall statistics
        overall_stats = {
            'total_mean_error': np.mean(total_errors) if total_errors else 0,
            'total_std_error': np.std(total_errors) if total_errors else 0,
            'total_max_error': np.max(total_errors) if total_errors else 0,
            'total_min_error': np.min(total_errors) if total_errors else 0,
            'num_observations': len(total_errors)
        }

        return {
            'camera_errors': camera_reprojection_errors,
            'overall_stats': overall_stats
        }

    def visualize_corner_pointcloud_3d(self, corner_pointcloud, title="Marker Corner Point Cloud", save_path=None):
        """
        Visualize a create_marker_corner_pointcloud_from_result() result in 3D (matplotlib).

        Args:
            corner_pointcloud: return value of create_marker_corner_pointcloud_from_result()
                               {'corner_points': Nx3, 'marker_info': list, 'marker_corners': dict}
            title: window title
            save_path: save path (None returns the Figure)

        Returns:
            matplotlib.figure.Figure (when save_path is None)
        """
        all_pts   = corner_pointcloud['corner_points']      # (N, 3)
        mk_corners = corner_pointcloud['marker_corners']    # {marker_id: (4*tags, 3)}
        mk_info   = corner_pointcloud['marker_info']        # list of dicts

        if len(all_pts) == 0:
            print("[viz] ⚠ no corner points to visualize.")
            return None

        # Horizontal alignment for display: the map is in the M10 board frame, so
        # drawing it as-is comes out tilted and hard to read. Level it against the
        # plane fitted through the environment marker centers (estimates unchanged).
        marker_ids = sorted(mk_corners.keys())
        env_centers = [np.asarray(mk_corners[m]).mean(axis=0)
                       for m in marker_ids if m not in self.vehicle_markers]
        if len(env_centers) < 3:
            env_centers = [np.asarray(mk_corners[m]).mean(axis=0) for m in marker_ids]
        ref_mid = (self.first_marker_id
                   if self.first_marker_id in mk_corners else min(marker_ids))
        R_disp, _ = level_display_frame(
            np.array(env_centers),
            x_toward=np.asarray(mk_corners[ref_mid]).mean(axis=0))
        # The origin stays at the world-origin marker — same convention as the
        # optimization frame (M10 = 0,0,0) and the other plots. The horizontal
        # alignment only contributes rotation.
        c_disp = np.asarray(mk_corners[ref_mid], dtype=float).mean(axis=0)
        mk_corners = {m: (np.asarray(v, dtype=float) - c_disp) @ R_disp.T
                      for m, v in mk_corners.items()}
        # keep the console range printout in the same display frame as the plot
        all_pts = (np.asarray(all_pts, dtype=float) - c_disp) @ R_disp.T

        fig = MplFigure(figsize=(14, 10))
        ax  = fig.add_subplot(111, projection='3d')

        # unique color per marker (plt.cm.get_cmap is removed in matplotlib 3.11)
        from matplotlib import colormaps as mpl_colormaps
        cmap = mpl_colormaps['tab20'].resampled(max(len(marker_ids), 1))

        for color_idx, mid in enumerate(marker_ids):
            corners = mk_corners[mid]           # (4*tags, 3)
            color   = cmap(color_idx)

            # corner scatter
            ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2],
                       color=color, s=60, zorder=5)

            # draw a separate square outline for each tag on the board
            for tag_start in range(0, len(corners), 4):
                tag_corners = corners[tag_start:tag_start + 4]
                loop = np.vstack([tag_corners, tag_corners[0]])
                ax.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                        color=color, linewidth=1.2, alpha=0.8)

            # ID label at the board center
            center = corners.mean(axis=0)
            ax.text(center[0], center[1], center[2],
                    f'M{mid}', fontsize=7, color=color,
                    ha='center', va='bottom')

        # corner index labels (0~3)
        corner_names = ['BR', 'BL', 'TL', 'TR']
        for info in mk_info:
            pt = R_disp @ (np.asarray(info['world_coords'], dtype=float) - c_disp)
            ci = info['corner_index']
            ax.text(pt[0], pt[1], pt[2],
                    corner_names[ci], fontsize=5, color='gray', alpha=0.6)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"{title}\n{len(all_pts)} corner points from {len(marker_ids)} markers"
                     f"\n(axes: display frame, marker plane levelled)", fontsize=10)

        # legend (up to 20 entries)
        legend_patches = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cmap(i), markersize=7, label=f'M{mid}')
            for i, mid in enumerate(marker_ids[:20])
        ]
        ax.legend(handles=legend_patches, loc='upper left',
                  fontsize=7, ncol=2, framealpha=0.7)

        fig.tight_layout()

        print(f"  Total: {len(all_pts)} corners from {len(marker_ids)} markers")
        print(f"  X: [{all_pts[:,0].min():.3f}, {all_pts[:,0].max():.3f}]  "
              f"Y: [{all_pts[:,1].min():.3f}, {all_pts[:,1].max():.3f}]  "
              f"Z: [{all_pts[:,2].min():.3f}, {all_pts[:,2].max():.3f}]")

        if save_path:
            # fig is not pyplot-managed — save through the figure itself
            # (plt.savefig/plt.close would act on an empty "current figure").
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[viz] ✅ corner point cloud saved: {save_path}")
            return None

        return fig

    def visualize_corner_reprojection_results(self, reprojection_results,
                                            image_path_mapping=None, new_image_directory=None, save_dir=None,
                                            return_images=False):
        """
        Visualize per-corner reprojection results on the images (with corner order matching) - OpenCV.
        Draws the distortion-applied projected coordinates directly on the original (distorted) image.

        Args:
            reprojection_results: result of compute_reprojection_errors_with_corner_pointcloud()
            image_path_mapping: filename -> full path mapping dict {filename: full_path} (takes priority)
            new_image_directory: new camera image directory (legacy, used only without image_path_mapping)
            save_dir: save directory
            return_images: True returns the images as a dictionary {camera_id: image_vis}

        Returns:
            dict or None: {camera_id: image_vis} when return_images=True, else None
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # dictionary for returned images
        images_dict = {} if return_images else None

        # collect image file paths
        if image_path_mapping:
            # new style: use the filename -> path mapping
            print(f"  Using image_path_mapping with {len(image_path_mapping)} files")
        elif new_image_directory:
            # legacy style: glob the directory
            print(f"  Using legacy directory scan: {new_image_directory}")
            new_image_files = []
            for ext in self.image_extensions:
                new_image_files.extend(glob.glob(os.path.join(new_image_directory, ext)))
            new_image_files.sort()

            # build the filename -> path mapping
            image_path_mapping = {}
            for img_path in new_image_files:
                filename = os.path.basename(img_path)
                image_path_mapping[filename] = img_path
        else:
            print("  ⚠ No image source provided (image_path_mapping or new_image_directory)")
            return

        # per-corner colors (for order matching) - BGR format (OpenCV)
        corner_colors = [
            (0, 0, 255),      # corner 0: red (Bottom Right)
            (0, 255, 0),      # corner 1: green (Bottom Left)
            (255, 0, 0),      # corner 2: blue (Top Left)
            (0, 255, 255)     # corner 3: yellow (Top Right)
        ]

        # visualize per camera
        for camera_id, error_info in reprojection_results['camera_errors'].items():
            filename = error_info['filename']

            # load the original image
            image_path = image_path_mapping.get(filename)

            if image_path is None:
                print(f"  ⚠ Image not found for {filename}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue

            image_vis = image.copy()

            # draw the detection/projection results of every marker
            for detection_result in error_info['detections']:
                marker_id = detection_result['marker_id']
                corner_details = detection_result['corner_details']
                mean_error = detection_result['mean_error']

                # process per corner (order matched);
                # corner_details stores distorted image coordinates
                for corner_detail in corner_details:
                    corner_idx = corner_detail['corner_index']
                    detected_corner = corner_detail['detected']    # original distorted coordinates
                    projected_corner = corner_detail['projected']  # projected coordinates with distortion applied
                    error = corner_detail['error']                 # computed in distorted space
                    is_valid = corner_detail['valid']

                    if not is_valid:
                        continue

                    color = corner_colors[corner_idx]

                    # detected corner (colored circle with white border)
                    detected_pt = tuple(detected_corner.astype(int))
                    cv2.circle(image_vis, detected_pt, radius=12, color=(255, 255, 255), thickness=2)  # white border
                    cv2.circle(image_vis, detected_pt, radius=10, color=color, thickness=-1)  # filled colored circle

                    # projected corner (X mark)
                    projected_pt = tuple(projected_corner.astype(int))
                    cv2.drawMarker(image_vis, projected_pt, color=color,
                                 markerType=cv2.MARKER_CROSS, markerSize=15, thickness=3)

                    # line between matching corners (shows the order match)
                    cv2.line(image_vis, detected_pt, projected_pt, color=color, thickness=2)

                    # error text (at the midpoint)
                    mid_x = int((detected_corner[0] + projected_corner[0]) / 2)
                    mid_y = int((detected_corner[1] + projected_corner[1]) / 2)

                    error_text = f'{error:.2f}px'

                    # measure text size
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(error_text, font, font_scale, font_thickness)

                    # black background rectangle (with a little padding)
                    padding = 5
                    rect_pt1 = (mid_x - text_width // 2 - padding, mid_y - text_height // 2 - padding)
                    rect_pt2 = (mid_x + text_width // 2 + padding, mid_y + text_height // 2 + padding + baseline)
                    cv2.rectangle(image_vis, rect_pt1, rect_pt2, (0, 0, 0), -1)  # black background
                    cv2.rectangle(image_vis, rect_pt1, rect_pt2, (255, 255, 255), 1)  # white border

                    # draw text (white)
                    text_org = (mid_x - text_width // 2, mid_y + text_height // 2)
                    cv2.putText(image_vis, error_text, text_org, font, font_scale,
                              (255, 255, 255), font_thickness, cv2.LINE_AA)

                # summary text at the marker center
                if corner_details:
                    # center of the valid corners (already undistorted coordinates)
                    valid_corners = [d['detected'] for d in corner_details if d['valid']]

                    if valid_corners:
                        center_detected = np.mean(valid_corners, axis=0)
                        center_x = int(center_detected[0])
                        center_y = int(center_detected[1] - 30)

                        info_text = f"M{marker_id}: {mean_error:.2f}px"

                        # measure text size
                        font_scale_large = 1.6
                        font_thickness_large = 2
                        (text_width, text_height), baseline = cv2.getTextSize(info_text, font, font_scale_large, font_thickness_large)

                        # black background rectangle (larger padding)
                        padding = 8
                        rect_pt1 = (center_x - text_width // 2 - padding, center_y - text_height // 2 - padding)
                        rect_pt2 = (center_x + text_width // 2 + padding, center_y + text_height // 2 + padding + baseline)
                        cv2.rectangle(image_vis, rect_pt1, rect_pt2, (0, 0, 0), -1)  # black background
                        cv2.rectangle(image_vis, rect_pt1, rect_pt2, (255, 255, 255), 2)  # white border

                        # draw text (white, bold)
                        text_org = (center_x - text_width // 2, center_y + text_height // 2)
                        cv2.putText(image_vis, info_text, text_org, font, font_scale_large,
                                  (255, 255, 255), font_thickness_large, cv2.LINE_AA)

            # save (keeps the exact original resolution, no zero padding)
            if save_dir:
                save_path = os.path.join(save_dir, f"corner_reprojection_cam{camera_id}_{filename}")
                cv2.imwrite(save_path, image_vis)
                print(f"[viz] saved corner reprojection visualization: {save_path}")

            # keep the image for returning
            if return_images:
                images_dict[camera_id] = image_vis.copy()

        if return_images:
            return images_dict
        return None


# convenience function
def run_uncertainty_factor_graph_optimization(camera_params, apriltag_config=None,
                                            marker_size=0.4, marker_size2=0.2,
                                            image_directory='tmp_img/',
                                            dist_coeffs=None, wheel_camera_params=None,
                                            wheel_dist_coeffs=None, wheel_directory='tmp_img/wheel2/',
                                            marker_size_mapping=None, first_marker_id=None,
                                            vehicle_markers=None, image_extensions=None,
                                            min_pair_obs=2,
                                            excluded_apriltag_ids=None,
                                            enable_visualization=True):
    """
    Run uncertainty-based factor graph optimization for environment stage (Stage 1).

    All configuration values must be passed explicitly - no global variable fallbacks.

    Returns:
        tuple: (system, graph, initial_estimate, result, detections, uncertainty_info)
    """
    system = create_uncertainty_system(
        camera_params=camera_params,
        apriltag_config=apriltag_config,
        marker_size=marker_size,
        marker_size2=marker_size2,
        image_directory=image_directory,
        wheel_directory=wheel_directory,
        dist_coeffs=dist_coeffs,
        wheel_camera_params=wheel_camera_params,
        wheel_dist_coeffs=wheel_dist_coeffs,
        marker_size_mapping=marker_size_mapping,
        first_marker_id=first_marker_id,
        vehicle_markers=vehicle_markers,
        image_extensions=image_extensions,
        min_pair_obs=min_pair_obs,
        excluded_apriltag_ids=excluded_apriltag_ids,
        enable_visualization=enable_visualization,
    )

    graph, initial_estimate, result, detections, uncertainty_info = system.run_uncertainty_optimization()
    return system, graph, initial_estimate, result, detections, uncertainty_info


def run_incremental_optimization_with_multiple_cameras(previous_result,
                                                     camera_params=None,
                                                     apriltag_config=None, marker_size=0.4,
                                                     marker_size2=0.2,
                                                     marker_size_mapping=None,
                                                     vehicle_markers=None,
                                                     vehicle_camera_configs=None,
                                                     geo_skip_rot_deg=4.0, geo_skip_pos_m=0.10):
    """
    Add multiple vehicle camera observations to an existing optimization result and re-optimize.
    Each camera uses its own config and image list from vehicle_camera_configs.

    Args:
        previous_result: previous gtsam.Values optimization result
        camera_params: fallback [fx, fy, cx, cy] when camera config is missing
        apriltag_config: AprilTag/AprilGrid settings dict (see DEFAULT_APRILTAG_CONFIG)
        marker_size: environment marker half-size
        marker_size2: vehicle/wheel marker half-size
        marker_size_mapping: optional {marker_id: size} override
        vehicle_markers: list of vehicle corner marker IDs
        vehicle_camera_configs: list of per-camera config dicts with 'images', 'intrinsic', etc.
        geo_skip_rot_deg/geo_skip_pos_m: geometric consistency thresholds for the
            vehicle observations, passed from config

    Returns:
        tuple: (graph, initial_estimate, result, detections, new_camera_ids)
    """
    if vehicle_camera_configs is None:
        vehicle_camera_configs = []
    if vehicle_markers is None:
        vehicle_markers = []
    if marker_size_mapping is None:
        marker_size_mapping = {}

    # collect selected image files per camera
    vehicle_image_files = []  # [(image_path, camera_config_idx), ...]

    print("\n" + "=" * 60)
    print("[stage3] loading vehicle camera images (GUI selection only)")
    print("=" * 60)
    print(f"  [debug] number of vehicle_camera_configs: {len(vehicle_camera_configs)}")

    for cam_idx, camera_config in enumerate(vehicle_camera_configs):
        selected_images = camera_config.get('images', [])
        print(f"  [debug] Camera {cam_idx}: {len(selected_images)} images in config")

        if selected_images:
            print(f"  Camera {cam_idx}: {len(selected_images)} images (from GUI selection)")
            for img_path in selected_images:
                vehicle_image_files.append((img_path, cam_idx))
        else:
            print(f"  Camera {cam_idx}: No images selected (skipped)")

    if not vehicle_image_files:
        print("[stage3] cannot find vehicle camera images - please select images in each camera tab.")
        return None, None, None, None, None

    print(f"\n[stage3] loaded {len(vehicle_image_files)} images")
    print("=" * 60 + "\n")

    # create system for the vehicle cameras
    system = UncertaintyFactorGraph(
        camera_params=camera_params,
        apriltag_config=apriltag_config,
        marker_size=marker_size,
        marker_size2=marker_size2,
        marker_size_mapping=marker_size_mapping,
        vehicle_markers=vehicle_markers,
        vehicle_camera_configs=vehicle_camera_configs,
        geo_skip_rot_deg=geo_skip_rot_deg,
        geo_skip_pos_m=geo_skip_pos_m,
    )

    # detect markers for each image using the corresponding camera settings
    all_detections = {}
    filename_to_camera_idx = {}  # filename -> camera index mapping

    for image_path, cam_idx in vehicle_image_files:
        filename = os.path.basename(image_path)
        # Detection key embeds the camera index: two cameras may legitimately
        # select images with identical basenames (per-camera folders both holding
        # a frame_000.png), and a bare-basename key would let the later camera
        # silently overwrite the earlier one's detections.
        det_key = f"cam{cam_idx}_{filename}"

        # get the corresponding camera settings
        camera_config = vehicle_camera_configs[cam_idx]
        if camera_config.get('intrinsic') is None or camera_config['intrinsic'][0] <= 1.0:
            raise ValueError(f"Vehicle camera {cam_idx} intrinsic not set (fx={camera_config.get('intrinsic')}). "
                             f"Please set all camera parameters in Camera Parameters dialog before running Stage 3.")
        camera_params = camera_config['intrinsic']
        dist_coeffs = camera_config['dist_coeffs']
        camera_model = camera_config.get('camera_model', 'pinhole')

        print(f"[stage3] image {filename}: camera {cam_idx} used - intrinsic: {camera_params}, dist:{dist_coeffs}, model:{camera_model}")

        # detect markers using the corresponding camera settings
        detections = system.detect_markers_with_uncertainty(
            image_path, camera_params=camera_params,
            dist_coeffs=dist_coeffs,
            camera_model=camera_model
        )

        if detections:
            all_detections[det_key] = detections
            filename_to_camera_idx[det_key] = cam_idx  # save mapping

            # add camera_id information to each detection (needed for projection)
            camera_id = 20000 + cam_idx
            for detection in detections:
                detection['camera_id'] = camera_id  # include camera_id!

            marker_ids = [d['marker_id'] for d in detections]
            print(f"  → detected markers: {marker_ids}, Camera ID: {camera_id}")
        else:
            print("  → no markers found")

    if not all_detections:
        print("[stage3] ❌ cannot find markers in vehicle images!")
        return None, None, None, None, None  # return 5 values

    # vehicle cameras: outlier observations are filtered by geo_skip uncertainty thresholds
    # before being added to the graph (no Huber kernel).
    graph, initial_estimate, new_camera_ids = system.create_combined_factor_graph_with_base_result(
        previous_result, all_detections, camera_start_id=20000, camera_type='vehicle',
        filename_to_camera_idx=filename_to_camera_idx
    )

    # factor graph status
    print("\n[stage3] factor graph status:")
    print(f"  - Total Factors: {graph.size()}")
    print(f"  - Variables: {initial_estimate.size()}")
    print(f"  - new camera IDs: {sorted(new_camera_ids)}")
    print(f"  - Filename to Camera mapping: {filename_to_camera_idx}")

    # run optimization
    try:
        result = system.optimize_poses(graph, initial_estimate)
        print("[stage3] ✅ vehicle camera incremental optimization succeeded!")
    except Exception as e:
        # Returning the initial estimate here would let the caller treat an
        # unoptimized guess as a successful result and export it silently.
        print(f"[stage3] ⚠ optimization failed: {e}")
        result = None

    # Stage 1 environment cameras and Stage 2 wheel cameras are not variables of
    # this graph; carry their optimized poses over so the final result still
    # holds every stage's cameras. On optimize failure result stays None so
    # the caller cannot mistake an unoptimized guess for a success.
    if result is not None:
        carry_over_stage_poses(result, previous_result, label='[stage3] ')

    # attach the filename_to_camera_idx info to all_detections
    # so projection code can use it
    for filename, cam_idx in filename_to_camera_idx.items():
        if filename in all_detections:
            camera_id = 20000 + cam_idx
            # keep filename and camera_id info on each detection
            for detection in all_detections[filename]:
                if 'camera_id' not in detection:
                    detection['camera_id'] = camera_id

    return graph, initial_estimate, result, all_detections, new_camera_ids


def save_optimization_results(graph, initial_estimate, result, detections, uncertainty_info, filename="optimization_results.pkl"):
    """
    Save the optimization results to a file.

    Args:
        graph: GTSAM NonlinearFactorGraph
        initial_estimate: GTSAM Values (initial values)
        result: GTSAM Values (optimization results)
        detections: detection information dictionary
        uncertainty_info: uncertainty information dictionary
        filename: file name to save (default: "optimization_results.pkl")
    """
    save_path = os.path.join(g_current_dir, filename)

    # save all data as a tuple
    data_to_save = (graph, initial_estimate, result, detections, uncertainty_info)

    # save to a file
    with open(save_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"\n[io] ✅ optimization results saved to: {save_path}")
    if result is not None:
        marker_count = sum(1 for key in result.keys() if gtsam.Symbol(key).chr() == ord('M'))
        camera_count = sum(1 for key in result.keys() if gtsam.Symbol(key).chr() == ord('C'))
        print(f"  - marker count: {marker_count}")
        print(f"  - camera count: {camera_count}")
        print(f"  - detection count: {sum(len(v) for v in detections.values())}")


def load_optimization_results(filename="optimization_results.pkl"):
    """
    Load saved optimization results from a file.

    Args:
        filename: file name to load (default: "optimization_results.pkl")

    Returns:
        tuple: (graph, initial_estimate, result, detections, uncertainty_info)
    """
    load_path = os.path.join(g_current_dir, filename)

    if not os.path.exists(load_path):
        print(f"\n[io] ❌ saved optimization results file not found: {load_path}")
        return None, None, None, None, None

    with open(load_path, 'rb') as f:
        data = pickle.load(f)

    graph, initial_estimate, result, detections, uncertainty_info = data

    print(f"\n[io] loaded optimization results from: {load_path}")
    if result is not None:
        marker_count = sum(1 for key in result.keys() if gtsam.Symbol(key).chr() == ord('M'))
        camera_count = sum(1 for key in result.keys() if gtsam.Symbol(key).chr() == ord('C'))
        print(f"  - marker count: {marker_count}")
        print(f"  - camera count: {camera_count}")
        print(f"  - detection count: {sum(len(v) for v in detections.values())}")

    return graph, initial_estimate, result, detections, uncertainty_info
