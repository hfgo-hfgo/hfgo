"""
Pose Covariance Matrix Calculator

This module takes a preprocessed image and corner positions as input and
provides functions that compute and return only the pose covariance matrix
(UT_pnp_cov).

The covariance is propagated from the corner covariances through PnP with the
Unscented Transform (MARSCOT).
"""

import cv2
import gtsam
import numpy as np
from scipy.spatial.transform import Rotation as Ro
from scipy.linalg import block_diag
import scipy.linalg
from Harris_matrix_extractor import *

# ============================================================================
# Global configuration variables
# ============================================================================

# Marker size globals (synchronized with fgo_system.py)
g_marker_size_50 = 0.161  # half of marker size for marker 50

# Marker ID to size mapping
g_marker_size_mapping = {
    50: g_marker_size_50,
    # Add more as needed: 51: 0.15, 52: 0.25, ...
}

# Vehicle markers (synchronized with fgo_system.py and config.py)
g_vehicle_markers = [29, 30, 32, 31]  # FL, RL, RR, FR


def get_marker_size_for_covariance(marker_id, marker_size_default, marker_size2_default,
                                   vehicle_markers=None):
    """
    Return marker size based on marker ID (for covariance calculation)

    Args:
        marker_id: Marker ID
        marker_size_default: Default marker size (for environment markers)
        marker_size2_default: Vehicle wheel marker size (for wheel markers: FL, RL, RR, FR)
        vehicle_markers: List of vehicle marker IDs (default: g_vehicle_markers)

    Returns:
        float: Marker size (half size)
    """
    # Check special marker mapping
    if marker_id in g_marker_size_mapping:
        return g_marker_size_mapping[marker_id]

    # Check vehicle markers (default: [29, 30, 32, 31] = FL, RL, RR, FR)
    if vehicle_markers is None:
        vehicle_markers = g_vehicle_markers
        print(f"⚠ WARNING: vehicle_markers not provided, using default: {g_vehicle_markers}")
        print("  Please configure vehicle markers in settings for accurate calibration.")

    if marker_id in vehicle_markers:
        return marker_size2_default

    # Default size (environment markers)
    return marker_size_default


def pose_local_coordinates(R0, t0, R1, t1):
    """
    Deviation of pose 1 from pose 0 on the SE(3) manifold: Logmap(T0^-1 * T1).

    Replaces subtracting Euler angles, which wraps at +-180 degrees, degenerates at
    pitch = +-90 degrees, and does not live in the tangent space GTSAM interprets the
    covariance in. GTSAM computes it rather than a local log map so the result always
    matches the retraction the linked build actually linearises with.

    Returns [v; omega]; GTSAM's [omega; v] is flipped because this module builds its
    covariance in [x, y, z, rot_x, rot_y, rot_z] order.
    """
    T0 = gtsam.Pose3(gtsam.Rot3(R0), gtsam.Point3(np.asarray(t0, dtype=float).reshape(3)))
    T1 = gtsam.Pose3(gtsam.Rot3(R1), gtsam.Point3(np.asarray(t1, dtype=float).reshape(3)))

    # Logmap(T0^-1 * T1) is localCoordinates by definition; it is spelled out this
    # way because fgo_system already relies on exactly this pair of calls, so it is
    # known to exist in the linked build.
    xi = np.asarray(gtsam.Pose3.Logmap(T0.inverse().compose(T1)), dtype=float).reshape(6)
    return np.concatenate([xi[3:], xi[:3]])


def compute_corner_covariances(corners, preprocessed_image, crop_offset=10):
    """
    Compute the per-corner 2x2 pixel covariances (MARSCOT's corner uncertainty).

    Returns exactly the values calculate_pose_covariance() uses just before
    propagating to the pose, so the GUI's uncertainty-scale tuning view can
    visualize the very uncertainty that actually enters the optimization.

    Args:
        corners: np.array, shape (N, 2) - corner pixel coordinates [x, y]
        preprocessed_image: np.array - preprocessed grayscale image (blur → gray)
        crop_offset: int - crop offset around each corner

    Returns:
        np.array, shape (N, 2, 2)
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    n_corners = corners.shape[0]

    norm_img = normalization_img(preprocessed_image)
    tag_corner_cov = np.zeros((n_corners, 2, 2), dtype=np.float32)

    for corner_idx in range(n_corners):
        corner_point_for_cov = np.array([round(corners[corner_idx, 0]),
                                         round(corners[corner_idx, 1])])

        y_start = max(0, corner_point_for_cov[1] - crop_offset)
        y_end = min(norm_img.shape[0], corner_point_for_cov[1] + crop_offset + 1)
        x_start = max(0, corner_point_for_cov[0] - crop_offset)
        x_end = min(norm_img.shape[1], corner_point_for_cov[0] + crop_offset + 1)

        crop_img = norm_img[y_start:y_end, x_start:x_end]

        # Corner position inside the crop. Equals (crop_offset, crop_offset) only
        # while the crop is unclamped; near an image border x_start/y_start stop at
        # 0, which shifts the corner towards the crop's top-left.
        local_x = int(corner_point_for_cov[0]) - x_start
        local_y = int(corner_point_for_cov[1]) - y_start

        covariance = None
        if (crop_img.size
                and 0 <= local_x < crop_img.shape[1]
                and 0 <= local_y < crop_img.shape[0]):
            [Ix2, Iy2, Ixy] = compute_derivatives(crop_img)
            covariance = harris_corner_covariance(Ix2, Iy2, Ixy, [local_x, local_y])

        # Corner outside the frame, or on a patch with no gradient to measure. An
        # isotropic block keeps the stacked covariance positive definite, so one
        # unusable corner cannot invalidate the whole board.
        if covariance is None:
            covariance = np.eye(2)

        tag_corner_cov[corner_idx] = covariance

    return tag_corner_cov


def calculate_pose_covariance(corners, preprocessed_image, camera_params, object_points,
                            crop_offset=10,
                            ut_configs=None, dist_coeffs=None,
                            camera_model='pinhole'):
    """
    Compute the pose covariance matrix from a preprocessed image and corner positions.

    The number of corners is unrestricted. For boards composed of multiple tags
    (e.g. an AprilGrid), pass the corners of every tag detected on the board at
    once and the covariance of the whole board pose is computed
    (N = 4 * number of detected tags).

    Args:
        corners: np.array, shape (N, 2) - corner positions [x, y] of the marker/board
        preprocessed_image: np.array - preprocessed grayscale image (undistorted, blurred, gray)
        camera_params: list [fx, fy, cx, cy] - camera intrinsic parameters
        object_points: np.array, shape (N, 3) - 3D object point coordinates
        crop_offset: int - crop offset around each corner (default: 10)
        ut_configs: dict - UT settings (default: {'alpha': 1, 'beta': 2, 'kappa': 1});
                    n_dof is derived automatically from the corner count (2 * N)
        camera_model: str - 'pinhole' or 'fisheye'. For fisheye, dist_coeffs are
                     interpreted as Kannala-Brandt coefficients and the corners are
                     normalized before solving PnP. The same camera model as the
                     pose estimation path must be used so the covariance is
                     propagated on the correct geometry.

    Returns:
        np.array: 6x6 pose covariance matrix [x, y, z, roll, pitch, yaw]
        None: if PnP fails
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    n_corners = corners.shape[0]

    if ut_configs is None:
        ut_configs = {'alpha': 1, 'beta': 2, 'kappa': 1}
    # The state being propagated is the stacked 2D corner positions, so the UT
    # degrees of freedom follow the corner count rather than a fixed 4-corner tag.
    ut_configs = dict(ut_configs)
    ut_configs['n_dof'] = 2 * n_corners

    camera_matrix = np.array([
        [camera_params[0], 0, camera_params[2]],
        [0, camera_params[1], camera_params[3]],
        [0, 0, 1]
    ])

    # Solve PnP for the reference pose.
    # Fisheye coefficients cannot be passed to cv2.solvePnP as-is — solvePnP
    # interprets them with the pinhole distortion model (k1,k2,p1,p2), and even
    # with zero coefficients the projection equation itself differs
    # (pinhole r=f·tanθ vs fisheye r=f·θ). The same model as fgo_system's pose
    # path must be used so the covariance is propagated on the same geometry as
    # the pose.
    is_fisheye = (str(camera_model).lower() == 'fisheye')
    fisheye_D = None
    if is_fisheye:
        if dist_coeffs is None:
            raise ValueError(
                "calculate_pose_covariance: camera_model='fisheye' but dist_coeffs is None. "
                "fisheye requires the 4 KB coefficients (k1,k2,k3,k4)."
            )
        fisheye_D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        if fisheye_D.shape[0] != 4:
            raise ValueError(
                f"calculate_pose_covariance: fisheye requires 4 KB coefficients "
                f"but {fisheye_D.shape[0]} were provided."
            )

    # Starting point shared by every solve for this board. It is filled by the
    # first call, which is the nominal pose, and reused by all the sigma points.
    #
    # A planar target has two poses that project to nearly the same image, and
    # solvePnP picks between them from scratch on every call. Perturbing the
    # corners by the ~1 px the unscented transform asks for is enough to flip that
    # choice, and the flip is 90-180 degrees wide. The transform then reads that
    # jump as the pose's sensitivity to corner noise and reports a rotation sigma
    # of up to 100 degrees, when the pose is not uncertain at all - the solver just
    # answered a different question.
    #
    # Seeding keeps every perturbed solve in the nominal pose's basin, which is
    # what the unscented transform assumes it is measuring: a local sensitivity,
    # not a search over both global solutions. Boards seen edge-on with a single
    # tag are where this bites; a well-conditioned board is unaffected, since its
    # two solutions are far apart in reprojection error to begin with.
    pnp_seed = [None]

    def perform_pnp(obj_pts, img_pts):
        """Solve PnP and return the pose as (rotation matrix, translation).

        The rotation is returned as a matrix rather than Euler angles so callers can
        take deviations with pose_local_coordinates() on the SE(3) manifold. Euler
        angles cannot be subtracted safely: they wrap at +-180 degrees and degenerate
        at pitch = +-90 degrees, and their differences do not live in the tangent
        space GTSAM interprets the covariance in.
        """
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        # A degenerate sigma point can make OpenCV raise rather than return
        # success=False. Report it as an ordinary failure so the caller rejects this
        # one covariance instead of the exception discarding the whole image.
        try:
            if is_fisheye:
                pts = cv2.fisheye.undistortPoints(
                    img_pts.reshape(-1, 1, 2).astype(np.float64),
                    camera_matrix,
                    fisheye_D
                ).reshape(-1, 2).astype(np.float32)
                K_pnp, D_pnp = np.eye(3), None
            else:
                pts = img_pts
                K_pnp, D_pnp = camera_matrix, dist_coeffs

            if pnp_seed[0] is None:
                success, rvec, tvec = cv2.solvePnP(obj_pts, pts, K_pnp, D_pnp,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                rvec0, tvec0 = pnp_seed[0]
                success, rvec, tvec = cv2.solvePnP(obj_pts, pts, K_pnp, D_pnp,
                                                   rvec=rvec0.copy(), tvec=tvec0.copy(),
                                                   useExtrinsicGuess=True,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            return False, np.eye(3), np.zeros(3)

        if not success:
            return False, np.eye(3), np.zeros(3)

        if pnp_seed[0] is None:
            pnp_seed[0] = (rvec.copy(), tvec.copy())

        R = Ro.from_rotvec(rvec.reshape(3)).as_matrix()
        return success, R, tvec.reshape(-1, 3)[0]

    # Reference pose
    success, exis_R, exis_tranvec = perform_pnp(object_points, corners)
    if not success:
        return None

    tag_corner_cov = compute_corner_covariances(corners, preprocessed_image,
                                                crop_offset=crop_offset)

    P_x = block_diag(*tag_corner_cov)

    return _calculate_covariance_unscented(corners, P_x, object_points,
                                           exis_R, exis_tranvec,
                                           perform_pnp, ut_configs)


def _calculate_covariance_unscented(corners, P_x, object_points, nominal_R, nominal_t,
                                    perform_pnp, ut_configs):
    """
    Covariance calculation using the Unscented Transform (MARSCOT).

    Each sigma point's deviation is taken in SE(3) local coordinates relative to
    the nominal pose (not Euler-angle subtraction) — see pose_local_coordinates()
    for the detailed rationale.
    """
    # The state vector is the stacked (x, y) corners, so the DoF follows the corner count
    n_dim = int(ut_configs['n_dof'])

    def sigma_point(xm, cov):
        alpha = float(ut_configs['alpha'])
        n_dof = float(ut_configs['n_dof'])
        beta = float(ut_configs['beta'])
        kappa = float(ut_configs['kappa'])
        lamb_da = alpha ** 2 * (n_dof + kappa) - n_dof

        # Corner blocks are conditioned upstream so this should not fail, but an
        # unguarded LinAlgError would unwind to the broad handler in
        # detect_markers_with_uncertainty and discard the whole image.
        try:
            sigma_diff = scipy.linalg.cholesky((lamb_da + n_dof) * cov, lower=False)
        except (scipy.linalg.LinAlgError, np.linalg.LinAlgError) as exc:
            print(f"  [MARSCOT] sigma point generation failed ({exc}); "
                  f"rejecting this covariance")
            return None, None

        xm_flat = xm.flatten()

        Xi = np.zeros((2 * n_dim + 1, n_dim))
        Wci = np.zeros(2 * n_dim + 1)

        Xi[0] = xm_flat
        Wci[0] = (lamb_da / (n_dof + lamb_da)) + (1 - alpha ** 2 + beta)

        for i in range(n_dim):
            Xi[1 + i] = xm_flat + sigma_diff[i]
            Xi[1 + n_dim + i] = xm_flat - sigma_diff[i]
            weight = 1.0 / (2.0 * (n_dof + lamb_da))
            Wci[1 + i] = weight
            Wci[1 + n_dim + i] = weight

        return Xi, Wci

    def UT_func(Xi, sigma_cov_weight, obj_pts):
        xcov = np.zeros((6, 6))

        for i in range(Xi.shape[0]):
            local_point = Xi[int(i)]
            local_point_reshaped = local_point.reshape(-1, 2)

            success, R, t = perform_pnp(obj_pts, local_point_reshaped)
            if not success:
                # The UT weights are a fixed quadrature rule, so skipping a point
                # biases the covariance low — and biases it downward specifically,
                # since the points that fail are the extreme perturbations carrying
                # the largest deviations. Reject rather than under-estimate.
                print(f"  [MARSCOT] sigma point {i}/{Xi.shape[0]} PnP failed; "
                      f"rejecting this covariance")
                return None

            tmp_deviation = pose_local_coordinates(nominal_R, nominal_t, R, t)
            xcov += sigma_cov_weight[int(i)] * np.outer(tmp_deviation, tmp_deviation)

        return xcov

    # UT uncertainty estimate
    sigma_points, sigma_cov_weight = sigma_point(corners.reshape(1, -1), P_x)
    if sigma_points is None:
        return None
    UT_pnp_cov = UT_func(sigma_points, sigma_cov_weight, object_points)

    return UT_pnp_cov


def calculate_pose_covariance_april(corners, preprocessed_image, camera_params,
                                  marker_id=0, marker_size=0.21/2, marker_size2=0.35/2,
                                  dist_coeffs=None,
                                  vehicle_markers=None,
                                  object_points=None, camera_model='pinhole'):
    """
    Convenience function for AprilTag - derives object_points from the marker ID
    when the caller does not supply them.

    Multi-tag boards (AprilGrid) must pass object_points explicitly, since their
    board-frame corner layout depends on the grid geometry rather than on a single
    tag size.

    Args:
        corners: np.array, shape (N, 2) - corner positions [x, y] of the marker/board
        preprocessed_image: np.array - preprocessed grayscale image
        camera_params: list [fx, fy, cx, cy] - camera intrinsic parameters
        marker_id: int - Marker ID (vehicle wheel markers use marker_size2)
        marker_size: float - default marker size (for environment markers)
        marker_size2: float - vehicle wheel marker size (for wheel markers: FL, RL, RR, FR)
        vehicle_markers: list - vehicle marker IDs (default: g_vehicle_markers)
        object_points: np.array, shape (N, 3) - marker-frame corner coordinates.
                     If None, a single-tag layout is generated from the marker size.
        camera_model: str - 'pinhole' or 'fisheye'. Must match the model used for the
                     pose estimate, otherwise the covariance is propagated through a
                     different projection than the pose it belongs to.

    Returns:
        np.array: 6x6 pose covariance matrix
        None: if failed
    """
    if object_points is None:
        # Use helper function to determine marker size
        current_marker_size = get_marker_size_for_covariance(
            marker_id,
            marker_size,
            marker_size2,
            vehicle_markers
        )

        # AprilTag object points (Bottom right, Bottom left, Top left, Top right)
        object_points = np.array([
            [current_marker_size, current_marker_size, 0.0],   # Bottom right
            [-current_marker_size, current_marker_size, 0.0],  # Bottom left
            [-current_marker_size, -current_marker_size, 0.0], # Top left
            [current_marker_size, -current_marker_size, 0.0]   # Top right
        ])

    return calculate_pose_covariance(corners, preprocessed_image, camera_params, object_points,
                                    dist_coeffs=dist_coeffs,
                                    camera_model=camera_model)


# Usage example
if __name__ == "__main__":
    # Example data
    camera_params = [959.8362, 960.2749, 626.4311, 356.1123]

    # Synthetic corner positions (4 corners)
    corners = np.array([
        [100.0, 100.0],  # Bottom right
        [50.0, 100.0],   # Bottom left
        [50.0, 50.0],    # Top left
        [100.0, 50.0]    # Top right
    ])

    # Synthetic preprocessed image (in practice: undistorted, blurred, gray)
    preprocessed_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    covariance_matrix = calculate_pose_covariance_april(
        corners=corners,
        preprocessed_image=preprocessed_image,
        camera_params=camera_params,
        marker_id=0,
    )

    if covariance_matrix is not None:
        print("Pose Covariance Matrix (6x6):")
        print(covariance_matrix)
        print("\nDiagonal elements (variances):")
        print("Position variances [x, y, z]:", np.diag(covariance_matrix[:3, :3]))
        print("Orientation variances [roll, pitch, yaw]:", np.diag(covariance_matrix[3:, 3:]))
    else:
        print("Covariance calculation failed")
