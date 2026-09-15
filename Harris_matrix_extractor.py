# This code is for the uncertainty analysis of a corner detector using differentials

import numpy as np
import cv2

# Pixel-noise standard deviation entering the corner covariance
#
#     C = SIGMA_N**2 * M^-1        (Foerstner / KLT corner localisation)
#
# in normalized intensity units ([0,1] after normalization_img).
#
# 1/6 is NOT a measured noise level. It is the scale the code has always had:
# before the gradient kernels were normalized, M was inflated 36x, which is
# algebraically identical to sigma_n = 1/6 with correct gradients. Keeping that
# value makes this refactor change nothing numerically - it only turns an
# implicit assumption into an explicit one.
#
# The absolute sigma this produces is therefore arbitrary (measured ~21x larger
# than the actual corner error on the CarMaker dataset; the effective noise that
# would make it honest is ~2 gray levels, i.e. SIGMA_N = 2.0/255). The factor
# graph only ever uses ratios between observations, so this constant cancels
# out of the optimization entirely; it matters only if the reported sigma is to
# be read in physical units (robust kernels, marginal reporting). For real
# camera images, estimate the sensor noise from flat patches and put it here.
SIGMA_N = 1.0 / 6.0

# Structure-tensor conditioning for harris_corner_covariance().
# RCOND caps how anisotropic a corner covariance may get: eigenvalues below
# max_eigenvalue * RCOND are lifted to that floor. Relative rather than absolute so
# it is independent of image intensity scaling and of the gradient kernel scale.
HARRIS_EIGENVALUE_RCOND = 1e-6

# Below this the patch carries no gradient at all, so no conditioning makes the
# inverse meaningful and the covariance is reported as unavailable. Expressed in
# the normalized (/6) gradient scale: /36 keeps the cutoff identical to the
# legacy unnormalized structure tensor's 1e-12.
HARRIS_MIN_EIGENVALUE = 1e-12 / 36.0


def compute_gradients(input_img):
    """Image gradient in intensity units per pixel (normalized Prewitt).

    The 1/6 makes this a true derivative estimate: the central difference
    I(x+1) - I(x-1) spans two pixels (hence 1/2), and the three rows are an
    average, not a sum (hence 1/3). Without it every gradient is 6x too large,
    the structure tensor 36x, and the corner covariance silently 36x too small.
    """
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float64) / 6.0
    ky = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float64) / 6.0

    I_x = cv2.filter2D(input_img, cv2.CV_64F, kx)
    I_y = cv2.filter2D(input_img, cv2.CV_64F, ky)

    return I_x, I_y


def compute_derivatives(input_img):
    Ix, Iy = compute_gradients(input_img)

    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Iy * Ix

    return Ix2, Iy2, Ixy


def harris_corner_covariance(Ix2, Iy2, Ixy, img_point):
    x = round(img_point[0])
    y = round(img_point[1])

    # Custom (unnormalized) Gaussian weighting window; replaces the previously
    # used cv2.GaussianBlur. Note kernel.sum() != 1.
    w = 7
    sigma = 1.5
    k1d = np.exp(-np.linspace(-(w//2), w//2, w)**2/(2*sigma**2))
    kernel = np.outer(k1d, k1d)  # 2D kernel (outer product)

    # Structure tensor M at each pixel
    M11 = cv2.filter2D(Ix2, -1, kernel, borderType=cv2.BORDER_REFLECT)
    M12 = cv2.filter2D(Ixy, -1, kernel, borderType=cv2.BORDER_REFLECT)
    M22 = cv2.filter2D(Iy2, -1, kernel, borderType=cv2.BORDER_REFLECT)

    M = np.array([[M11[y, x], M12[y, x]],
                  [M12[y, x], M22[y, x]]])

    # A corner on a low-texture patch drives the eigenvalues towards zero, and
    # inverting that is unusable two ways: det exactly 0 yields a zero block that
    # makes the caller's Cholesky fail (discarding the whole image), while a merely
    # tiny det yields a 1e20-scale covariance that swamps every other observation.
    # Flooring the eigenvalues keeps the inverse finite and bounded, so an
    # uninformative corner ends up merely uncertain rather than fatal.
    M = (M + M.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.clip(eigenvalues, 0.0, None)   # kill negative float noise
    max_eig = eigenvalues.max()

    # Featureless patch: report it so the caller applies its own fallback rather
    # than this function inventing a magnitude.
    if max_eig <= HARRIS_MIN_EIGENVALUE:
        return None

    eigenvalues = np.maximum(eigenvalues, max_eig * HARRIS_EIGENVALUE_RCOND)
    covariance_matrix = SIGMA_N ** 2 * (
        eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T)

    return covariance_matrix


def normalization_img(input_img):
    norm_img = input_img / 255

    return norm_img
