"""
compare_image_quality.py

Compares two images to determine which has better *actual* quality,
independent of pixel dimensions (i.e. resistant to fake upscaling).

Usage:
    python compare_image_quality.py imageA.jpg imageB.jpg

Requires:
    pip install opencv-python numpy scipy
Optional (for BRISQUE/NIQE/MUSIQ no-reference scores):
    pip install pyiqa torch
"""

import sys
import cv2
import numpy as np
from PIL import Image as PILImage

# HEIC/HEIF (the default format Apple Photos/iPhone exports) has no reliable
# OS-level decoder behind cv2.imread, so PIL needs this optional plugin
# registered before PIL.Image.open can read those files. Mirrors the
# brisque/pyiqa pattern below: a missing optional dependency must never
# crash a scan, HEIC files just fail to decode and get treated the same as
# any other unreadable file.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Maximum image dimension for FFT-based analysis. Images larger than this on
# either side are downsampled before FFT to cap memory: a 6K×4K image (192 MB
# float64) would peak at over 1.5 GB through the FFT pipeline (complex128 FFT,
# power spectrum, index arrays, radial sum); capping at 2048 keeps peak per-worker
# memory under ~35 MB without meaningful loss of frequency-cutoff accuracy (the
# metric is a fraction of Nyquist, which is scale-invariant). This constant is a
# safety bound and should not be raised without re-verifying memory usage against
# the largest images your typical scan encounters.
EFFECTIVE_RES_MAX_PX = 2048


def _load_bgr_via_pil(path):
    """Fallback decode for formats cv2 can't read at all (currently just
    HEIC/HEIF), via PIL + the registered pillow-heif opener. Returns BGR
    uint8 to match cv2's channel convention, since callers (load_gray's
    cvtColor, brisque_score) both expect BGR like every cv2.imread result.
    Returns None (rather than raising) on any decode failure -- a HEIC file
    with no HEIF plugin installed, or a genuinely corrupt file, is treated
    the same as any other unreadable file."""
    try:
        with PILImage.open(path) as pil_img:
            rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        img = _load_bgr_via_pil(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return img, gray


def laplacian_sharpness(gray, target_size=(512, 512)):
    """Scale-normalized sharpness. Resize to a common size first,
    since Laplacian variance is not comparable across resolutions."""
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(resized, cv2.CV_64F)
    return lap.var()


def effective_resolution(gray):
    """
    Estimate the true information cutoff frequency via the radially
    averaged power spectrum. Returns:
      - cutoff_fraction: fraction of Nyquist frequency where real signal
        ends (1.0 = uses full native resolution, lower = effectively
        upscaled/soft beyond that point)
      - equivalent_pixels: cutoff_fraction * min(h, w), a rough proxy for
        "true" linear resolution regardless of stored dimensions
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0, 0.0  # too small for any meaningful frequency analysis

    # Downsample large images before FFT to cap memory. The cutoff_fraction
    # is a fraction of Nyquist (scale-invariant), so this doesn't change the
    # result. equivalent_pixels below uses the ORIGINAL min(h, w) so the
    # absolute estimate remains correct regardless of downsampling.
    orig_min_side = min(h, w)
    if max(h, w) > EFFECTIVE_RES_MAX_PX:
        scale = EFFECTIVE_RES_MAX_PX / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = gray.shape

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2

    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy)

    radial_power = np.bincount(r.ravel(), power.ravel())[:max_r]
    radial_power = radial_power / (np.arange(1, max_r + 1))  # normalize by ring area
    log_power = np.log(radial_power + 1e-8)

    # Smooth and find where the spectrum flattens into the noise floor.
    kernel = np.ones(5) / 5
    smoothed = np.convolve(log_power, kernel, mode="same")

    noise_floor = np.median(smoothed[-max(5, max_r // 20):])
    threshold = noise_floor + 0.5  # tolerance above floor

    cutoff_idx = 1  # fallback: near-zero when no ring exceeds the floor
    for i in range(max_r - 1, 0, -1):
        if smoothed[i] > threshold:
            cutoff_idx = i
            break

    cutoff_fraction = cutoff_idx / max_r
    equivalent_pixels = cutoff_fraction * orig_min_side
    return cutoff_fraction, equivalent_pixels


def noise_estimate(gray):
    """Fast noise sigma estimate (Immerkaer's method)."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0  # noise indistinguishable from signal at this size
    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    M = np.array(M, dtype=np.float64)
    conv = cv2.filter2D(gray, -1, M)
    sigma = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    return sigma


def blockiness_score(gray, block_size=8):
    """Detects JPEG-style blocking artifacts by measuring discontinuity
    strength at block boundaries vs within blocks."""
    h, w = gray.shape
    diff_h = np.abs(np.diff(gray, axis=1))
    diff_v = np.abs(np.diff(gray, axis=0))

    boundary_cols = np.arange(block_size - 1, w - 1, block_size)
    boundary_rows = np.arange(block_size - 1, h - 1, block_size)

    boundary_energy_h = diff_h[:, boundary_cols].mean() if len(boundary_cols) else 0
    boundary_energy_v = diff_v[boundary_rows, :].mean() if len(boundary_rows) else 0
    overall_h = diff_h.mean()
    overall_v = diff_v.mean()

    score = ((boundary_energy_h - overall_h) + (boundary_energy_v - overall_v)) / 2
    return max(score, 0)


def brisque_score(img_bgr):
    """
    BRISQUE via the `brisque` package (pip install brisque[opencv-python-headless]).
    Correct API is class-based (BRISQUE().score()), not a module-level function.
    Requires RGB input, not BGR, since the package's reference implementation
    builds its ndarray from PIL (RGB) images.
    Lower score = better perceived quality.
    """
    try:
        from brisque import BRISQUE
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        obj = BRISQUE(url=False)
        return float(obj.score(img=img_rgb))
    except Exception:
        return None


def niqe_score(path):
    """Optional secondary check via pyiqa (pip install pyiqa torch). Lower = better."""
    try:
        import pyiqa
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        niqe = pyiqa.create_metric("niqe", device=device)
        return float(niqe(path))
    except Exception:
        return None


def analyze(path):
    img, gray = load_gray(path)
    sharpness = laplacian_sharpness(gray)
    cutoff_fraction, eq_px = effective_resolution(gray)
    noise = noise_estimate(gray)
    blockiness = blockiness_score(gray)
    brisque = brisque_score(img)
    niqe = niqe_score(path)

    return {
        "path": path,
        "dimensions": (gray.shape[1], gray.shape[0]),
        "sharpness_normalized": sharpness,
        "effective_resolution_fraction": cutoff_fraction,
        "effective_resolution_px_equiv": eq_px,
        "noise_sigma": noise,
        "blockiness": blockiness,
        "brisque": brisque,
        "niqe": niqe,
    }


def report(a, b):
    print(f"\n{'Metric':30s} {'Image A':>18s} {'Image B':>18s}")
    print("-" * 68)
    print(f"{'Dimensions':30s} {str(a['dimensions']):>18s} {str(b['dimensions']):>18s}")
    print(f"{'Sharpness (normalized)':30s} {a['sharpness_normalized']:18.2f} {b['sharpness_normalized']:18.2f}")
    print(f"{'Effective res. fraction':30s} {a['effective_resolution_fraction']:18.3f} {b['effective_resolution_fraction']:18.3f}")
    print(f"{'Effective res. (px equiv)':30s} {a['effective_resolution_px_equiv']:18.1f} {b['effective_resolution_px_equiv']:18.1f}")
    print(f"{'Noise sigma (lower=cleaner)':30s} {a['noise_sigma']:18.3f} {b['noise_sigma']:18.3f}")
    print(f"{'Blockiness (lower=better)':30s} {a['blockiness']:18.3f} {b['blockiness']:18.3f}")
    if a["brisque"] is not None and b["brisque"] is not None:
        print(f"{'BRISQUE (lower=better)':30s} {a['brisque']:18.2f} {b['brisque']:18.2f}")
    else:
        print("\n(Install brisque[opencv-python-headless] for BRISQUE)")

    if a["niqe"] is not None and b["niqe"] is not None:
        print(f"{'NIQE (lower=better)':30s} {a['niqe']:18.2f} {b['niqe']:18.2f}")
    else:
        print("(Install pyiqa + torch for NIQE)")

    print("\nInterpretation:")
    print("- Higher 'effective resolution fraction' = less upscaled / more real detail per pixel.")
    print("- Higher sharpness (at matched scale) = more true detail, not just size.")
    print("- Lower noise, lower blockiness, lower BRISQUE/NIQE = better perceived quality.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_image_quality.py imageA imageB")
        sys.exit(1)

    result_a = analyze(sys.argv[1])
    result_b = analyze(sys.argv[2])
    report(result_a, result_b)
