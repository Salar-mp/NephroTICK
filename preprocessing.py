"""
preprocessing.py
----------------
Colour normalisation applied to dipstick images before pad analysis.

Three complementary techniques address lighting variation without requiring
any additional hardware:

  Gray World:       Assumes the average of all scene colours should be neutral
                    grey. Corrects global casts (warm bulbs, fluorescent tint)
                    by scaling each channel so the per-channel means are equal.

  White Patch:      Assumes the brightest region in the frame is a perfect
                    white reflector. Scales each channel relative to its top
                    percentile value. More aggressive than Gray World; best
                    when a genuinely white area is clearly visible.

  White Body Delta: Samples the strip's own white plastic backing — always
                    present in the photograph — and corrects the pad colour
                    relative to it. Because the backing and the pad are
                    illuminated by exactly the same light source, this gives
                    a per-image correction that requires no reference card and
                    no extra photograph from the user.

In the main pipeline, Gray World is applied first as a global pass and White
Body Delta is applied afterwards as a localised fine correction. Together they
account for the ambient colour temperature and any residual spatial variation
across the frame.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def gray_world_normalize(img: np.ndarray) -> np.ndarray:
    """
    Corrects a global colour cast using a neutral-pixel Gray World estimate.

    Rather than averaging all pixels (which biases the estimate when a
    strongly-coloured object — such as a blue latex glove or the green
    protein pad — occupies a significant portion of the frame), the
    illuminant is estimated exclusively from low-saturation, mid-to-high
    brightness pixels: the white paper, strip backing, and bench surface
    that form the neutral background.

    Pixels are retained for the estimate when their HSV saturation is below
    60/255 and their value (brightness) is above 60/255. If fewer than 1% of
    pixels qualify, the algorithm falls back to full-image averaging so that
    scenes with no visible neutral background still receive some correction.

    Once the per-channel means are computed from the neutral mask, every
    pixel in the image is scaled so those means become equal, neutralising
    the ambient colour cast without being misled by coloured foreground objects.
    """
    img_f = img.astype(np.float64)

    # Build a neutral-pixel mask in HSV space.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    neutral = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 60)

    min_fraction = 0.01
    if neutral.sum() < neutral.size * min_fraction:
        # Fall back to whole-image average when no neutral region is visible.
        neutral = np.ones(img.shape[:2], dtype=bool)

    mb = img_f[:, :, 0][neutral].mean()
    mg = img_f[:, :, 1][neutral].mean()
    mr = img_f[:, :, 2][neutral].mean()
    overall = (mb + mg + mr) / 3.0

    out = img_f.copy()
    if mb > 0:
        out[:, :, 0] = np.clip(img_f[:, :, 0] * (overall / mb), 0, 255)
    if mg > 0:
        out[:, :, 1] = np.clip(img_f[:, :, 1] * (overall / mg), 0, 255)
    if mr > 0:
        out[:, :, 2] = np.clip(img_f[:, :, 2] * (overall / mr), 0, 255)
    return out.astype(np.uint8)


def white_patch_normalize(img: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Corrects a colour cast using the White Patch (Max-RGB) algorithm.

    Each channel is scaled so its high-percentile value reaches 255.
    Using a percentile rather than the absolute maximum avoids over-correction
    from isolated specular highlights or sensor noise.
    """
    img_f = img.astype(np.float64)
    out = img_f.copy()
    for c in range(3):
        ch_max = np.percentile(img_f[:, :, c], percentile)
        if ch_max > 0:
            out[:, :, c] = np.clip(img_f[:, :, c] * (255.0 / ch_max), 0, 255)
    return out.astype(np.uint8)


def sample_white_body(
    img: np.ndarray,
    pad_bbox: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float, float]]:
    """
    Estimates the colour of the strip's white plastic backing by sampling
    the regions directly above and below the protein pad bounding box.

    The strip body (white/cream plastic) is expected to be adjacent to the
    pad at the same horizontal position. Both candidate regions are searched;
    pixels are filtered to keep only high-value, low-saturation content
    (HSV: saturation < 60, value > 160), then averaged using a trimmed mean.
    Returns a (B, G, R) triple, or None if too few white pixels are found.
    """
    x, y, w, h = pad_bbox
    img_h = img.shape[0]

    raw_pixels = []
    for offset_y in [-h, h]:
        ry = y + offset_y
        if ry < 0 or ry + h > img_h:
            continue
        region = img[ry : ry + h, x : x + w]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 160)
        if mask.sum() < 30:
            continue
        raw_pixels.append(region[mask].astype(np.float64))

    if not raw_pixels:
        return None

    pixels = np.vstack(raw_pixels)
    lo = np.percentile(pixels, 10, axis=0)
    hi = np.percentile(pixels, 90, axis=0)
    keep = np.all((pixels >= lo) & (pixels <= hi), axis=1)
    trimmed = pixels[keep] if keep.sum() >= 10 else pixels
    return tuple(trimmed.mean(axis=0))  # (B, G, R)


def check_blur(image_path: str, threshold: float = 80.0) -> Tuple[bool, float]:
    """
    Estimates image sharpness using the variance of the Laplacian.

    A sharp image has many strong edges, producing a high Laplacian variance.
    A blurry image (out of focus, hand-shake) has weak edges and a low variance.

    Returns (is_blurry, variance). Variance below `threshold` is flagged as
    blurry. Threshold of 80 is calibrated for Streamlit camera captures
    (typically 640×480–1280×720). Increase for higher-resolution inputs.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 0.0
    variance = float(cv2.Laplacian(img, cv2.CV_64F).var())
    return variance < threshold, variance


def apply_white_body_correction(
    rgb: Tuple[float, float, float],
    white_bgr: Tuple[float, float, float],
    true_white: float = 238.0,
) -> Tuple[float, float, float]:
    """
    Corrects a pad colour reading using the sampled white body as reference.

    Each channel is scaled by the ratio of the expected white value to the
    observed white body value under the current lighting. The correction is
    capped at ±40% to prevent runaway scaling on unusually dark or bright
    samples.

    true_white is set to 238 rather than 255 because the Albustix strip
    plastic is cream rather than optically pure white.
    """
    r, g, b = rgb
    wb, wg, wr = white_bgr

    def safe_scale(measured: float) -> float:
        if measured <= 0:
            return 1.0
        return max(0.6, min(1.4, true_white / measured))

    return (
        min(255.0, r * safe_scale(wr)),
        min(255.0, g * safe_scale(wg)),
        min(255.0, b * safe_scale(wb)),
    )
