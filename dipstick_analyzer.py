"""
dipstick_analyzer.py
--------------------
Core colour analysis for the Albustix protein pad.

Classification works in three stages:

  1. The sampled (and preprocessed) RGB is converted to CIE L*a*b* colour
     space, which separates lightness (L) from colour (a, b). Full CIE76
     Delta-E across all three channels is used for comparison. An a,b-only
     distance function is also provided (see delta_e_ab_only) for conditions
     where the reference and sample are photographed in the same session under
     identical lighting — in that case, dropping L gives additional robustness
     to brightness variation.

  2. The a,b point is compared against a reference scale of six protein
     concentrations using Euclidean distance in a,b space.

  3. If the nearest match falls within a per-band threshold (65% of the
     smallest adjacent inter-band gap), it is accepted as that label.
     Otherwise the result is reported as TRACE, meaning the reading sits
     ambiguously between two adjacent bands on the scale.

The reference scale defaults to values captured under controlled conditions
(IMG_8050.jpeg). When a live calibration is provided via reference_calibrator.py,
those values are used instead for a session-specific match.
"""

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2


# Reference scale: RGB values for each protein concentration band on the
# Albustix strip, measured from a controlled photograph (IMG_8050.jpeg).
PROTEIN_SCALE = [
    {"label": "NEG",   "rgb": (195, 209, 168), "description": "Negative — no protein detected"},
    {"label": "SP/TR", "rgb": (171, 192, 140), "description": "Trace — within normal limits"},
    {"label": "+",     "rgb": (129, 172, 112), "description": "30 mg/dL — mild proteinuria"},
    {"label": "++",    "rgb": ( 93, 155, 103), "description": "100 mg/dL — moderate proteinuria"},
    {"label": "+++",   "rgb": ( 54, 127,  85), "description": "300 mg/dL — significant proteinuria"},
    {"label": "++++",  "rgb": ( 30,  95,  62), "description": "2000 mg/dL — severe proteinuria"},
]

# Labels at ++ or above trigger a clinician review flag.
CLINICIAN_FLAG_LABELS = {"++", "+++", "++++", "TRACE"}

# Fallback threshold used when no live calibration is available.
BAND_MATCH_THRESHOLD = 12.0


@dataclass
class ProteinResult:
    label: str
    description: str
    delta_e: float           # Distance in a,b space to the nearest reference band
    sampled_rgb: tuple
    clinician_flag: bool
    calibration_source: str  # "live" or "fallback"
    preprocessing: str       # e.g. "gray_world+white_body", "none"


# ── Colour conversion ──────────────────────────────────────────────────────────

def rgb_to_lab(r: float, g: float, b: float) -> tuple:
    """Converts an sRGB triplet (0–255) to CIE L*a*b* via OpenCV."""
    pixel = np.uint8([[[b, g, r]]])  # OpenCV expects BGR
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    L, a, bb = lab[0, 0]
    # OpenCV encodes L in [0,255] and shifts a,b by 128
    return (L * 100.0 / 255.0, float(a) - 128.0, float(bb) - 128.0)


def delta_e_ab_only(lab1: tuple, lab2: tuple) -> float:
    """
    Colour distance using only the a and b channels, ignoring lightness.

    Dropping L means that two pads of the same hue photographed at different
    brightness levels will have zero distance from each other. This is the
    key property that makes classification robust to shadows, dim rooms, and
    variable flash intensity.
    """
    return math.sqrt((lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2)


def delta_e_cie76(lab1: tuple, lab2: tuple) -> float:
    """Full CIE76 Delta-E including lightness — used for threshold calculation."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))


# ── Colour extraction ──────────────────────────────────────────────────────────

def extract_average_color(region: np.ndarray) -> tuple:
    """
    Returns the trimmed-mean BGR colour of a cropped region as (R, G, B).

    The brightest and darkest 10% of pixels in each channel are discarded
    before averaging to reduce the effect of specular highlights and shadows
    within the pad area.
    """
    f = region.astype(np.float64)
    result = []
    for c in range(3):
        ch = f[:, :, c].ravel()
        lo, hi = np.percentile(ch, 10), np.percentile(ch, 90)
        trimmed = ch[(ch >= lo) & (ch <= hi)]
        result.append(float(trimmed.mean()) if len(trimmed) else float(ch.mean()))
    b_mean, g_mean, r_mean = result
    return (r_mean, g_mean, b_mean)


# ── Reference scale helpers ────────────────────────────────────────────────────

def _build_reference_lab(scale: list) -> list:
    """Converts each entry in the protein scale from RGB to L*a*b*."""
    return [rgb_to_lab(*entry["rgb"]) for entry in scale]


def _per_band_thresholds(scale: list, ref_labs: list) -> list:
    """
    Calculates a per-band matching threshold for each entry in the scale.

    Each band's threshold is set to 75% of its smallest adjacent gap — the
    full Delta-E distance to its nearest neighbour. This percentage gives
    well-separated bands (e.g. +, ++) a generous threshold that tolerates
    natural variation in strip colours, while tightly-spaced bands (e.g.
    SP/TR and NEG) remain narrow enough to detect ambiguous readings and
    report them as TRACE.

    75% was chosen over the more commonly cited 65% to account for
    real-world variability between photo sessions. Readings that fall
    outside all per-band thresholds are reported as TRACE.
    """
    n = len(scale)
    thresholds = []
    for i in range(n):
        gaps = []
        if i > 0:
            gaps.append(delta_e_cie76(ref_labs[i], ref_labs[i - 1]))
        if i < n - 1:
            gaps.append(delta_e_cie76(ref_labs[i], ref_labs[i + 1]))
        thresholds.append(min(gaps) * 0.75)
    return thresholds


# ── In-photo colour correction ─────────────────────────────────────────────────

def warp_lab_to_hardcoded(pad_lab: tuple, inphoto_scale: list) -> tuple:
    """
    Applies an affine colour correction that maps a pad colour measured in an
    in-photo reference frame to the equivalent position in the hardcoded
    reference frame.

    When the strip and the Albustix bottle chart are photographed together, the
    chart squares can be measured under the same lighting as the pad. An affine
    transform is fitted from the six (in-photo Lab, hardcoded Lab) band pairs
    using least squares. Applying this transform to the pad's Lab value accounts
    for session-to-session colour drift — changes in colour temperature, camera
    white balance, exposure — without requiring a separate reference photograph.

    After warping, the pad Lab value is compared against the hardcoded scale
    using the standard per-band thresholds, so the classification thresholds
    remain calibrated to controlled-conditions measurements.
    """
    inphoto_labs  = np.array([list(rgb_to_lab(*e["rgb"])) for e in inphoto_scale])
    hardcoded_labs = np.array([list(rgb_to_lab(*e["rgb"])) for e in PROTEIN_SCALE])

    # Augmented least-squares: [L, a, b, 1] * T = [L', a', b']
    X = np.column_stack([inphoto_labs, np.ones(len(inphoto_labs))])
    T, _, _, _ = np.linalg.lstsq(X, hardcoded_labs, rcond=None)

    v = np.append(np.array(pad_lab, dtype=np.float64), 1.0)
    return tuple(float(x) for x in v @ T)


# ── Classification ─────────────────────────────────────────────────────────────

def classify_protein(
    rgb: tuple,
    reference_scale: Optional[list] = None,
    inphoto_scale: Optional[list] = None,
    preprocessing: str = "none",
) -> ProteinResult:
    """
    Classifies a pad colour reading against the protein reference scale.

    Calibration priority (highest → lowest):

      1. In-photo chart (inphoto_scale) — when the Albustix bottle chart is
         visible in the same frame as the strip, its six squares are used to
         fit a per-image affine colour correction. The pad's Lab value is
         warped into the hardcoded reference frame and then classified with
         the standard per-band thresholds. This fully eliminates cross-session
         colour drift.

      2. Separate live scale (reference_scale) — a scale extracted from a
         separately photographed reference image. Per-band thresholds derived
         from the live inter-band gaps are used for classification.

      3. Hardcoded fallback (PROTEIN_SCALE) — the factory reference values
         captured under controlled conditions. A single global threshold
         (BAND_MATCH_THRESHOLD) is applied.

    Full CIE76 Delta-E (L, a, b) is used throughout.
    """
    sample_lab = rgb_to_lab(*rgb)

    # ── Path 1: in-photo affine warp ─────────────────────────────────────────
    if inphoto_scale is not None:
        warped_lab     = warp_lab_to_hardcoded(sample_lab, inphoto_scale)
        hardcoded_labs = _build_reference_lab(PROTEIN_SCALE)
        thresholds     = _per_band_thresholds(PROTEIN_SCALE, hardcoded_labs)
        distances      = [delta_e_cie76(warped_lab, hl) for hl in hardcoded_labs]
        best_idx       = int(np.argmin(distances))
        best_dist      = distances[best_idx]
        matched        = best_dist <= thresholds[best_idx]
        scale          = PROTEIN_SCALE
        calibration_source = "inphoto"

    # ── Path 2: separate live reference scale ─────────────────────────────────
    elif reference_scale is not None:
        scale          = reference_scale
        ref_labs       = _build_reference_lab(scale)
        thresholds     = _per_band_thresholds(scale, ref_labs)
        distances      = [delta_e_cie76(sample_lab, rl) for rl in ref_labs]
        best_idx       = int(np.argmin(distances))
        best_dist      = distances[best_idx]
        matched        = best_dist <= thresholds[best_idx]
        calibration_source = "live"

    # ── Path 3: hardcoded fallback ────────────────────────────────────────────
    else:
        scale          = PROTEIN_SCALE
        ref_labs       = _build_reference_lab(scale)
        distances      = [delta_e_cie76(sample_lab, rl) for rl in ref_labs]
        best_idx       = int(np.argmin(distances))
        best_dist      = distances[best_idx]
        matched        = best_dist <= BAND_MATCH_THRESHOLD
        calibration_source = "fallback"

    if matched:
        entry       = scale[best_idx]
        label       = entry["label"]
        description = entry.get("description", label)
    else:
        label       = "TRACE"
        description = "Reading falls between two scale bands — verify manually"

    return ProteinResult(
        label=label,
        description=description,
        delta_e=round(best_dist, 2),
        sampled_rgb=tuple(round(v) for v in rgb),
        clinician_flag=(label in CLINICIAN_FLAG_LABELS),
        calibration_source=calibration_source,
        preprocessing=preprocessing,
    )
