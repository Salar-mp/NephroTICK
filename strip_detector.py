"""
strip_detector.py
-----------------
Automatic detection of the protein pad within a dipstick photograph.

The protein pad is the only strongly-coloured region in the frame. Everything
else (strip backing, background surface) is white or grey. The pad is isolated
by masking yellow-green to dark-green pixels in HSV space, filtering blobs by
size and shape, and taking the largest qualifying region as the pad.

Once the pad is located, the white plastic body adjacent to it is sampled for
the white body delta correction (see preprocessing.py). The full preprocessing
pipeline — Gray World normalization followed by white body correction — is
applied before colour extraction, so the RGB value passed to the classifier
is already compensated for ambient lighting.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dipstick_analyzer import extract_average_color, classify_protein, ProteinResult
from preprocessing import (
    gray_world_normalize,
    sample_white_body,
    apply_white_body_correction,
)
from reference_calibrator import calibrate_from_image


# ── Pad detection ─────────────────────────────────────────────────────────────

def find_protein_pad(
    img: np.ndarray,
    debug: bool = False,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Locates the protein pad and returns its bounding box as (x, y, w, h).

    Yellow-green to dark-green pixels are isolated in HSV space. Blobs are
    filtered by relative area (0.01%–5% of image) and aspect ratio (roughly
    square). The largest passing blob is taken as the pad. Returns None if
    no candidate is found.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Covers the full range from the NEG yellow-green through to the ++++ dark green.
    lower = np.array([ 30,  30,  50])
    upper = np.array([105, 255, 240])
    mask  = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img.shape[:2]
    total_area   = h_img * w_img

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area   = cv2.contourArea(c)
        aspect = max(w, h) / max(min(w, h), 1)
        rel    = area / total_area

        if not (0.0001 < rel < 0.05):
            continue
        if aspect > 2.5:
            continue
        if area < 300:
            continue

        candidates.append((area, x, y, w, h))

    if not candidates:
        return None

    if len(candidates) == 1:
        _, x, y, w, h = candidates[0]
    else:
        # When the reference chart is in the same photo as the strip, multiple
        # green blobs are found — the chart squares are clustered together while
        # the strip pad sits on its own. Selecting the most isolated blob
        # (the one furthest from all others) reliably picks the pad over the chart.
        centers = [(x + w // 2, y + h // 2) for (_, x, y, w, h) in candidates]
        isolation = []
        for i, (cx, cy) in enumerate(centers):
            min_dist = min(
                ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                for j, (ox, oy) in enumerate(centers) if j != i
            )
            isolation.append(min_dist)
        best_i = isolation.index(max(isolation))
        _, x, y, w, h = candidates[best_i]

    if debug:
        _save_debug(img, mask, x, y, w, h)

    return (x, y, w, h)


def _save_debug(img, mask, x, y, w, h):
    """Writes debug_mask.jpg and debug_detection.jpg when --debug is passed."""
    cv2.imwrite("debug_mask.jpg", mask)
    ann = img.copy()
    cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 255, 0), 4)
    cv2.putText(ann, "PAD", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    scale = min(1.0, 1000 / max(img.shape[:2]))
    small = cv2.resize(ann, (0, 0), fx=scale, fy=scale)
    cv2.imwrite("debug_detection.jpg", small)
    print("  Debug images saved: debug_mask.jpg, debug_detection.jpg")


# ── Analysis pipeline ─────────────────────────────────────────────────────────

def analyze_strip_auto(
    image_path: str,
    reference_scale: Optional[list] = None,
    use_gray_world: bool = True,
    use_white_body: bool = False,
    debug: bool = False,
) -> Tuple[ProteinResult, Tuple[int, int, int, int]]:
    """
    Full pipeline: load → normalise → detect pad → calibrate → classify.

    Calibration priority (highest first):
      1. In-photo chart — if the Albustix bottle colour chart is visible in the
         same frame as the strip, its six squares are detected automatically and
         used as the live reference. This is the most accurate path because both
         the pad and the chart are captured under identical lighting conditions,
         eliminating cross-session colour drift entirely.
      2. Supplied reference_scale — a scale extracted by the caller from a
         separate reference photograph (e.g. the bundled IMG_8050.jpeg).
      3. Hardcoded fallback — the factory reference values in PROTEIN_SCALE,
         used only when no other calibration is available.

    Preprocessing flags:
      use_gray_world  — neutral-pixel Gray World normalization (default on).
                        Estimates white balance from low-saturation background
                        pixels only, so coloured objects in frame (gloves,
                        reference chart squares) do not bias the correction.
      use_white_body  — strip plastic backing correction (default off).

    Returns (ProteinResult, bounding_box). Raises FileNotFoundError if the
    image cannot be opened, or RuntimeError if the pad is not found.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")

    # Step 1: Global colour normalisation — corrects ambient colour cast.
    preprocessing_steps = []
    if use_gray_world:
        img = gray_world_normalize(img)
        preprocessing_steps.append("gray_world")

    # Step 2: Attempt in-photo calibration from the reference chart in the frame.
    # calibrate_from_image reads all green blobs before pad detection removes the
    # pad blob, so it still has access to all 7 qualifying regions at this point.
    inphoto_scale = calibrate_from_image(img)
    if inphoto_scale is not None:
        preprocessing_steps.append("inphoto_cal")

    # Step 3: Locate the protein pad.
    bbox = find_protein_pad(img, debug=debug)
    if bbox is None:
        raise RuntimeError(
            "Could not locate the protein pad.\n"
            "  • Place the strip on a plain background\n"
            "  • Ensure the pad is in focus\n"
            "  • Use the flashlight for consistent illumination"
        )

    x, y, w, h = bbox
    region = img[y : y + h, x : x + w]
    rgb = extract_average_color(region)

    # Step 4: Per-image correction using the white strip body as reference.
    if use_white_body:
        white_bgr = sample_white_body(img, bbox)
        if white_bgr is not None:
            rgb = apply_white_body_correction(rgb, white_bgr)
            preprocessing_steps.append("white_body")

    preprocessing_label = "+".join(preprocessing_steps) if preprocessing_steps else "none"

    result = classify_protein(
        rgb,
        reference_scale=reference_scale,   # separate live ref (fallback if no in-photo)
        inphoto_scale=inphoto_scale,        # in-photo chart (highest priority)
        preprocessing=preprocessing_label,
    )
    return result, bbox


def annotate_result(
    image_path: str,
    result: ProteinResult,
    bbox: Tuple[int, int, int, int],
    out_path: str = "result_annotated.jpg",
) -> None:
    """
    Saves a copy of the image with the detected pad highlighted and the
    result printed alongside it. Green box indicates no clinician action
    needed; red indicates a flag for review. Preprocessing and calibration
    source are both noted in the annotation.
    """
    img = cv2.imread(image_path)
    x, y, w, h = bbox

    color = (0, 200, 0) if not result.clinician_flag else (0, 60, 255)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)

    label_lines = [
        f"Result: {result.label}  [{result.calibration_source} cal | {result.preprocessing}]",
        f"{result.description}",
        f"dE(a,b)={result.delta_e:.1f}  RGB={result.sampled_rgb}",
        "FLAG FOR CLINICIAN" if result.clinician_flag else "No clinician action needed",
    ]
    ty = max(y - 10, 140)
    for i, line in enumerate(label_lines):
        cv2.putText(img, line,
                    (x, ty - (len(label_lines) - 1 - i) * 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    scale = min(1.0, 1200 / max(img.shape[:2]))
    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imwrite(out_path, small)
    print(f"  Annotated result saved → {out_path}")
