"""
reference_calibrator.py
-----------------------
Live calibration from a photograph of the Albustix reference colour chart.

The reference chart printed on the bottle shows six protein concentration
bands side by side under standardised printing conditions. Photographing the
chart under the same lighting conditions as the test strip — and applying the
same Gray World normalization to both — allows the classifier to match against
colours rendered under the current session's lighting rather than the fixed
values captured at calibration time.

detect_reference_bands() finds all six coloured blobs in the bottle photograph,
sorts them by vertical position (top = NEG, bottom = ++++), validates
orientation, and returns a reference scale in the same format as PROTEIN_SCALE.
The returned scale is passed directly into classify_protein() via strip_detector.
"""

import cv2
import numpy as np
from typing import Optional
from preprocessing import gray_world_normalize


# Expected number of distinct colour bands on the reference chart.
EXPECTED_BANDS = 6

PROTEIN_LABELS = ["NEG", "SP/TR", "+", "++", "+++", "++++"]
PROTEIN_DESCRIPTIONS = [
    "Negative — no protein detected",
    "Trace — within normal limits",
    "30 mg/dL — mild proteinuria",
    "100 mg/dL — moderate proteinuria",
    "300 mg/dL — significant proteinuria",
    "2000 mg/dL — severe proteinuria",
]


def detect_reference_bands(
    image_path: str,
    use_gray_world: bool = True,
) -> Optional[list]:
    """
    Detects the six reference colour bands from a bottle photograph.

    Returns a list of six dicts in PROTEIN_SCALE format (label, rgb,
    description), ordered NEG → ++++ by vertical position. Returns None
    if exactly six bands cannot be identified.

    Gray World normalization is applied by default, matching the same
    preprocessing applied to sample images so that both are corrected
    consistently before comparison.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")

    if use_gray_world:
        img = gray_world_normalize(img)

    blobs = _find_colour_blobs(img)

    if len(blobs) != EXPECTED_BANDS:
        print(f"  Calibration: expected {EXPECTED_BANDS} bands, found {len(blobs)}. "
              "Falling back to hardcoded scale.")
        return None

    # Sort by vertical centre: top of image = NEG (lightest), bottom = ++++ (darkest).
    blobs.sort(key=lambda b: b["cy"])

    # Orientation check: the top blob (NEG) should be lighter in L* than the
    # bottom blob (++++). If not, the bottle is upside-down — reverse the order.
    top_L = _bgr_to_L(blobs[0]["bgr"])
    bot_L = _bgr_to_L(blobs[-1]["bgr"])
    if top_L < bot_L:
        blobs.reverse()

    scale = []
    for i, blob in enumerate(blobs):
        b, g, r = blob["bgr"]
        scale.append({
            "label":       PROTEIN_LABELS[i],
            "rgb":         (int(r), int(g), int(b)),
            "description": PROTEIN_DESCRIPTIONS[i],
        })

    return scale


def calibrate_from_image(img: np.ndarray) -> Optional[list]:
    """
    Extracts a live reference scale from an image that contains both the
    protein pad and the Albustix bottle colour chart in the same frame.

    This is the preferred calibration path when the strip and chart are
    photographed together under identical lighting — the chart colours are
    measured under exactly the same conditions as the pad, so no cross-session
    colour drift is possible.

    Algorithm:
      1. Find all qualifying green blobs (same HSV mask as pad detection).
      2. Identify the most isolated blob as the protein pad.
      3. From the remaining blobs, select the six most tightly clustered
         ones — these form the reference chart column.
      4. Sort those six by dominant axis (vertical or horizontal chart
         orientation) and validate that lightness decreases from NEG to ++++.
      5. Return a PROTEIN_SCALE-format list, or None if no valid chart found.

    Requires at least 7 qualifying blobs (1 pad + 6 chart squares). When
    more than 7 are present (specular highlights, shadows etc.), the six
    most clustered non-pad blobs are taken.
    """
    all_blobs = _find_colour_blobs_with_bbox(img)

    if len(all_blobs) < EXPECTED_BANDS + 1:
        return None  # Not enough blobs for pad + full chart

    # ── Step 1: isolate the pad blob ──────────────────────────────────────────
    centers = [(b["cx"], b["cy"]) for b in all_blobs]
    isolation = []
    for i, (cx, cy) in enumerate(centers):
        min_d = min(
            ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            for j, (ox, oy) in enumerate(centers) if j != i
        )
        isolation.append(min_d)
    pad_idx = isolation.index(max(isolation))
    non_pad = [b for i, b in enumerate(all_blobs) if i != pad_idx]

    # ── Step 2: find the 6 blobs that form the reference chart column ─────────
    # The chart squares are printed in a straight column (or row), so they share
    # a narrow band of x-coordinates (vertical chart) or y-coordinates
    # (horizontal chart). This axis-alignment check is much more robust than
    # generic nearest-neighbour clustering when many spurious blobs are present.
    chart_blobs = _select_chart_column(non_pad)
    if chart_blobs is None:
        return None

    # ── Step 3: sort along the dominant axis ─────────────────────────────────
    xs = [b["cx"] for b in chart_blobs]
    ys = [b["cy"] for b in chart_blobs]
    if max(ys) - min(ys) >= max(xs) - min(xs):
        chart_blobs.sort(key=lambda b: b["cy"])   # vertical strip
    else:
        chart_blobs.sort(key=lambda b: b["cx"])   # horizontal strip

    # ── Step 4: orientation validation ───────────────────────────────────────
    top_L = _bgr_to_L(chart_blobs[0]["bgr"])
    bot_L = _bgr_to_L(chart_blobs[-1]["bgr"])
    if top_L < bot_L:
        chart_blobs.reverse()  # chart is upside-down or right-to-left

    # ── Step 5: build scale ───────────────────────────────────────────────────
    scale = []
    for i, blob in enumerate(chart_blobs):
        b, g, r = blob["bgr"]
        scale.append({
            "label":       PROTEIN_LABELS[i],
            "rgb":         (int(r), int(g), int(b)),
            "description": PROTEIN_DESCRIPTIONS[i],
        })
    return scale


def _select_chart_column(blobs: list, axis_tolerance: float = 0.15) -> Optional[list]:
    """
    Identifies the six reference-chart blobs from a mixed set that may also
    contain spurious blobs (reflections, strip body, coloured backgrounds).

    The Albustix bottle reference chart is always printed as a straight column
    (or, if the bottle is rotated, a straight row). All six squares therefore
    share a narrow band of x-coordinates (vertical chart) or y-coordinates
    (horizontal chart). Spurious blobs scattered elsewhere in the frame will
    not satisfy this alignment constraint.

    Strategy:
      1. For each candidate axis (x or y), sort all blobs along that axis and
         try every consecutive window of six. A window qualifies when the span
         of the *perpendicular* coordinate is small relative to the span of the
         *parallel* coordinate — i.e. the six blobs form a narrow column/row
         rather than a scattered cloud.
      2. Among qualifying windows, prefer the one with the smallest
         perpendicular spread (most column-like).
      3. Fall back to the old nearest-neighbour clustering when no axis-aligned
         window is found (e.g. chart photographed at a steep angle).

    axis_tolerance controls how wide the column may be: the perpendicular span
    must be ≤ axis_tolerance × parallel span. Default 0.15 (15% of height).
    """
    if len(blobs) < EXPECTED_BANDS:
        return None
    if len(blobs) == EXPECTED_BANDS:
        return blobs

    n = len(blobs)
    best_group  = None
    best_perp   = float("inf")

    for sort_key, perp_key in [("cx", "cy"), ("cy", "cx")]:
        sorted_blobs = sorted(blobs, key=lambda b: b[sort_key])
        for start in range(n - EXPECTED_BANDS + 1):
            group = sorted_blobs[start : start + EXPECTED_BANDS]
            parallel = max(b[sort_key] for b in group) - min(b[sort_key] for b in group)
            perp     = max(b[perp_key] for b in group) - min(b[perp_key] for b in group)

            if perp < 1:    # degenerate: all at same perpendicular coord
                continue
            # A valid column/row has a SMALL span along the sorted axis
            # (all squares share nearly the same x for vertical, same y for
            # horizontal) relative to the span in the other direction.
            if parallel > axis_tolerance * perp:
                continue       # too spread along the column axis — not a column

            if perp < best_perp:
                best_perp  = perp
                best_group = group

    if best_group is not None:
        return best_group

    # ── Fallback: nearest-neighbour clustering (original approach) ────────────
    centers = [(b["cx"], b["cy"]) for b in blobs]
    scores = []
    for i, (cx, cy) in enumerate(centers):
        d = min(
            ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            for j, (ox, oy) in enumerate(centers) if j != i
        )
        scores.append(d)
    ranked = sorted(range(len(blobs)), key=lambda i: scores[i])
    return [blobs[i] for i in ranked[:EXPECTED_BANDS]]


def _find_colour_blobs(img: np.ndarray) -> list:
    """
    Finds green-range colour blobs in the image using HSV masking.

    Uses the same hue range as pad detection (H 30–105, S >30, V >50).
    Blobs are filtered by relative area (0.01%–5%) and aspect ratio (<2.5).
    Returns a list of dicts with keys bgr (mean colour) and cy (vertical centre).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([ 30,  30,  50])
    upper = np.array([105, 255, 240])
    mask  = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = img.shape[:2]
    total_area   = h_img * w_img

    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        rel  = area / total_area
        if not (0.0001 < rel < 0.05):
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 2.5:
            continue
        if area < 300:
            continue

        region = img[y : y + h, x : x + w]
        mean_bgr = region.reshape(-1, 3).mean(axis=0)
        blobs.append({"bgr": mean_bgr, "cy": y + h / 2})

    return blobs


def _find_colour_blobs_with_bbox(img: np.ndarray) -> list:
    """
    Same as _find_colour_blobs but also records each blob's horizontal centre
    (cx) and bounding box so callers can reason about position in both axes.
    Used by calibrate_from_image() for pad/chart separation.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([ 30,  30,  50])
    upper = np.array([105, 255, 240])
    mask  = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = img.shape[:2]
    total_area   = h_img * w_img

    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        rel  = area / total_area
        if not (0.0001 < rel < 0.05):
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 2.5:
            continue
        if area < 300:
            continue

        region = img[y : y + h, x : x + w]
        mean_bgr = region.reshape(-1, 3).mean(axis=0)
        blobs.append({
            "bgr": mean_bgr,
            "cx": x + w / 2,
            "cy": y + h / 2,
            "bbox": (x, y, w, h),
        })

    return blobs


def _bgr_to_L(bgr: np.ndarray) -> float:
    """Returns the L* lightness of a BGR colour, used for orientation validation."""
    b, g, r = bgr
    pixel = np.uint8([[[b, g, r]]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    return float(lab[0, 0, 0])


def print_calibration_summary(live_scale: list, hardcoded_scale: list) -> None:
    """
    Prints a side-by-side comparison of the live and hardcoded reference scales.
    Distances are reported in a,b space only (consistent with classification).
    """
    from dipstick_analyzer import rgb_to_lab, delta_e_ab_only
    print("\n  Live calibration summary (a,b distance):")
    print(f"  {'Band':<8} {'Live RGB':<22} {'Ref RGB':<22} {'Δ(a,b)'}")
    print("  " + "-" * 62)
    for live, ref in zip(live_scale, hardcoded_scale):
        lr, lg, lb = live["rgb"]
        rr, rg, rb = ref["rgb"]
        live_lab = rgb_to_lab(lr, lg, lb)
        ref_lab  = rgb_to_lab(rr, rg, rb)
        de = delta_e_ab_only(live_lab, ref_lab)
        print(f"  {live['label']:<8} ({lr:3},{lg:3},{lb:3})          "
              f"({rr:3},{rg:3},{rb:3})          {de:.1f}")
    print()
