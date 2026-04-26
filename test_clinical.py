"""
test_clinical.py
----------------
Validation tests for the ClinicalGrade analyser pipeline.

Covers: Gray World output integrity, CIE76 inter-band discrimination,
correct classification of known samples under live calibration,
live calibration consistency with the hardcoded reference, white body
correction bounds, and preprocessing label recording.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from dipstick_analyzer import (
    rgb_to_lab, delta_e_cie76, delta_e_ab_only,
    classify_protein, PROTEIN_SCALE,
    _build_reference_lab, _per_band_thresholds,
)
from preprocessing import (
    gray_world_normalize, sample_white_body, apply_white_body_correction,
)
from strip_detector import analyze_strip_auto
from reference_calibrator import detect_reference_bands

SAMPLES_DIR   = os.path.join(os.path.dirname(__file__), "Samples")
REFERENCE_IMG = os.path.join(os.path.dirname(__file__), "IMG_8050.jpeg")

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if condition:
        passed += 1
    else:
        failed += 1


# ── Test 1: Gray World normalization ──────────────────────────────────────────
print("\nTest 1: Gray World normalization")
img = cv2.imread(REFERENCE_IMG)
if img is not None:
    norm = gray_world_normalize(img)
    check("Output shape preserved", norm.shape == img.shape)
    check("Output dtype preserved", norm.dtype == img.dtype)

    # Neutral-pixel Gray World balances the white background pixels, not the
    # whole image. The correct invariant is that neutral (low-saturation) pixels
    # become channel-balanced after normalisation.
    hsv_orig = cv2.cvtColor(img,  cv2.COLOR_BGR2HSV)
    hsv_norm = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    neutral_mask = (hsv_orig[:, :, 1] < 60) & (hsv_orig[:, :, 2] > 60)
    if neutral_mask.sum() > 100:
        orig_neutral = img.astype(np.float64)[neutral_mask]
        norm_neutral = norm.astype(np.float64)[neutral_mask]
        orig_spread  = float(max(orig_neutral.mean(axis=0)) - min(orig_neutral.mean(axis=0)))
        norm_spread  = float(max(norm_neutral.mean(axis=0)) - min(norm_neutral.mean(axis=0)))
        check(
            "Neutral-pixel channel imbalance reduced after normalization",
            norm_spread <= orig_spread,
            f"neutral spread {orig_spread:.1f} → {norm_spread:.1f}",
        )
    else:
        check("Neutral pixels available for balance check", False, "too few neutral pixels")
else:
    check("Reference image available", False, "IMG_8050.jpeg not found")


# ── Test 2: CIE76 inter-band gaps are discriminative ─────────────────────────
print("\nTest 2: Full Delta-E inter-band gaps are discriminative")
labs = [rgb_to_lab(*e["rgb"]) for e in PROTEIN_SCALE]
for i in range(len(labs) - 1):
    d_full = delta_e_cie76(labs[i], labs[i + 1])
    d_ab   = delta_e_ab_only(labs[i], labs[i + 1])
    check(
        f"{PROTEIN_SCALE[i]['label']} → {PROTEIN_SCALE[i+1]['label']} full gap > 5",
        d_full > 5.0,
        f"ΔE={d_full:.2f}  Δ(a,b)={d_ab:.2f}",
    )


# ── Test 3: Known samples classify correctly with live calibration ────────────
print("\nTest 3: Known sample images under live calibration")
# Live calibration is required for consistent results when reference and sample
# photographs are taken in different sessions. The hardcoded fallback uses a
# global threshold and will correctly report borderline readings as TRACE.
sample_expectations = [
    ("sample1_plus.jpeg", "+"),
    ("sample1_trace.PNG",  "TRACE"),
    ("sample2_plus.PNG",   "+"),
]
if os.path.isfile(REFERENCE_IMG):
    live_scale = detect_reference_bands(REFERENCE_IMG, use_gray_world=False)
    for filename, expected in sample_expectations:
        path = os.path.join(SAMPLES_DIR, filename)
        if not os.path.isfile(path):
            check(f"{filename}", False, "file not found")
            continue
        try:
            result, _ = analyze_strip_auto(
                path,
                reference_scale=live_scale,
                use_gray_world=False,
                use_white_body=False,
            )
            check(
                f"{filename} → {expected}",
                result.label == expected,
                f"got {result.label}  ΔE={result.delta_e}  cal={result.calibration_source}",
            )
        except Exception as e:
            check(f"{filename}", False, str(e))
else:
    check("Reference image available for Test 3", False, "IMG_8050.jpeg not found")


# ── Test 4: Live calibration returns 6 ordered bands ─────────────────────────
print("\nTest 4: Live calibration band detection and ordering")
if os.path.isfile(REFERENCE_IMG):
    # The app always calibrates the bundled reference without Gray World
    # (use_gray_world=False) so that the raw sensor colours of the chart
    # squares are captured; Gray World is applied only to sample images.
    live_scale = detect_reference_bands(REFERENCE_IMG, use_gray_world=False)
    if live_scale is not None:
        check("Live calibration returned 6 bands", len(live_scale) == 6)
        check("Labels ordered NEG → ++++",
              [b["label"] for b in live_scale] == ["NEG", "SP/TR", "+", "++", "+++", "++++"])
        # Full ΔE between live and hardcoded should be within a reasonable range
        for live, ref in zip(live_scale, PROTEIN_SCALE):
            ll = rgb_to_lab(*live["rgb"])
            rl = rgb_to_lab(*ref["rgb"])
            d  = delta_e_cie76(ll, rl)
            check(
                f"  {live['label']} full ΔE drift < 40",
                d < 40.0,
                f"ΔE = {d:.2f}",
            )
    else:
        check("Live calibration from IMG_8050.jpeg", False, "returned None")
else:
    check("Reference image available for Test 4", False, "IMG_8050.jpeg not found")


# ── Test 5: White body correction stays within ±40% bounds ───────────────────
print("\nTest 5: White body correction bounded within ±40%")
warm_white_bgr = (210.0, 215.0, 230.0)   # warm-tinted strip backing
test_rgb       = (129.0, 172.0, 112.0)   # approximate "+" pad colour
corrected = apply_white_body_correction(test_rgb, warm_white_bgr, true_white=238.0)
channel_names = ["R", "G", "B"]
for ch, orig, corr in zip(channel_names, test_rgb, corrected):
    ratio = corr / orig if orig > 0 else 1.0
    check(
        f"  {ch} channel correction within 0.6–1.4×",
        0.6 <= ratio <= 1.4,
        f"{orig:.0f} → {corr:.0f}  ratio={ratio:.2f}",
    )


# ── Test 6: Preprocessing labels are recorded correctly ──────────────────────
print("\nTest 6: Preprocessing label recorded in result")
path = os.path.join(SAMPLES_DIR, "sample1_plus.jpeg")
if os.path.isfile(path):
    r_gw, _   = analyze_strip_auto(path, use_gray_world=True,  use_white_body=False)
    r_gw_wb,_ = analyze_strip_auto(path, use_gray_world=True,  use_white_body=True)
    r_none, _ = analyze_strip_auto(path, use_gray_world=False, use_white_body=False)
    check("Gray World label recorded",
          "gray_world" in r_gw.preprocessing, r_gw.preprocessing)
    check("Both steps recorded when both enabled",
          "gray_world" in r_gw_wb.preprocessing and "white_body" in r_gw_wb.preprocessing,
          r_gw_wb.preprocessing)
    check("No preprocessing recorded when disabled",
          r_none.preprocessing == "none", r_none.preprocessing)
else:
    check("Sample image available for label test", False, "sample1_plus.jpeg not found")


# ── Test 7: a,b-only function is available and numerically correct ────────────
print("\nTest 7: a,b-only distance function")
# Two colours differing only in L should have zero a,b distance
lab_bright = (80.0, -20.0, 25.0)
lab_dim    = (40.0, -20.0, 25.0)
d_ab   = delta_e_ab_only(lab_bright, lab_dim)
d_full = delta_e_cie76(lab_bright, lab_dim)
check("Same hue, different L: a,b distance = 0", d_ab == 0.0, f"Δ(a,b)={d_ab}")
check("Same hue, different L: full ΔE > 0", d_full > 0.0, f"ΔE={d_full:.1f}")

# Two colours differing only in a,b should give equal a,b and full distances
lab_a = (60.0, -20.0, 25.0)
lab_b = (60.0, -10.0, 15.0)
d_ab2   = delta_e_ab_only(lab_a, lab_b)
d_full2 = delta_e_cie76(lab_a, lab_b)
check("Same L, different a,b: distances equal", abs(d_ab2 - d_full2) < 0.001,
      f"Δ(a,b)={d_ab2:.3f}  ΔE={d_full2:.3f}")


# ── Summary ───────────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n  {total} tests — {passed} passed, {failed} failed\n")
sys.exit(0 if failed == 0 else 1)
