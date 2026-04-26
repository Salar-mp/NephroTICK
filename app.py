"""
app.py — NephroTICK Web Interface
----------------------------------
Run with:  streamlit run app.py

Fully automatic pipeline:
  1. Calibrates from the bundled reference photo (IMG_8050.jpeg) at startup.
  2. Applies Gray World normalization to every image.
  3. Detects the protein pad, extracts its colour, and classifies it.

No settings to configure. Upload or photograph the strip and press Analyse.
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NephroTICK", layout="centered")

st.markdown("""
<style>
  .result-card  { padding:1.6rem 2rem; border-radius:12px; margin:1.2rem 0; }
  .result-safe  { background:#eafaf1; border-left:6px solid #27ae60; }
  .result-flag  { background:#fdedec; border-left:6px solid #e74c3c; }
  .result-label { font-size:2.6rem; font-weight:700; margin:0; }
  .result-desc  { font-size:1.05rem; margin:0.4rem 0 0 0; color:#444; }
  .flag-text    { font-size:1rem; font-weight:600; color:#c0392b; margin-top:0.6rem; }
  .cal-badge    { font-size:0.8rem; color:#888; margin-top:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Import analyser modules ───────────────────────────────────────────────────
try:
    from strip_detector import analyze_strip_auto, annotate_result
    from reference_calibrator import detect_reference_bands
    from dipstick_analyzer import PROTEIN_SCALE
    from preprocessing import check_blur
    modules_ok = True
except ImportError as e:
    modules_ok = False
    import_error = str(e)

APP_DIR     = os.path.dirname(os.path.abspath(__file__))
BUNDLED_REF = os.path.join(APP_DIR, "IMG_8050.jpeg")


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_to_temp(image_data, suffix=".jpg") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(image_data.getvalue() if hasattr(image_data, "getvalue") else image_data.read())
    tmp.flush()
    return tmp.name


def annotate_to_pil(image_path, result, bbox) -> Image.Image:
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix="_ann.jpg").name
    annotate_result(image_path, result, bbox, out_path=out_path)
    img = cv2.cvtColor(cv2.imread(out_path), cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


# ── Auto-calibrate once at startup ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_calibration(ref_path: str):
    if not os.path.isfile(ref_path):
        return None
    try:
        return detect_reference_bands(ref_path, use_gray_world=False)
    except Exception:
        return None


# ── Header ────────────────────────────────────────────────────────────────────
st.title("NephroTICK")
st.caption("Albustix urine dipstick protein classifier")
st.divider()

if not modules_ok:
    st.error(
        f"Could not import analysis modules: `{import_error}`\n\n"
        "Run from the **ClinicalGrade** folder:\n"
        "```\ncd ClinicalGrade\nstreamlit run app.py\n```"
    )
    st.stop()

# Load calibration — always from bundled reference unless overridden in Advanced
bundled_scale = load_calibration(BUNDLED_REF)


# ── Sidebar: status + advanced override only ──────────────────────────────────
with st.sidebar:
    st.header("Status")

    if bundled_scale:
        st.success("Reference calibration active", icon="✅")
    else:
        st.warning("Bundled reference not found.\nUsing hardcoded scale.")

    st.caption(
        "**Pipeline (automatic)**\n\n"
        "Gray World normalization  →  pad detection  →  colour extraction  →  classification"
    )

    st.divider()

    # Advanced override — hidden away for supervisor/research use
    with st.expander("Advanced: use a different reference photo"):
        st.caption(
            "Photograph the bottle colour chart under the same lighting as your strip. "
            "This gives the most accurate results in your specific conditions."
        )
        ref_method = st.radio("Source", ["Upload file", "Camera"], key="ref_src")
        custom_ref_path = None
        if ref_method == "Upload file":
            ref_up = st.file_uploader(
                "Bottle reference photo", type=["jpg", "jpeg", "png"],
                key="ref_upload", label_visibility="collapsed"
            )
            if ref_up:
                custom_ref_path = save_to_temp(ref_up, suffix=".jpg")
        else:
            ref_cam = st.camera_input("Photograph the bottle", key="ref_cam")
            if ref_cam:
                custom_ref_path = save_to_temp(ref_cam, suffix=".jpg")

    st.divider()
    st.caption("NephroTICK — Group Project")


# ── Strip image input ─────────────────────────────────────────────────────────
st.subheader("Photograph the strip")
st.caption(
    "Place the strip on a plain white background. "
    "Use the phone flashlight and keep the camera steady."
)

tab_upload, tab_camera = st.tabs(["Upload a photo", "Use camera"])

strip_path = None

with tab_upload:
    strip_up = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"],
        key="strip_upload", label_visibility="collapsed"
    )
    if strip_up:
        tmp_path = save_to_temp(strip_up, suffix=".jpg")
        st.image(strip_up, caption="Uploaded strip", use_container_width=True)
        blurry, variance = check_blur(tmp_path)
        if blurry:
            st.warning(
                f"⚠️ Image looks blurry (sharpness score: {variance:.0f} — aim for >80). "
                "Results may be less accurate — consider retaking with better focus."
            )
        strip_path = tmp_path  # still allow analysis for uploads (user can't retake easily)

with tab_camera:
    strip_cam = st.camera_input("Take a photo", key="strip_cam", label_visibility="collapsed")

    # ── ROI overlay injected onto the live camera video element ───────────────
    components.html("""
    <script>
    (function () {
      /*
       * Two overlays:
       *
       * 1. PAD TARGET (inner solid box) — fixed physical size.
       *    The Albustix protein pad is ~6 mm × 6 mm.
       *    We measure the device's CSS pixel-per-mm ratio via a hidden div,
       *    then scale so the box represents ~6 mm × 6 mm on screen.
       *    At the correct distance (~20 cm), the green pad on the strip will
       *    exactly fill this box. Move closer if it looks smaller; farther if larger.
       *
       * 2. STRIP GUIDE (outer dashed box) — 70 % × 55 % of frame.
       *    Ensures the full strip (and reference colour chart) stays in frame.
       */

      // Physical size of the protein pad in mm
      const PAD_MM = 6;

      function getPxPerMm(doc) {
        const el = doc.createElement('div');
        el.style.cssText = 'width:10mm;height:1px;position:absolute;visibility:hidden;';
        doc.body.appendChild(el);
        const px = el.getBoundingClientRect().width / 10;
        doc.body.removeChild(el);
        return px > 0 ? px : 3.78;  // fallback: 96dpi → 3.78px/mm
      }

      function drawShadowText(ctx, text, x, y) {
        ctx.shadowColor = 'rgba(0,0,0,0.65)';
        ctx.shadowBlur  = 4;
        ctx.fillText(text, x, y);
        ctx.shadowBlur  = 0;
      }

      function tryInject() {
        try {
          const doc   = window.parent.document;
          const video = doc.querySelector('video');
          if (!video) { setTimeout(tryInject, 300); return; }
          if (doc.getElementById('nt-roi')) return;

          const wrap = video.parentElement;
          if (getComputedStyle(wrap).position === 'static') wrap.style.position = 'relative';

          const cv = doc.createElement('canvas');
          cv.id = 'nt-roi';
          cv.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:10;';
          wrap.appendChild(cv);

          const pxPerMm = getPxPerMm(doc);
          const PAD_PX  = PAD_MM * pxPerMm;   // physical pad size in CSS pixels

          function draw() {
            const vw = video.offsetWidth, vh = video.offsetHeight;
            if (!vw || !vh) { requestAnimationFrame(draw); return; }
            if (cv.width !== vw || cv.height !== vh) { cv.width = vw; cv.height = vh; }

            const ctx = cv.getContext('2d');
            ctx.clearRect(0, 0, vw, vh);
            ctx.lineCap = 'round';

            // ── 1. Outer strip guide (dashed green) ──────────────────────────
            const ox = vw * 0.15, oy = vh * 0.20;
            const ow = vw * 0.70, oh = vh * 0.55;
            ctx.strokeStyle = 'rgba(39,210,90,0.55)';
            ctx.lineWidth   = 2;
            ctx.setLineDash([8, 5]);
            ctx.strokeRect(ox, oy, ow, oh);
            ctx.setLineDash([]);

            // Corner brackets on outer box
            const cs = Math.min(ow, oh) * 0.10;
            ctx.strokeStyle = 'rgba(39,210,90,0.75)';
            ctx.lineWidth   = 3;
            [[ox,ow+ox,oy,oh+oy]].forEach(([x0,x1,y0,y1]) => {
              [[x0,y0,+cs,0,0,+cs],[x1,y0,-cs,0,0,+cs],
               [x0,y1,+cs,0,0,-cs],[x1,y1,-cs,0,0,-cs]].forEach(([x,y,d1,_d2,d3,d4])=>{
                ctx.beginPath();
                ctx.moveTo(x+d1, y);
                ctx.lineTo(x, y);
                ctx.lineTo(x, y+d4);
                ctx.stroke();
              });
            });

            const fs = Math.max(10, Math.round(vw * 0.030));
            ctx.font      = `500 ${fs}px Arial, sans-serif`;
            ctx.fillStyle = 'rgba(39,210,90,0.80)';
            ctx.textAlign = 'center';
            drawShadowText(ctx, 'Keep full strip + reference chart inside', vw/2, oy - 6);

            // ── 2. Inner pad target (solid white box, pad-sized) ─────────────
            const px = (vw - PAD_PX) / 2;
            const py = (vh - PAD_PX) / 2;

            // Bright fill so the box stands out on any background
            ctx.fillStyle = 'rgba(255,255,255,0.12)';
            ctx.fillRect(px, py, PAD_PX, PAD_PX);

            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth   = 2;
            ctx.strokeRect(px, py, PAD_PX, PAD_PX);

            // Coloured inner border
            ctx.strokeStyle = '#ffdd00';
            ctx.lineWidth   = 1.5;
            const inset = 3;
            ctx.strokeRect(px+inset, py+inset, PAD_PX-2*inset, PAD_PX-2*inset);

            // Corner ticks on inner box
            const tc = Math.min(PAD_PX * 0.35, 14);
            ctx.strokeStyle = '#ffee44';
            ctx.lineWidth   = 2.5;
            [[px,py,+tc,0,0,+tc],[px+PAD_PX,py,-tc,0,0,+tc],
             [px,py+PAD_PX,+tc,0,0,-tc],[px+PAD_PX,py+PAD_PX,-tc,0,0,-tc]]
            .forEach(([x,y,dx,_,dz,dw])=>{
              ctx.beginPath();
              ctx.moveTo(x+dx,y); ctx.lineTo(x,y); ctx.lineTo(x,y+dw);
              ctx.stroke();
            });

            // Labels on inner box
            const fsP = Math.max(9, Math.round(vw * 0.028));
            ctx.font      = `700 ${fsP}px Arial, sans-serif`;
            ctx.fillStyle = '#ffee44';
            ctx.textAlign = 'center';
            drawShadowText(ctx, 'Protein pad', vw/2, py - 6);

            ctx.font      = `500 ${Math.max(8, fsP-1)}px Arial, sans-serif`;
            ctx.fillStyle = 'rgba(255,238,100,0.85)';
            drawShadowText(ctx, '← closer   farther →', vw/2, py + PAD_PX + fsP + 4);

            requestAnimationFrame(draw);
          }
          draw();
        } catch (e) {
          console.warn('NephroTICK ROI overlay:', e);
        }
      }
      tryInject();
    })();
    </script>
    """, height=0)
    if strip_cam:
        tmp_path = save_to_temp(strip_cam, suffix=".jpg")
        # ── Blur check ────────────────────────────────────────────────────────
        blurry, variance = check_blur(tmp_path)
        if blurry:
            st.warning(
                f"⚠️ Image looks blurry (sharpness score: {variance:.0f} — aim for >80). "
                "Hold the camera steady and tap the subject to focus, then retake."
            )
            # Don't set strip_path — forces the user to retake
        else:
            strip_path = tmp_path

st.divider()

# ── Analyse button ────────────────────────────────────────────────────────────
analyse_clicked = st.button(
    "Analyse", type="primary",
    disabled=(strip_path is None),
    use_container_width=True
)

if analyse_clicked and strip_path:

    # ── Resolve reference scale ───────────────────────────────────────────────
    reference_scale = bundled_scale  # default: bundled IMG_8050.jpeg

    try:
        custom_ref_path
    except NameError:
        custom_ref_path = None

    if custom_ref_path:
        with st.spinner("Calibrating from your reference photo..."):
            try:
                custom_scale = detect_reference_bands(custom_ref_path, use_gray_world=False)
                if custom_scale:
                    reference_scale = custom_scale
                    st.success("Calibrated from your reference photo.")
                else:
                    st.warning("Could not read your reference photo — using bundled calibration.")
            except Exception as e:
                st.warning(f"Calibration error ({e}) — using bundled calibration.")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    with st.spinner("Analysing..."):
        try:
            result, bbox = analyze_strip_auto(
                strip_path,
                reference_scale=reference_scale,
                use_gray_world=True,
                use_white_body=False,
            )
        except FileNotFoundError as e:
            st.error(f"Could not open image: {e}")
            st.stop()
        except RuntimeError as e:
            st.error(
                str(e) + "\n\n**Tips:**\n"
                "- Place the strip on a plain white background\n"
                "- Ensure the protein pad is clearly visible\n"
                "- Use the flashlight for consistent lighting"
            )
            st.stop()

    # ── Result ────────────────────────────────────────────────────────────────
    card_cls  = "result-flag" if result.clinician_flag else "result-safe"
    flag_html = '<p class="flag-text">⚑ Flag for clinician review</p>' if result.clinician_flag else ""

    st.markdown(f"""
    <div class="result-card {card_cls}">
      <p class="result-label">{result.label}</p>
      <p class="result-desc">{result.description}</p>
      {flag_html}
      <p class="cal-badge">ΔE {result.delta_e:.2f} &nbsp;·&nbsp; {result.calibration_source} calibration</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Annotated image ───────────────────────────────────────────────────────
    try:
        st.image(
            annotate_to_pil(strip_path, result, bbox),
            caption="Detected pad region",
            use_container_width=True
        )
    except Exception:
        pass

    # ── Scale reference ───────────────────────────────────────────────────────
    with st.expander("Protein scale reference"):
        for entry in PROTEIN_SCALE:
            marker = "  ◀ current result" if entry["label"] == result.label else ""
            st.markdown(f"**{entry['label']}** — {entry['description']}{marker}")

    # ── Technical details (for research/validation) ───────────────────────────
    with st.expander("Technical details"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Delta-E", f"{result.delta_e:.2f}")
        c2.metric("Calibration", result.calibration_source)
        c3.metric("Preprocessing", result.preprocessing)
        st.caption(f"Sampled RGB: {result.sampled_rgb}")
