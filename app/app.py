import os
import sys
import io
import base64
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.processor import GeoSRProcessor, compute_sharpness_metrics
from api.satellite_fetcher import query_sentinel2_stac, stream_sentinel_crop, CITY_COORDINATES

PRESETS = {
    "stadium_airport": {
        "title": "Safdarjung Runway and JLN Stadium",
        "category": "Urban Infrastructure and Aviation",
        "desc": "Safdarjung airport runway, Jawaharlal Nehru Stadium arena, circular traffic roundabouts, and tree-lined arterial corridors.",
        "input": ROOT / "data" / "presets" / "stadium_airport.png",
        "sr": ROOT / "outputs" / "enhanced" / "stadium_airport_geosr_residual_4x.png",
        "bicubic": ROOT / "outputs" / "enhanced" / "stadium_airport_bicubic_4x.png",
    },
    "urban_core": {
        "title": "Metropolitan Urban Core and Flyovers",
        "category": "High-Density Built-up",
        "desc": "High-density residential grids, multi-level highway interchanges, rail corridors, and commercial building blocks.",
        "input": ROOT / "data" / "presets" / "urban_core.png",
        "sr": ROOT / "outputs" / "enhanced" / "urban_core_geosr_residual_4x.png",
        "bicubic": ROOT / "outputs" / "enhanced" / "urban_core_bicubic_4x.png",
    },
    "yamuna_river": {
        "title": "Yamuna River Corridor and Bridges",
        "category": "Hydrological and Riparian",
        "desc": "Meandering river channel, active floodplain boundaries, riparian vegetation, and highway bridge crossings.",
        "input": ROOT / "data" / "presets" / "yamuna_river.png",
        "sr": ROOT / "outputs" / "enhanced" / "yamuna_river_geosr_residual_4x.png",
        "bicubic": ROOT / "outputs" / "enhanced" / "yamuna_river_bicubic_4x.png",
    },
    "agricultural_grid": {
        "title": "Agricultural Parcel Grid",
        "category": "Agro-Ecological and Rural",
        "desc": "Geometric field boundary networks, agricultural crop canopies, and rural irrigation channels.",
        "input": ROOT / "data" / "presets" / "agricultural_grid.png",
        "sr": ROOT / "outputs" / "enhanced" / "agricultural_grid_geosr_residual_4x.png",
        "bicubic": ROOT / "outputs" / "enhanced" / "agricultural_grid_bicubic_4x.png",
    },
}


@st.cache_resource
def get_processor():
    return GeoSRProcessor()


def pil_to_base64(img: Image.Image, fmt="JPEG", quality=92) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def render_curtain_slider(img_left: Image.Image, img_right: Image.Image,
                          label_left="GeoSR 4× residual reconstruction", label_right="Bicubic 4× baseline"):
    """Renders an interactive curtain wipe slider with hover-to-zoom."""
    b64_left = pil_to_base64(img_left)
    b64_right = pil_to_base64(img_right)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .slider-wrap {{
            position: relative; width: 100%; height: 580px; overflow: hidden;
            border-radius: 6px; border: 1px solid #334155;
            user-select: none; cursor: ew-resize; background: #0f172a;
        }}
        .img-layer {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            object-fit: contain; pointer-events: none;
        }}
        .left-layer {{ clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%); }}
        .curtain {{
            position: absolute; top: 0; bottom: 0; left: 50%; width: 2px;
            background: #ffffff; box-shadow: 0 0 10px rgba(0,0,0,0.6); z-index: 10;
        }}
        .handle {{
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            width: 36px; height: 36px; border-radius: 50%; background: #fff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.4); display: flex;
            align-items: center; justify-content: center;
            font-weight: 700; font-size: 14px; color: #0f172a;
        }}
        .badge {{
            position: absolute; top: 10px; padding: 4px 10px; border-radius: 3px;
            font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
            text-transform: uppercase; z-index: 8;
        }}
        .badge-l {{ left: 10px; background: rgba(2,132,199,0.9); color: #fff; }}
        .badge-r {{ right: 10px; background: rgba(51,65,85,0.88); color: #cbd5e1; }}
        /* Zoom lens */
        .zoom-lens {{
            display: none; position: absolute; width: 180px; height: 180px;
            border: 2px solid #38bdf8; border-radius: 50%; pointer-events: none;
            z-index: 20; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.5);
        }}
        .zoom-lens img {{
            position: absolute; pointer-events: none;
        }}
    </style>
    </head>
    <body>
        <div class="slider-wrap" id="ctr">
            <img class="img-layer" id="imgR" src="data:image/jpeg;base64,{b64_right}" alt="Baseline">
            <img class="img-layer left-layer" id="imgL" src="data:image/jpeg;base64,{b64_left}" alt="Enhanced">
            <div class="badge badge-l">{label_left}</div>
            <div class="badge badge-r">{label_right}</div>
            <div class="curtain" id="line"><div class="handle">&harr;</div></div>
            <div class="zoom-lens" id="lens"><img id="lensImg" src="data:image/jpeg;base64,{b64_left}"></div>
        </div>
        <p style="text-align:center;color:#94a3b8;font-size:12px;margin-top:6px;">
            Drag divider to compare | Hold Shift + hover to magnify (3x zoom)
        </p>
        <script>
            const ctr=document.getElementById('ctr'), imgL=document.getElementById('imgL'),
                  line=document.getElementById('line'), lens=document.getElementById('lens'),
                  lensImg=document.getElementById('lensImg');
            let dragging=false, shiftHeld=false;

            function setPos(cx) {{
                const r=ctr.getBoundingClientRect();
                let pct=Math.max(1,Math.min(99,((cx-r.left)/r.width)*100));
                imgL.style.clipPath=`polygon(0 0,${{pct}}% 0,${{pct}}% 100%,0 100%)`;
                line.style.left=pct+'%';
            }}

            ctr.addEventListener('mousedown',e=>{{dragging=true;setPos(e.clientX);}});
            window.addEventListener('mouseup',()=>{{dragging=false;}});
            window.addEventListener('mousemove',e=>{{
                if(dragging)setPos(e.clientX);
                shiftHeld=e.shiftKey;
                if(shiftHeld){{
                    const r=ctr.getBoundingClientRect();
                    const mx=e.clientX-r.left, my=e.clientY-r.top;
                    lens.style.display='block';
                    lens.style.left=(mx-90)+'px'; lens.style.top=(my-90)+'px';
                    const zoom=3;
                    const natW=imgL.naturalWidth, natH=imgL.naturalHeight;
                    const scaleX=natW/r.width, scaleY=natH/r.height;
                    lensImg.style.width=(r.width*zoom)+'px'; lensImg.style.height=(r.height*zoom)+'px';
                    lensImg.style.left=(-mx*zoom+90)+'px'; lensImg.style.top=(-my*zoom+90)+'px';
                }} else {{ lens.style.display='none'; }}
            }});
            ctr.addEventListener('mouseleave',()=>{{lens.style.display='none';}});

            ctr.addEventListener('touchstart',e=>{{dragging=true;if(e.touches.length)setPos(e.touches[0].clientX);}});
            window.addEventListener('touchend',()=>{{dragging=false;}});
            window.addEventListener('touchmove',e=>{{if(dragging&&e.touches.length)setPos(e.touches[0].clientX);}});
        </script>
    </body>
    </html>
    """
    components.html(html, height=620)


def render_zoom_comparison(img_left: Image.Image, img_right: Image.Image,
                           label_left="Sentinel-2 Baseline", label_right="GeoSR Enhanced"):
    """Interactive side-by-side with synchronized hover-to-zoom magnifier."""
    b64_l = pil_to_base64(img_left)
    b64_r = pil_to_base64(img_right)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ background:transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .grid {{ display:flex; gap:8px; width:100%; }}
        .panel {{
            flex:1; position:relative; overflow:hidden; border-radius:6px;
            border:1px solid #334155; background:#0f172a; height:560px;
        }}
        .panel img {{ width:100%; height:100%; object-fit:contain; display:block; }}
        .panel-label {{
            position:absolute; top:8px; left:8px; padding:3px 8px; border-radius:3px;
            font-size:11px; font-weight:600; text-transform:uppercase;
            background:rgba(15,23,42,0.85); color:#94a3b8; letter-spacing:0.03em;
            z-index:5;
        }}
        .magnifier {{
            display:none; position:absolute; width:200px; height:200px;
            border:2px solid #38bdf8; border-radius:6px; pointer-events:none;
            z-index:10; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.5);
        }}
        .magnifier img {{ position:absolute; pointer-events:none; }}
        .zoom-label {{
            position:absolute; bottom:4px; right:4px; font-size:10px; color:#38bdf8;
            background:rgba(0,0,0,0.7); padding:1px 5px; border-radius:2px; z-index:11;
        }}
    </style>
    </head>
    <body>
        <div class="grid">
            <div class="panel" id="panelL">
                <div class="panel-label">{label_left}</div>
                <img id="srcL" src="data:image/jpeg;base64,{b64_l}">
                <div class="magnifier" id="magL"><img id="magImgL" src="data:image/jpeg;base64,{b64_l}"><div class="zoom-label">4x</div></div>
            </div>
            <div class="panel" id="panelR">
                <div class="panel-label">{label_right}</div>
                <img id="srcR" src="data:image/jpeg;base64,{b64_r}">
                <div class="magnifier" id="magR"><img id="magImgR" src="data:image/jpeg;base64,{b64_r}"><div class="zoom-label">4x</div></div>
            </div>
        </div>
        <p style="text-align:center;color:#94a3b8;font-size:12px;margin-top:6px;">
            Hover over either panel to magnify both views simultaneously (4x zoom)
        </p>
        <script>
            const zoom=4;
            function setupSync(panelId, magId, magImgId, otherMagId, otherMagImgId, otherPanelId) {{
                const panel=document.getElementById(panelId);
                const mag=document.getElementById(magId), magImg=document.getElementById(magImgId);
                const oMag=document.getElementById(otherMagId), oMagImg=document.getElementById(otherMagImgId);
                const oPanel=document.getElementById(otherPanelId);

                panel.addEventListener('mousemove', e => {{
                    const r=panel.getBoundingClientRect();
                    const or2=oPanel.getBoundingClientRect();
                    const mx=e.clientX-r.left, my=e.clientY-r.top;
                    const px=mx/r.width, py=my/r.height;

                    [mag,oMag].forEach(m=>{{m.style.display='block';}});

                    mag.style.left=(mx-100)+'px'; mag.style.top=(my-100)+'px';
                    magImg.style.width=(r.width*zoom)+'px'; magImg.style.height=(r.height*zoom)+'px';
                    magImg.style.left=(-mx*zoom+100)+'px'; magImg.style.top=(-my*zoom+100)+'px';

                    const omx=px*or2.width, omy=py*or2.height;
                    oMag.style.left=(omx-100)+'px'; oMag.style.top=(omy-100)+'px';
                    oMagImg.style.width=(or2.width*zoom)+'px'; oMagImg.style.height=(or2.height*zoom)+'px';
                    oMagImg.style.left=(-omx*zoom+100)+'px'; oMagImg.style.top=(-omy*zoom+100)+'px';
                }});
                panel.addEventListener('mouseleave', ()=>{{
                    [mag,oMag].forEach(m=>{{m.style.display='none';}});
                }});
            }}
            setupSync('panelL','magL','magImgL','magR','magImgR','panelR');
            setupSync('panelR','magR','magImgR','magL','magImgL','panelL');
        </script>
    </body>
    </html>
    """
    components.html(html, height=600)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GeoSR | Satellite Super-Resolution Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1300px;
        padding-top: 1.5rem;
    }
    .header-bar {
        padding: 1rem 1.4rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 1.2rem;
    }
    .h-title {
        font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em;
        color: #0f172a; margin-bottom: 0.15rem;
    }
    .h-desc { font-size: 0.88rem; color: #475569; margin-bottom: 0.5rem; }
    .spec-tag {
        display: inline-block; padding: 2px 8px; font-size: 0.72rem;
        font-weight: 600; border-radius: 4px; margin-right: 5px;
        background: #e2e8f0; color: #334155; border: 1px solid #cbd5e1;
    }
    .kpi-card {
        padding: 0.85rem 1rem; background: #f8fafc;
        border: 1px solid #e2e8f0; border-radius: 6px; text-align: center;
    }
    .kpi-val { font-size: 1.3rem; font-weight: 700; color: #0284c7; }
    .kpi-lbl {
        font-size: 0.7rem; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.04em; margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-bar">
    <div class="h-title">GeoSR | Satellite Super-Resolution Platform</div>
    <div class="h-desc">4× AI-assisted visual enhancement for Sentinel-2 RGB imagery</div>
    <div>
        <span class="spec-tag">MISSION: COPERNICUS SENTINEL-2 MSI</span>
        <span class="spec-tag">ENGINE: DETAILEDSR (4X)</span>
        <span class="spec-tag">OUTPUT: 4× PIXEL GRID</span>
        <span class="spec-tag">MODE: BICUBIC + LEARNED RESIDUAL</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Data Source")
    data_source = st.radio(
        "Select Pipeline Mode:",
        [
            "Live Copernicus Satellite Stream",
            "Regional Benchmark Presets",
            "Local File Upload"
        ],
        index=1
    )

    st.markdown("---")
    st.markdown("### Display adjustment")
    enh_strength = st.select_slider(
        "Optional cosmetic adjustment",
        options=["none", "gentle"],
        value="none",
        help="Off preserves the neural model output. Gentle only adjusts display contrast; it does not recover geographic detail."
    )

    st.markdown("---")
    st.markdown("### Platform Details")
    st.markdown("""
    - **Constellation:** Sentinel-2A / 2B / 2C
    - **Input:** Sentinel-2 RGB or local raster imagery
    - **Output:** 4× enlarged pixel grid
    - **Method:** Bicubic base plus learned EDSR residual
    - **Baseline:** Bicubic interpolation
    - **Note:** Output is AI-enhanced imagery, not verified new ground detail.
    """)

# ============================================================
# DATA INGESTION
# ============================================================

sr_img = None
bi_img = None
in_img = None
source_label = ""

if data_source == "Live Copernicus Satellite Stream":
    st.markdown("#### Live Copernicus Sentinel-2 Stream")
    st.caption("Query the open Sentinel-2 L2A catalog for any location on Earth. Enter coordinates or select a preset city.")

    input_mode = st.radio("Location Input:", ["Preset City", "Custom Coordinates"], horizontal=True)

    col_q1, col_q2 = st.columns([2, 1])
    if input_mode == "Preset City":
        with col_q1:
            city_sel = st.selectbox("Target Location:", list(CITY_COORDINATES.keys()))
            coords = CITY_COORDINATES[city_sel]
            lat_val, lon_val = coords["lat"], coords["lon"]
        with col_q2:
            max_cloud = st.slider("Max Cloud Cover (%)", 1, 30, 10)
    else:
        with col_q1:
            c1, c2 = st.columns(2)
            with c1:
                lat_val = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.585, step=0.01, format="%.4f")
            with c2:
                lon_val = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.205, step=0.01, format="%.4f")
        with col_q2:
            max_cloud = st.slider("Max Cloud Cover (%)", 1, 30, 10)

    if st.button("Query Sentinel-2 and Super-Resolve"):
        with st.spinner("Connecting to Sentinel-2 STAC catalog..."):
            try:
                meta = query_sentinel2_stac(lon_val, lat_val, max_cloud=max_cloud)
                st.success(f"Scene acquired: {meta['id']} | Date: {meta['datetime'][:10]} | Cloud: {meta['cloud_cover']}%")
                in_img = stream_sentinel_crop(meta["visual_url"], crop_size=400)
                source_label = f"Live Stream ({lat_val:.3f}, {lon_val:.3f})"

                proc = get_processor()
                in_np = np.asarray(in_img, dtype=np.float32) / 255.0
                res = proc.enhance_rgb_array(in_np, strength=enh_strength)

                sr_img = Image.fromarray((res["sr"] * 255.0).astype(np.uint8))
                bi_img = Image.fromarray((res["bicubic"] * 255.0).astype(np.uint8))
            except Exception as err:
                st.error(f"Satellite ingestion failed: {err}")
                st.stop()
    else:
        st.info("Enter any latitude/longitude on Earth or pick a city, then click 'Query Sentinel-2 and Super-Resolve'.")
        st.stop()

elif data_source == "Regional Benchmark Presets":
    preset_id = st.selectbox(
        "Select Regional Landmark (Delhi NCR Sentinel-2 Granule):",
        options=list(PRESETS.keys()),
        format_func=lambda k: PRESETS[k]["title"]
    )
    p = PRESETS[preset_id]
    st.caption(f"**{p['category']}** | {p['desc']}")
    source_label = p["title"]

    if p["sr"].exists() and p["bicubic"].exists():
        sr_img = Image.open(p["sr"]).convert("RGB")
        bi_img = Image.open(p["bicubic"]).convert("RGB")
        in_img = Image.open(p["input"]).convert("RGB")
    else:
        with st.spinner("Processing 4x Super-Resolution..."):
            proc = get_processor()
            res = proc.enhance_file(p["input"], strength=enh_strength)
            sr_img = Image.fromarray((res["sr"] * 255.0).astype(np.uint8))
            bi_img = Image.fromarray((res["bicubic"] * 255.0).astype(np.uint8))
            in_img = Image.open(p["input"]).convert("RGB")

else:
    st.markdown("#### Local File Upload")
    uploaded = st.file_uploader(
        "Upload satellite imagery (GeoTIFF, JP2, PNG, JPEG):",
        type=["tif", "tiff", "jp2", "png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        temp_dir = ROOT / "outputs" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded.name
        with open(temp_path, "wb") as f:
            f.write(uploaded.getvalue())
        with st.spinner("Processing through 4x Super-Resolution engine..."):
            proc = get_processor()
            res = proc.enhance_file(temp_path, strength=enh_strength)
            sr_img = Image.fromarray((res["sr"] * 255.0).astype(np.uint8))
            bi_img = Image.fromarray((res["bicubic"] * 255.0).astype(np.uint8))
            in_img = Image.open(temp_path).convert("RGB")
            source_label = uploaded.name
    else:
        st.info("Upload a file above, or switch to 'Regional Benchmark Presets' or 'Live Copernicus Satellite Stream'.")
        st.stop()

# ============================================================
# RESULTS
# ============================================================

if sr_img is not None and bi_img is not None:
    sr_np = np.asarray(sr_img, dtype=np.float32) / 255.0
    bi_np = np.asarray(bi_img, dtype=np.float32) / 255.0
    in_np = np.asarray(in_img, dtype=np.float32) / 255.0

    m_in = compute_sharpness_metrics(in_np)
    m_bi = compute_sharpness_metrics(bi_np)
    m_sr = compute_sharpness_metrics(sr_np)

    lap_gain = ((m_sr["laplacian_variance"] - m_bi["laplacian_variance"]) / max(1e-3, m_bi["laplacian_variance"])) * 100.0
    grad_gain = ((m_sr["gradient_energy"] - m_bi["gradient_energy"]) / max(1e-3, m_bi["gradient_energy"])) * 100.0

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi-card"><div class="kpi-val">4×</div><div class="kpi-lbl">Pixel-grid scale</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">16x</div><div class="kpi-lbl">Pixel Density ({in_img.width}&sup2; &rarr; {sr_img.width}&sup2;)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{m_sr["laplacian_variance"]:.1f}</div><div class="kpi-lbl">Output sharpness statistic</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val">{m_sr["gradient_energy"]:.1f}</div><div class="kpi-lbl">Output edge statistic</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Tabs
    tab_curtain, tab_zoom, tab_metrics = st.tabs([
        "Curtain Split Comparison",
        "Synchronized Zoom Comparison",
        "Quantitative Analytics"
    ])

    with tab_curtain:
        st.caption("Drag the vertical divider to compare. Hold Shift and hover to magnify (3x zoom lens).")
        render_curtain_slider(sr_img, bi_img)

    with tab_zoom:
        st.caption("Hover over either panel to magnify both views simultaneously at 4x. Move your cursor to inspect roads, rooftops, and field boundaries.")
        render_zoom_comparison(bi_img, sr_img)

    with tab_metrics:
        st.markdown("#### Image statistics")
        st.caption("Sharpness and edge statistics describe pixel contrast, not verified spatial accuracy.")
        st.table({
            "Metric": [
                "Pixel-grid scale",
                "Method",
                "Total Pixels",
                "Laplacian Variance (Sharpness)",
                "Gradient Energy (Edge Density)",
            ],
            "Sentinel-2 Baseline (10m)": [
                "4× bicubic",
                "Interpolation baseline",
                f"{in_img.width} x {in_img.height} ({in_img.width*in_img.height:,} px)",
                f"{m_bi['laplacian_variance']:.2f}",
                f"{m_bi['gradient_energy']:.2f}",
            ],
            "GeoSR model output": [
                "4× model output",
                "Bicubic + learned residual",
                f"{sr_img.width} x {sr_img.height} ({sr_img.width*sr_img.height:,} px)",
                f"{m_sr['laplacian_variance']:.2f}",
                f"{m_sr['gradient_energy']:.2f}",
            ],
            "Improvement": [
                "Not a physical GSD claim",
                "Not a ground-truth accuracy claim",
                f"16x ({in_img.width*in_img.height:,} -> {sr_img.width*sr_img.height:,})",
                f"+{lap_gain:.0f}%",
                f"+{grad_gain:.0f}%",
            ]
        })

    # Export
    st.markdown("---")
    st.markdown("### Export")
    btn1, btn2, _ = st.columns([1, 1, 2])

    buf_sr = io.BytesIO()
    sr_img.save(buf_sr, format="PNG")
    with btn1:
        st.download_button("Download Enhanced 4X (PNG)", buf_sr.getvalue(),
                           "geosr_enhanced_4x.png", "image/png", use_container_width=True)

    buf_comp = io.BytesIO()
    comp = Image.new("RGB", (sr_img.width * 2, sr_img.height))
    comp.paste(bi_img, (0, 0))
    comp.paste(sr_img, (sr_img.width, 0))
    comp.save(buf_comp, format="PNG")
    with btn2:
        st.download_button("Download Comparison (PNG)", buf_comp.getvalue(),
                           "geosr_comparison.png", "image/png", use_container_width=True)
