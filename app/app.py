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
                          label_left="GeoSR reconstruction", label_right="Bicubic baseline"):
    """Renders an interactive curtain wipe slider with hover-to-zoom."""
    b64_left = pil_to_base64(img_left)
    b64_right = pil_to_base64(img_right)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: transparent; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif; }}
        .slider-wrap {{
            position: relative; width: 100%; height: 580px; overflow: hidden;
            border-radius: 18px; user-select: none; cursor: ew-resize; background: #000;
        }}
        .img-layer {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            object-fit: contain; pointer-events: none;
        }}
        .left-layer {{ clip-path: polygon(0 0, 50% 0, 50% 100%, 0 100%); }}
        .curtain {{
            position: absolute; top: 0; bottom: 0; left: 50%; width: 1.5px;
            background: rgba(255,255,255,0.85); z-index: 10;
        }}
        .handle {{
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            width: 40px; height: 40px; border-radius: 50%; background: rgba(255,255,255,0.9);
            backdrop-filter: blur(8px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.35); display: flex;
            align-items: center; justify-content: center;
            font-weight: 500; font-size: 15px; color: #1c1c1e;
        }}
        .badge {{
            position: absolute; top: 14px; padding: 6px 13px; border-radius: 980px;
            font-size: 12.5px; font-weight: 500; letter-spacing: 0.01em;
            z-index: 8; background: rgba(28,28,30,0.55); backdrop-filter: blur(12px);
            color: #f5f5f7;
        }}
        .badge-l {{ left: 14px; }}
        .badge-r {{ right: 14px; color: #98989d; }}
        .zoom-lens {{
            display: none; position: absolute; width: 190px; height: 190px;
            border: 1px solid rgba(255,255,255,0.35); border-radius: 50%; pointer-events: none;
            z-index: 20; overflow: hidden; box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        }}
        .zoom-lens img {{ position: absolute; pointer-events: none; }}
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
        <p style="text-align:center;color:#6e6e73;font-size:13px;margin-top:14px;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',sans-serif;">
            Drag to compare &nbsp;&middot;&nbsp; hold shift and hover to magnify
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
                    lens.style.left=(mx-95)+'px'; lens.style.top=(my-95)+'px';
                    const zoom=3;
                    const natW=imgL.naturalWidth, natH=imgL.naturalHeight;
                    lensImg.style.width=(r.width*zoom)+'px'; lensImg.style.height=(r.height*zoom)+'px';
                    lensImg.style.left=(-mx*zoom+95)+'px'; lensImg.style.top=(-my*zoom+95)+'px';
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
                           label_left="Sentinel-2 baseline", label_right="GeoSR enhanced"):
    """Interactive side-by-side with synchronized hover-to-zoom magnifier."""
    b64_l = pil_to_base64(img_left)
    b64_r = pil_to_base64(img_right)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ background:transparent; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif; }}
        .grid {{ display:flex; gap:12px; width:100%; }}
        .panel {{
            flex:1; position:relative; overflow:hidden; border-radius:18px;
            background:#000; height:560px;
        }}
        .panel img {{ width:100%; height:100%; object-fit:contain; display:block; }}
        .panel-label {{
            position:absolute; top:14px; left:14px; padding:6px 13px; border-radius:980px;
            font-size:12.5px; font-weight:500;
            background:rgba(28,28,30,0.55); backdrop-filter: blur(12px); color:#f5f5f7;
            z-index:5;
        }}
        .magnifier {{
            display:none; position:absolute; width:210px; height:210px;
            border:1px solid rgba(255,255,255,0.35); border-radius:14px; pointer-events:none;
            z-index:10; overflow:hidden; box-shadow:0 16px 48px rgba(0,0,0,0.5);
        }}
        .magnifier img {{ position:absolute; pointer-events:none; }}
        .zoom-label {{
            position:absolute; bottom:6px; right:6px; font-size:10.5px; color:#f5f5f7;
            background:rgba(28,28,30,0.7); padding:2px 7px; border-radius:980px; z-index:11;
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
        <p style="text-align:center;color:#6e6e73;font-size:13px;margin-top:14px;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',sans-serif;">
            Hover either panel to magnify both views together
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

                    mag.style.left=(mx-105)+'px'; mag.style.top=(my-105)+'px';
                    magImg.style.width=(r.width*zoom)+'px'; magImg.style.height=(r.height*zoom)+'px';
                    magImg.style.left=(-mx*zoom+105)+'px'; magImg.style.top=(-my*zoom+105)+'px';

                    const omx=px*or2.width, omy=py*or2.height;
                    oMag.style.left=(omx-105)+'px'; oMag.style.top=(omy-105)+'px';
                    oMagImg.style.width=(or2.width*zoom)+'px'; oMagImg.style.height=(or2.height*zoom)+'px';
                    oMagImg.style.left=(-omx*zoom+105)+'px'; oMagImg.style.top=(-omy*zoom+105)+'px';
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
    page_title="GeoSR",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --bg: #000000;
        --surface-1: #1c1c1e;
        --surface-2: #2c2c2e;
        --hairline: rgba(255,255,255,0.08);
        --hairline-strong: rgba(255,255,255,0.14);
        --text-primary: #f5f5f7;
        --text-secondary: #98989d;
        --text-tertiary: #6e6e73;
        --accent: #2f9bff;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--bg) !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", sans-serif;
        color: var(--text-primary);
    }
    [data-testid="stHeader"] { background-color: transparent !important; }

    [data-testid="stSidebar"] {
        background-color: var(--surface-1) !important;
        border-right: 1px solid var(--hairline);
    }
    [data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important; font-weight: 590 !important;
        font-size: 15px !important; letter-spacing: -0.01em;
    }
    [data-testid="stMarkdownContainer"] p, label, .stCaption { color: var(--text-secondary) !important; }

    .reportview-container .main .block-container { max-width: 1180px; padding-top: 2.2rem; }

    .hero {
        padding: 0 0 1.8rem 0;
        margin-bottom: 1.6rem;
        border-bottom: 1px solid var(--hairline);
    }
    .hero-title {
        font-size: 2.1rem; font-weight: 590; letter-spacing: -0.025em;
        color: var(--text-primary); margin-bottom: 0.4rem; line-height: 1.15;
    }
    .hero-sub {
        font-size: 1rem; font-weight: 400; color: var(--text-secondary);
        letter-spacing: -0.005em; max-width: 620px; line-height: 1.5;
    }

    .stat-row {
        display: flex; padding: 1.4rem 0; margin-bottom: 1.6rem;
        border-bottom: 1px solid var(--hairline);
    }
    .stat-item { flex: 1; padding: 0 1.6rem; border-left: 1px solid var(--hairline); }
    .stat-item:first-child { border-left: none; padding-left: 0; }
    .stat-val {
        font-size: 2rem; font-weight: 590; letter-spacing: -0.02em; color: var(--text-primary);
        line-height: 1.1;
    }
    .stat-lbl {
        font-size: 0.83rem; font-weight: 400; color: var(--text-tertiary);
        margin-top: 0.3rem; letter-spacing: -0.005em;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid var(--hairline); }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.92rem; font-weight: 450; color: var(--text-tertiary);
        background: transparent; padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important; border-bottom-color: var(--text-primary) !important;
    }

    .stButton button, .stDownloadButton button {
        background: var(--surface-2) !important; color: var(--text-primary) !important;
        border: none !important; border-radius: 980px !important;
        font-weight: 500 !important; font-size: 0.9rem !important;
        padding: 0.55rem 1.3rem !important;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background: var(--accent) !important; color: #ffffff !important;
    }

    [data-testid="stTable"] table { background: transparent !important; color: var(--text-secondary) !important; }
    [data-testid="stTable"] th {
        background: transparent !important; color: var(--text-tertiary) !important;
        font-weight: 500 !important; font-size: 0.82rem !important;
        border-bottom: 1px solid var(--hairline-strong) !important;
    }
    [data-testid="stTable"] td {
        border-color: var(--hairline) !important; font-size: 0.88rem !important;
        color: var(--text-secondary) !important;
    }

    [data-testid="stAlert"] {
        background: var(--surface-1); border: 1px solid var(--hairline); border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">GeoSR</div>
    <div class="hero-sub">4x AI-assisted visual enhancement for Sentinel-2 satellite imagery, streamed live from the Copernicus catalog.</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Data source")
    data_source = st.radio(
        "Select pipeline mode",
        [
            "Live Copernicus Satellite Stream",
            "Regional Benchmark Presets",
            "Local File Upload"
        ],
        index=1,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Display")
    enh_strength = st.select_slider(
        "Cosmetic adjustment",
        options=["none", "gentle"],
        value="none",
        help="Off preserves the neural model output. Gentle only adjusts display contrast; it does not recover geographic detail."
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Constellation — Sentinel-2A / 2B / 2C
    Input — Sentinel-2 RGB or local raster imagery
    Output — 4x enlarged pixel grid
    Method — Bicubic base plus learned EDSR residual
    """)

# ============================================================
# DATA INGESTION
# ============================================================

sr_img = None
bi_img = None
in_img = None
source_label = ""

if data_source == "Live Copernicus Satellite Stream":
    st.markdown("#### Live Copernicus Sentinel-2 stream")
    st.caption("Query the open Sentinel-2 L2A catalog for any location on Earth. Enter coordinates or select a preset city.")

    input_mode = st.radio("Location input", ["Preset City", "Custom Coordinates"], horizontal=True)

    col_q1, col_q2 = st.columns([2, 1])
    if input_mode == "Preset City":
        with col_q1:
            city_sel = st.selectbox("Target location", list(CITY_COORDINATES.keys()))
            coords = CITY_COORDINATES[city_sel]
            lat_val, lon_val = coords["lat"], coords["lon"]
        with col_q2:
            max_cloud = st.slider("Max cloud cover (%)", 1, 30, 10)
    else:
        with col_q1:
            c1, c2 = st.columns(2)
            with c1:
                lat_val = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.585, step=0.01, format="%.4f")
            with c2:
                lon_val = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.205, step=0.01, format="%.4f")
        with col_q2:
            max_cloud = st.slider("Max cloud cover (%)", 1, 30, 10)

    if st.button("Query Sentinel-2 and super-resolve"):
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
        st.info("Enter any latitude/longitude on Earth or pick a city, then click 'Query Sentinel-2 and super-resolve'.")
        st.stop()

elif data_source == "Regional Benchmark Presets":
    preset_id = st.selectbox(
        "Select regional landmark (Delhi NCR Sentinel-2 granule)",
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
        with st.spinner("Processing 4x super-resolution..."):
            proc = get_processor()
            res = proc.enhance_file(p["input"], strength=enh_strength)
            sr_img = Image.fromarray((res["sr"] * 255.0).astype(np.uint8))
            bi_img = Image.fromarray((res["bicubic"] * 255.0).astype(np.uint8))
            in_img = Image.open(p["input"]).convert("RGB")

else:
    st.markdown("#### Local file upload")
    uploaded = st.file_uploader(
        "Upload satellite imagery (GeoTIFF, JP2, PNG, JPEG)",
        type=["tif", "tiff", "jp2", "png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        temp_dir = ROOT / "outputs" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded.name
        with open(temp_path, "wb") as f:
            f.write(uploaded.getvalue())
        with st.spinner("Processing through 4x super-resolution engine..."):
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

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-item">
            <div class="stat-val">4x</div>
            <div class="stat-lbl">Pixel-grid scale</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">16x</div>
            <div class="stat-lbl">Pixel density &middot; {in_img.width}&sup2; &rarr; {sr_img.width}&sup2;</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">{m_sr["laplacian_variance"]:.1f}</div>
            <div class="stat-lbl">Sharpness statistic</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">{m_sr["gradient_energy"]:.1f}</div>
            <div class="stat-lbl">Edge statistic</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_curtain, tab_zoom, tab_metrics = st.tabs([
        "Curtain comparison",
        "Synchronized zoom",
        "Quantitative analytics"
    ])

    with tab_curtain:
        st.caption("Drag the vertical divider to compare. Hold Shift and hover to magnify.")
        render_curtain_slider(sr_img, bi_img)

    with tab_zoom:
        st.caption("Hover over either panel to magnify both views simultaneously. Move your cursor to inspect roads, rooftops, and field boundaries.")
        render_zoom_comparison(bi_img, sr_img)

    with tab_metrics:
        st.markdown("#### Image statistics")
        st.caption("Sharpness and edge statistics describe pixel contrast, not verified spatial accuracy.")
        st.table({
            "Metric": [
                "Pixel-grid scale",
                "Method",
                "Total pixels",
                "Laplacian variance (sharpness)",
                "Gradient energy (edge density)",
            ],
            "Sentinel-2 baseline (10m)": [
                "4x bicubic",
                "Interpolation baseline",
                f"{in_img.width} x {in_img.height} ({in_img.width*in_img.height:,} px)",
                f"{m_bi['laplacian_variance']:.2f}",
                f"{m_bi['gradient_energy']:.2f}",
            ],
            "GeoSR model output": [
                "4x model output",
                "Bicubic + learned residual",
                f"{sr_img.width} x {sr_img.height} ({sr_img.width*sr_img.height:,} px)",
                f"{m_sr['laplacian_variance']:.2f}",
                f"{m_sr['gradient_energy']:.2f}",
            ],
            "Change": [
                "Not a physical GSD claim",
                "Not a ground-truth accuracy claim",
                f"16x ({in_img.width*in_img.height:,} to {sr_img.width*sr_img.height:,})",
                f"+{lap_gain:.0f}%",
                f"+{grad_gain:.0f}%",
            ]
        })

    st.markdown("---")
    st.markdown("### Export")
    btn1, btn2, _ = st.columns([1, 1, 2])

    buf_sr = io.BytesIO()
    sr_img.save(buf_sr, format="PNG")
    with btn1:
        st.download_button("Download enhanced (PNG)", buf_sr.getvalue(),
                           "geosr_enhanced_4x.png", "image/png", use_container_width=True)

    buf_comp = io.BytesIO()
    comp = Image.new("RGB", (sr_img.width * 2, sr_img.height))
    comp.paste(bi_img, (0, 0))
    comp.paste(sr_img, (sr_img.width, 0))
    comp.save(buf_comp, format="PNG")
    with btn2:
        st.download_button("Download comparison (PNG)", buf_comp.getvalue(),
                           "geosr_comparison.png", "image/png", use_container_width=True)