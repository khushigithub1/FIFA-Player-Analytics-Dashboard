from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# PATHS
# =========================================================
BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = BASE_DIR / "data"   / "players_20.csv"
MODEL_PATH = BASE_DIR / "models" / "fifa_xgboost_model.pkl"

FEATURES = ["age", "height_cm", "weight_kg", "potential", "experience"]

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="FIFA Player Analytics",
    page_icon="⚽",
    layout="wide",
)

# =========================================================
# THEME STATE
# =========================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def _toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

# ── token maps ────────────────────────────────────────────
T = {
    # backgrounds
    "bg_page":       "#080d18"      if is_dark else "#f4f6fb",
    "bg_sidebar":    "#080d18"      if is_dark else "#ffffff",
    "bg_card":       "rgba(255,255,255,.04)"   if is_dark else "rgba(255,255,255,.85)",
    "bg_card_hover": "rgba(99,102,241,.06)"    if is_dark else "rgba(99,102,241,.08)",
    "bg_insight":    "rgba(80,100,220,.08)"    if is_dark else "rgba(99,102,241,.07)",
    "bg_insight_hov":"rgba(80,100,220,.16)"    if is_dark else "rgba(99,102,241,.14)",
    "bg_metric":     "rgba(255,255,255,.03)"   if is_dark else "rgba(255,255,255,.90)",
    "bg_pred":       "linear-gradient(135deg,rgba(50,90,220,.25),rgba(100,30,180,.20),rgba(20,20,60,.30))"
                     if is_dark else
                     "linear-gradient(135deg,rgba(99,102,241,.12),rgba(168,85,247,.10),rgba(236,72,153,.08))",
    # borders
    "border_card":   "rgba(255,255,255,.09)"   if is_dark else "rgba(0,0,0,.08)",
    "border_metric": "rgba(255,255,255,.08)"   if is_dark else "rgba(0,0,0,.07)",
    "border_sidebar":"rgba(255,255,255,.07)"   if is_dark else "rgba(0,0,0,.06)",
    # text
    "text_primary":  "#e2e8f0"      if is_dark else "#0f172a",
    "text_secondary":"#9ca3af"      if is_dark else "#475569",
    "text_muted":    "#6f7890"      if is_dark else "#94a3b8",
    "text_brand":    "linear-gradient(90deg,#e0e7ff,#a5b4fc)" if is_dark else
                     "linear-gradient(90deg,#4f46e5,#7c3aed)",
    "text_h1":       "linear-gradient(100deg,#e0e7ff,#a5b4fc,#c084fc)" if is_dark else
                     "linear-gradient(100deg,#3730a3,#6366f1,#9333ea)",
    # orb colours
    "orb1":          "radial-gradient(circle,#6366f1,#4f46e5)" if is_dark else
                     "radial-gradient(circle,#c7d2fe,#a5b4fc)",
    "orb2":          "radial-gradient(circle,#a855f7,#7c3aed)" if is_dark else
                     "radial-gradient(circle,#e9d5ff,#d8b4fe)",
    "orb_opacity":   ".18"          if is_dark else ".25",
    # grid
    "grid":          "rgba(255,255,255,.06)" if is_dark else "rgba(0,0,0,.06)",
    "zeroline":      "rgba(255,255,255,.08)" if is_dark else "rgba(0,0,0,.08)",
    # chart font
    "chart_font":    "#c4c9d6"      if is_dark else "#374151",
    # toggle
    "toggle_bg":     "#1e293b"      if is_dark else "#e2e8f0",
    "toggle_knob":   "#6366f1"      if is_dark else "#6366f1",
    "toggle_icon":   "🌙"           if is_dark else "☀️",
    "toggle_label":  "Light Mode"   if is_dark else "Dark Mode",
}

# =========================================================
# MASTER CSS  (injected with live token values)
# =========================================================
st.markdown(f"""
<style>

/* ═══════════════════════════════════════════════
   GPU HINTS
═══════════════════════════════════════════════ */
*{{box-sizing:border-box}}
.kpi-card,.prediction-card,.insight,.glass-card,
[data-testid="stMetric"],[data-testid="stButton"]>button,
[data-testid="stDownloadButton"]>button,
[data-testid="stPlotlyChart"]{{
    will-change:transform,opacity,box-shadow;
    transform:translateZ(0);
    backface-visibility:hidden;
}}

/* ═══════════════════════════════════════════════
   KEYFRAMES
═══════════════════════════════════════════════ */
@keyframes fadeUp{{
    from{{opacity:0;transform:translateY(24px) scale(.97)}}
    to  {{opacity:1;transform:translateY(0)    scale(1)  }}
}}
@keyframes fadeIn{{from{{opacity:0}}to{{opacity:1}}}}
@keyframes slideInLeft{{
    from{{opacity:0;transform:translateX(-22px)}}
    to  {{opacity:1;transform:translateX(0)}}
}}
@keyframes glowPulse{{
    0%,100%{{box-shadow:0 0 0 0 rgba(99,102,241,0),0 0 0 0 rgba(168,85,247,0)}}
    50%    {{box-shadow:0 0 28px 4px rgba(99,102,241,.28),0 0 42px 8px rgba(168,85,247,.12)}}
}}
@keyframes borderSpin{{
    0%  {{background-position:0%   50%}}
    50% {{background-position:100% 50%}}
    100%{{background-position:0%   50%}}
}}
@keyframes floatOrb{{
    0%,100%{{transform:translateY(0)    translateX(0)    scale(1)   }}
    33%    {{transform:translateY(-28px) translateX(14px) scale(1.04)}}
    66%    {{transform:translateY(14px)  translateX(-18px)scale(.97) }}
}}
@keyframes popIn{{
    0%  {{opacity:0;transform:scale(.5)}}
    70% {{transform:scale(1.08)}}
    100%{{opacity:1;transform:scale(1)}}
}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}

/* ═══════════════════════════════════════════════
   PAGE SHELL — smooth theme transition
═══════════════════════════════════════════════ */
.block-container{{
    padding-top:1.5rem;padding-bottom:2.5rem;max-width:1500px;
}}
/* whole-app smooth colour transition on mode switch */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] *,
[data-testid="stSidebar"],
[data-testid="stSidebar"] *{{
    transition:background-color .45s ease,
               color .35s ease,
               border-color .35s ease,
               box-shadow .35s ease !important;
}}
[data-testid="stAppViewContainer"]{{
    background-color:{T["bg_page"]} !important;
}}

/* ── Kill the black top header bar ── */
[data-testid="stHeader"]{{
    background-color:{T["bg_page"]} !important;
    border-bottom:1px solid {T["border_card"]} !important;
    transition:background-color .45s ease !important;
}}
/* the thin rainbow decoration line under the header */
[data-testid="stDecoration"]{{
    display:none !important;
}}
/* toolbar icons area */
[data-testid="stToolbar"]{{
    background-color:{T["bg_page"]} !important;
    transition:background-color .45s ease !important;
}}
/* top-level app background (catches anything above the container) */
.stApp{{
    background-color:{T["bg_page"]} !important;
    transition:background-color .45s ease !important;
}}

/* ═══════════════════════════════════════════════
   AMBIENT ORBS
═══════════════════════════════════════════════ */
.main::before,.main::after{{
    content:"";position:fixed;border-radius:50%;
    pointer-events:none;z-index:0;
    filter:blur(90px);opacity:{T["orb_opacity"]};
    animation:floatOrb 13s ease-in-out infinite;
    transition:background .6s ease, opacity .6s ease;
}}
.main::before{{
    width:520px;height:520px;top:-110px;left:-90px;
    background:{T["orb1"]};
    animation-delay:0s;
}}
.main::after{{
    width:420px;height:420px;bottom:-80px;right:-80px;
    background:{T["orb2"]};
    animation-delay:-6.5s;
}}

/* ═══════════════════════════════════════════════
   TYPOGRAPHY
═══════════════════════════════════════════════ */
body,[data-testid="stAppViewContainer"]{{color:{T["text_primary"]}}}
h1{{
    animation:fadeUp .5s cubic-bezier(.22,1,.36,1) both;
    background:{T["text_h1"]};
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
}}
h2{{animation:fadeUp .55s cubic-bezier(.22,1,.36,1) .05s both}}
h3{{animation:fadeUp .55s cubic-bezier(.22,1,.36,1) .08s both}}
p,label,[data-testid="stMarkdownContainer"]{{color:{T["text_primary"]}}}

/* ═══════════════════════════════════════════════
   METRIC CARDS
═══════════════════════════════════════════════ */
[data-testid="stMetric"]{{
    border-radius:16px !important;
    padding:16px 18px !important;
    background:{T["bg_metric"]} !important;
    border:1px solid {T["border_metric"]} !important;
    transition:transform .22s cubic-bezier(.22,1,.36,1),
               box-shadow .22s ease, border-color .22s ease !important;
    animation:fadeUp .6s cubic-bezier(.22,1,.36,1) both;
    position:relative;overflow:hidden;
}}
[data-testid="stMetric"]::after{{
    content:"";position:absolute;inset:0;border-radius:inherit;
    background:linear-gradient(135deg,rgba(99,102,241,.10) 0%,transparent 70%);
    opacity:0;transition:opacity .25s ease;pointer-events:none;
}}
[data-testid="stMetric"]:hover{{
    transform:translateY(-4px) scale(1.015) !important;
    box-shadow:0 10px 32px rgba(0,0,0,.22),
               0 0 0 1px rgba(99,102,241,.30) !important;
    border-color:rgba(99,102,241,.35) !important;
}}
[data-testid="stMetric"]:hover::after{{opacity:1}}
[data-testid="stMetricValue"]{{animation:popIn .5s cubic-bezier(.22,1,.36,1) .1s both}}
[data-testid="stMetricLabel"]{{color:{T["text_secondary"]} !important}}
.stColumn:nth-child(1) [data-testid="stMetric"]{{animation-delay:.05s}}
.stColumn:nth-child(2) [data-testid="stMetric"]{{animation-delay:.12s}}
.stColumn:nth-child(3) [data-testid="stMetric"]{{animation-delay:.19s}}
.stColumn:nth-child(4) [data-testid="stMetric"]{{animation-delay:.26s}}

/* ═══════════════════════════════════════════════
   PREDICTION CARD
═══════════════════════════════════════════════ */
.prediction-card{{
    position:relative;text-align:center;
    padding:44px 32px;border-radius:24px;overflow:hidden;
    background:{T["bg_pred"]};
    border:1px solid rgba(100,120,255,.25);
    animation:fadeUp .55s cubic-bezier(.22,1,.36,1) both,
              glowPulse 3.5s ease-in-out 1s infinite;
    transition:transform .28s cubic-bezier(.22,1,.36,1);
}}
.prediction-card::before{{
    content:"";position:absolute;inset:-1px;border-radius:inherit;
    background:linear-gradient(270deg,#6366f1,#a855f7,#ec4899,#6366f1);
    background-size:400% 400%;
    animation:borderSpin 5s ease infinite;
    z-index:-1;
}}
.prediction-card:hover{{transform:scale(1.025) translateY(-3px)}}
.prediction-value{{
    font-size:84px;font-weight:900;line-height:1;
    background:linear-gradient(135deg,#a5b4fc,#e879f9,#fb923c);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
    animation:popIn .6s cubic-bezier(.22,1,.36,1) .2s both;
}}

/* ═══════════════════════════════════════════════
   INSIGHT BOXES
═══════════════════════════════════════════════ */
.insight{{
    padding:14px 18px;margin:10px 0;border-radius:12px;
    background:{T["bg_insight"]};
    border-left:4px solid #6272e5;
    color:{T["text_primary"]};
    transition:transform .2s cubic-bezier(.22,1,.36,1),
               background .2s ease,border-left-color .2s ease,
               box-shadow .2s ease;
    animation:slideInLeft .5s cubic-bezier(.22,1,.36,1) both;
}}
.insight:hover{{
    transform:translateX(6px);
    background:{T["bg_insight_hov"]};
    border-left-color:#a5b4fc;
    box-shadow:-2px 0 0 0 #a5b4fc;
}}

/* ═══════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════ */
[data-testid="stButton"]>button{{
    border-radius:14px !important;
    font-weight:700 !important;letter-spacing:.4px;
    position:relative;overflow:hidden;
    transition:transform .2s cubic-bezier(.22,1,.36,1),
               box-shadow .2s ease,filter .2s ease !important;
}}
[data-testid="stButton"]>button::after{{
    content:"";position:absolute;inset:0;border-radius:inherit;
    background:linear-gradient(120deg,transparent 30%,
        rgba(255,255,255,.18) 50%,transparent 70%);
    background-size:200% 100%;background-position:-200% 0;
    transition:background-position .4s ease;pointer-events:none;
}}
[data-testid="stButton"]>button:hover::after{{background-position:200% 0}}
[data-testid="stButton"]>button:hover{{
    transform:translateY(-3px) scale(1.02) !important;
    box-shadow:0 8px 28px rgba(99,102,241,.45) !important;
    filter:brightness(1.1);
}}
[data-testid="stButton"]>button:active{{
    transform:translateY(1px) scale(.97) !important;
    filter:brightness(.95);
}}
[data-testid="stDownloadButton"]>button{{
    border-radius:12px !important;
    transition:transform .2s ease,box-shadow .2s ease !important;
}}
[data-testid="stDownloadButton"]>button:hover{{
    transform:translateY(-2px) !important;
    box-shadow:0 6px 20px rgba(0,0,0,.30) !important;
}}

/* ═══════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════ */
[data-testid="stTabs"] button{{
    border-radius:10px 10px 0 0 !important;
    transition:color .2s ease,background .2s ease,transform .18s ease !important;
}}
[data-testid="stTabs"] button:hover{{
    color:#a5b4fc !important;
    background:rgba(99,102,241,.09) !important;
    transform:translateY(-1px) !important;
}}

/* ═══════════════════════════════════════════════
   CHARTS
═══════════════════════════════════════════════ */
[data-testid="stPlotlyChart"]{{
    border-radius:16px;overflow:hidden;
    animation:fadeIn .7s ease .1s both;
    transition:box-shadow .25s ease,transform .25s cubic-bezier(.22,1,.36,1);
}}
[data-testid="stPlotlyChart"]:hover{{
    box-shadow:0 6px 36px rgba(0,0,0,.28),
               0 0 0 1px rgba(99,102,241,.18);
    transform:scale(1.005);
}}

/* ═══════════════════════════════════════════════
   DATAFRAME
═══════════════════════════════════════════════ */
[data-testid="stDataFrame"]{{
    border-radius:14px;overflow:hidden;
    animation:fadeIn .55s ease both;
    transition:box-shadow .22s ease;
}}
[data-testid="stDataFrame"]:hover{{
    box-shadow:0 4px 24px rgba(0,0,0,.22);
}}

/* ═══════════════════════════════════════════════
   ALERTS
═══════════════════════════════════════════════ */
[data-testid="stAlert"]{{
    border-radius:14px !important;
    animation:fadeUp .5s ease both;
    transition:transform .2s ease,box-shadow .2s ease;
}}
[data-testid="stAlert"]:hover{{
    transform:translateY(-2px);
    box-shadow:0 4px 16px rgba(0,0,0,.18);
}}

/* ═══════════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════════ */
hr{{
    border:none !important;height:1px !important;
    background:linear-gradient(90deg,transparent,rgba(99,102,241,.45),transparent) !important;
    margin:1.2rem 0 !important;animation:fadeIn 1s ease both;
}}

/* ═══════════════════════════════════════════════
   THEME TOGGLE BUTTON  (pill in sidebar)
═══════════════════════════════════════════════ */
.theme-toggle-wrap{{
    display:flex;align-items:center;justify-content:space-between;
    padding:10px 14px;border-radius:14px;
    background:{T["toggle_bg"]};
    border:1px solid rgba(99,102,241,.20);
    cursor:pointer;margin-bottom:4px;
    transition:background .3s ease,box-shadow .3s ease;
}}
.theme-toggle-wrap:hover{{
    box-shadow:0 0 0 2px rgba(99,102,241,.35);
}}
.theme-toggle-label{{
    font-size:.82rem;font-weight:700;
    color:{T["text_secondary"]};letter-spacing:.3px;
}}
.theme-toggle-icon{{font-size:1.1rem;line-height:1}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR CSS  (mode-aware)
# =========================================================
st.markdown(f"""
<style>
[data-testid="stSidebar"]{{
    background:{T["bg_sidebar"]} !important;
    border-right:1px solid {T["border_sidebar"]};
}}
[data-testid="stSidebar"]>div:first-child{{padding-top:1.2rem}}

.sidebar-brand{{padding:8px 4px 16px 4px}}
.sidebar-brand-title{{
    font-size:1.3rem;font-weight:800;letter-spacing:-.4px;
    background:{T["text_brand"]};
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;animation:fadeIn .6s ease both;
}}
.sidebar-brand-subtitle{{
    color:{T["text_muted"]};font-size:.78rem;margin-top:4px;
    animation:fadeIn .8s ease both;
}}
.sidebar-divider{{
    height:1px;
    background:linear-gradient(90deg,transparent,{T["border_sidebar"]},transparent);
    margin:8px 0 16px 0;
}}
.sidebar-section-title{{
    color:{T["text_muted"]};font-size:.72rem;font-weight:700;
    text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;
}}
[data-testid="stSidebar"] div[role="radiogroup"]{{gap:5px}}
[data-testid="stSidebar"] div[role="radiogroup"]>label{{
    background:transparent;border:1px solid transparent;
    border-radius:12px;padding:9px 12px;
    transition:background .2s ease,border-color .2s ease,
               transform .2s cubic-bezier(.22,1,.36,1),box-shadow .2s ease;
    cursor:pointer;
}}
[data-testid="stSidebar"] div[role="radiogroup"]>label:hover{{
    background:rgba(99,102,241,.09);
    border-color:rgba(99,102,241,.28);
    transform:translateX(5px);
    box-shadow:inset 3px 0 0 rgba(99,102,241,.55);
}}
[data-testid="stSidebar"] div[role="radiogroup"]>label>div:first-child{{display:none}}
[data-testid="stSidebar"] div[role="radiogroup"] label p{{
    font-size:.92rem;font-weight:600;margin:0;color:{T["text_primary"]};
}}
[data-testid="stSidebar"] [data-testid="stMetric"]{{
    transition:transform .18s ease !important;
}}
[data-testid="stSidebar"] [data-testid="stMetric"]:hover{{
    transform:translateX(4px) !important;
    box-shadow:none !important;
}}
.sidebar-status{{
    display:inline-flex;align-items:center;gap:6px;
    margin-top:10px;padding:4px 10px;border-radius:20px;
    font-size:.68rem;font-weight:700;
    background:rgba(34,197,94,.12);color:#4ade80;
    border:1px solid rgba(34,197,94,.20);
}}
.sidebar-status-dot{{
    width:7px;height:7px;border-radius:50%;
    background:#4ade80;animation:blink 1.8s ease-in-out infinite;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA & MODEL
# =========================================================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["experience"] = df["age"] - 18
    return df

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")
    return joblib.load(MODEL_PATH)

try:
    df    = load_data()
    model = load_model()
except Exception as e:
    st.error("Application could not load the required files.")
    st.code(str(e))
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">⚽ FIFA Analytics</div>
            <div class="sidebar-brand-subtitle">Player Intelligence Platform</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Theme toggle ──────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Appearance</div>', unsafe_allow_html=True)

    st.button(
        f"{T['toggle_icon']}  Switch to {T['toggle_label']}",
        on_click=_toggle_theme,
        use_container_width=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Workspace</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard","🤖 Player Prediction","🔎 Player Explorer",
         "📊 Insights","🧠 Model Performance","ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Dataset info ──────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">Dataset</div>', unsafe_allow_html=True)
    st.metric(label="Players", value=f"{len(df):,}")
    st.caption("FIFA Players 20 Dataset")

    st.markdown("""
        <div class="sidebar-status">
            <span class="sidebar-status-dot"></span> Live
        </div>
    """, unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.title("⚽ FIFA Player Analytics")
st.caption(
    "Player performance analysis · scouting insights · "
    "clustering · XGBoost-based rating prediction."
)
st.divider()

# =========================================================
# CHART HELPER
# =========================================================
def _chart_layout(fig, height=450, **kw):
    fig.update_layout(
        height=height,
        margin=dict(l=14, r=14, t=50, b=14),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["chart_font"]),
        xaxis=dict(
            gridcolor=T["grid"],
            zerolinecolor=T["zeroline"],
        ),
        yaxis=dict(
            gridcolor=T["grid"],
            zerolinecolor=T["zeroline"],
        ),
        **kw,
    )
    return fig

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    avg_overall   = df["overall"].mean()
    avg_potential = df["potential"].mean()
    top_rating    = df["overall"].max()

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "Total Players",     f"{len(df):,}"),
        (c2, "Average Overall",   f"{avg_overall:.2f}"),
        (c3, "Average Potential", f"{avg_potential:.2f}"),
        (c4, "Top Rating",        f"{top_rating:.0f}"),
    ]:
        with col:
            with st.container(border=True):
                st.metric(label=lbl, value=val)

    st.markdown("## 📊 Performance Overview")
    col1, col2 = st.columns(2)

    with col1:
        counts = df["overall"].value_counts().sort_index().reset_index()
        counts.columns = ["Overall", "Players"]
        fig = px.bar(
            counts, x="Overall", y="Players",
            title="Overall Rating Distribution",
            color="Players",
            color_continuous_scale=["#312e81","#6366f1","#a5b4fc"],
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False)
        _chart_layout(fig)
        st.plotly_chart(fig, width="stretch")

    with col2:
        age_stats = (
            df.groupby("age")
            .agg(avg_overall=("overall","mean"), player_count=("overall","count"))
            .reset_index()
        )
        age_stats = age_stats[age_stats["player_count"] >= 50]
        fig = px.line(
            age_stats, x="age", y="avg_overall",
            markers=True, title="Average Overall by Age",
            color_discrete_sequence=["#a5b4fc"],
        )
        fig.update_traces(
            line=dict(width=2.5),
            marker=dict(size=7, color="#818cf8", line=dict(color="#c4b5fd", width=1.5)),
        )
        fig.update_layout(xaxis_title="Age", yaxis_title="Average Overall")
        _chart_layout(fig)
        st.plotly_chart(fig, width="stretch")

# =========================================================
# PREDICTION
# =========================================================
elif page == "🤖 Player Prediction":

    st.markdown("## 🤖 AI Player Rating Prediction")
    st.write("Enter the same five features used during model training.")

    col1, col2 = st.columns(2)
    with col1:
        age    = st.slider("Age",         16, 45,  25)
        height = st.slider("Height (cm)", 150, 210, 181)
        weight = st.slider("Weight (kg)",  45, 120,  75)
    with col2:
        potential  = st.slider("Potential", 40, 99, 70)
        experience = age - 18
        st.number_input("Experience", value=experience, disabled=True)

    st.info(
        "Experience is automatically calculated as Age − 18, "
        "matching the feature-engineering logic used during training."
    )

    input_data = pd.DataFrame(
        [[age, height, weight, potential, experience]], columns=FEATURES
    )

    if st.button("⚡ Predict Overall Rating", type="primary", use_container_width=True):

        prediction = float(model.predict(input_data)[0])
        prediction = np.clip(prediction, 1, 99)

        if   prediction >= 85: category = "Elite Player ⭐⭐⭐"
        elif prediction >= 75: category = "High Rated Player ⭐⭐"
        elif prediction >= 65: category = "Good Player ⭐"
        else:                  category = "Developing Player"

        st.markdown("## 🎯 Prediction Result")

        p1, p2 = st.columns(2)
        with p1:
            with st.container(border=True):
                st.subheader("Predicted Overall")
                st.metric("Rating", f"{prediction:.1f}")
        with p2:
            with st.container(border=True):
                st.subheader("Performance Category")
                st.metric("Category", category)

        st.markdown(f"""
            <div class="prediction-card" style="margin-top:20px;">
                <div style="color:{T['text_secondary']};font-size:13px;
                            text-transform:uppercase;letter-spacing:1.2px;
                            margin-bottom:8px;">Overall Rating</div>
                <div class="prediction-value">{prediction:.1f}</div>
                <div style="margin-top:12px;font-size:20px;font-weight:700;
                            color:#c4b5fd;letter-spacing:.3px;">{category}</div>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("Prediction Inputs")
        st.dataframe(input_data, hide_index=True, width="stretch")

        pred_df = input_data.copy()
        pred_df["predicted_overall"] = round(prediction, 2)
        pred_df["category"] = category

        st.download_button(
            "⬇ Download Prediction",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="player_prediction.csv",
            mime="text/csv",
        )

# =========================================================
# PLAYER EXPLORER
# =========================================================
elif page == "🔎 Player Explorer":

    st.markdown("## 🔎 Player Explorer")

    search     = st.text_input("Search Player")
    min_rating = st.slider("Minimum Overall", 40, 95, 70)

    filtered = df[df["overall"] >= min_rating].copy()
    if search:
        filtered = filtered[
            filtered["short_name"].astype(str)
            .str.contains(search, case=False, na=False)
        ]

    columns = [
        c for c in ["short_name","overall","potential","age",
                    "height_cm","weight_kg","nationality","club"]
        if c in filtered.columns
    ]

    st.write(f"Players found: **{len(filtered):,}**")
    st.dataframe(filtered[columns].head(250), hide_index=True, width="stretch")

    st.download_button(
        "⬇ Download Filtered Players",
        data=filtered[columns].to_csv(index=False).encode("utf-8"),
        file_name="filtered_players.csv",
        mime="text/csv",
    )

# =========================================================
# INSIGHTS
# =========================================================
elif page == "📊 Insights":

    st.markdown("## 📊 FIFA Insights")
    tab1, tab2, tab3 = st.tabs(["Age & Performance", "Potential", "Market Analysis"])

    with tab1:
        sample = df[["age","overall"]].dropna().sample(min(5000, len(df)), random_state=42)
        fig = px.scatter(
            sample, x="age", y="overall", opacity=.45,
            title="Age vs Overall Rating",
            color_discrete_sequence=["#818cf8"],
        )
        _chart_layout(fig)
        st.plotly_chart(fig, width="stretch")

        st.markdown("## 🔍 Key Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("⭐ The dataset contains 18,278 FIFA players with an average overall rating of 66.24.")
        with col2:
            st.info("📈 Player ratings generally increase through the prime-age range before gradually declining.")
        with col3:
            st.info("🤖 The XGBoost deployment model achieved an R² score of 0.94968 on the clean test split.")

    with tab2:
        sample = df[["potential","overall"]].dropna().sample(min(5000, len(df)), random_state=42)
        fig = px.scatter(
            sample, x="potential", y="overall", opacity=.45,
            title="Potential vs Overall Rating",
            color_discrete_sequence=["#a78bfa"],
        )
        _chart_layout(fig)
        st.plotly_chart(fig, width="stretch")

        st.markdown("""
            <div class="insight">
                <b>Insight:</b> Potential is strongly related to current overall rating
                and is particularly useful for identifying future talent.
            </div>
        """, unsafe_allow_html=True)

    with tab3:
        if "value_eur" in df.columns:
            sample = df[["value_eur","overall"]].dropna()
            sample = sample[sample["value_eur"] > 0].sample(min(5000, len(sample)), random_state=42)
            fig = px.scatter(
                sample, x="value_eur", y="overall", log_x=True, opacity=.45,
                title="Player Value vs Overall",
                color_discrete_sequence=["#34d399"],
            )
            _chart_layout(fig)
            st.plotly_chart(fig, width="stretch")

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "🧠 Model Performance":

    st.markdown("## 🧠 Model Performance")

    model_scores = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "XGBoost - Original Notebook",
            "XGBoost - Deployment Model",
            "Neural Network",
        ],
        "R2 Score": [0.791393, 0.923355, 0.949680, 0.921643],
    })

    fig = px.bar(
        model_scores, x="Model", y="R2 Score",
        text="R2 Score", title="Model Comparison",
        color="R2 Score",
        color_continuous_scale=["#312e81","#6366f1","#a5b4fc"],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
    fig.update_yaxes(range=[0, 1.08])
    fig.update_layout(coloraxis_showscale=False)
    _chart_layout(fig)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(model_scores, hide_index=True, width="stretch")
    st.metric("Deployment Model R²", "0.94968")

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature":    FEATURES,
            "Importance": model.feature_importances_,
        }).sort_values("Importance")

        fig = px.bar(
            importance, x="Importance", y="Feature",
            orientation="h", title="XGBoost Feature Importance",
            color="Importance",
            color_continuous_scale=["#4c1d95","#8b5cf6","#ddd6fe"],
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False)
        _chart_layout(fig, height=320)
        st.plotly_chart(fig, width="stretch")

    st.download_button(
        "⬇ Download Model Comparison",
        data=model_scores.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

# =========================================================
# ABOUT
# =========================================================
else:

    st.markdown("## ℹ️ About the Project")

    st.write(
        """
        This project performs FIFA player data analysis,
        preprocessing, machine-learning model comparison,
        clustering and player-performance prediction.

        The deployed application uses an XGBoost regression model
        to predict FIFA overall rating from player characteristics.
        """
    )

    st.subheader("Technology Stack")
    st.write("Python · Pandas · NumPy · Scikit-learn · XGBoost · Plotly · Streamlit")

    st.subheader("Use Cases")
    for use in [
        "Player scouting",
        "Talent identification",
        "Player benchmarking",
        "Performance analysis",
        "Football analytics",
    ]:
        st.markdown(
            f'<div class="insight">• {use}</div>',
            unsafe_allow_html=True,
        )

    st.success("FIFA Player Analytics Dashboard is ready for deployment.")
