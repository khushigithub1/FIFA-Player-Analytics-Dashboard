from pathlib import Path
import textwrap

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "players_20.csv"
MODEL_PATH = BASE_DIR / "models" / "fifa_xgboost_model.pkl"

FEATURES = [
    "age",
    "height_cm",
    "weight_kg",
    "potential",
    "experience",
]

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="FIFA Player Analytics",
    page_icon="⚽",
    layout="wide",
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero {
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 20px;
        background:
            linear-gradient(
                135deg,
                rgba(50,70,180,0.25),
                rgba(20,20,40,0.35)
            );
        border: 1px solid rgba(255,255,255,0.10);
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #9ca3af;
        font-size: 17px;
    }

    .kpi-card {
        padding: 18px;
        border-radius: 15px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .kpi-title {
        color: #9ca3af;
        font-size: 14px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
    }

    .prediction-card {
        text-align: center;
        padding: 35px;
        border-radius: 20px;
        background:
            linear-gradient(
                135deg,
                rgba(50,90,220,0.25),
                rgba(100,30,180,0.18)
            );
        border: 1px solid rgba(100,120,255,0.25);
    }

    .prediction-value {
        font-size: 72px;
        font-weight: 900;
    }

    .insight {
        padding: 14px 18px;
        margin: 10px 0;
        border-radius: 10px;
        background: rgba(80,100,220,0.08);
        border-left: 4px solid #6272e5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Same feature engineering used during model training
    df["experience"] = df["age"] - 18

    return df


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# =========================================================
# INITIALIZE
# =========================================================
try:

    df = load_data()
    model = load_model()

except Exception as e:

    st.error("Application could not load the required files.")
    st.code(str(e))
    st.stop()



# =========================================================
# PREMIUM SIDEBAR
# =========================================================

st.markdown(
    """
    <style>

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0b0f19 0%,
            #111827 55%,
            #0b0f19 100%
        );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Sidebar main content */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    /* Brand */
    .sidebar-brand {
        padding: 8px 4px 18px 4px;
    }

    .sidebar-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.4px;
    }

    .sidebar-brand-subtitle {
        color: #8b93a7;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* Divider */
    .sidebar-divider {
        height: 1px;
        background: rgba(255,255,255,0.08);
        margin: 8px 0 18px 0;
    }

    /* Navigation heading */
    .sidebar-section-title {
        color: #6f7890;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    /* Radio buttons */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 6px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 12px;
        padding: 9px 12px;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(255,255,255,0.055);
        border-color: rgba(255,255,255,0.07);
    }

    /* Hide default radio circle */
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    /* Navigation text */
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 0.92rem;
        font-weight: 600;
        margin: 0;
    }

    /* Footer card */
    .sidebar-footer {
        margin-top: 28px;
        padding: 14px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.035);
    }

    .sidebar-footer-title {
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .sidebar-footer-text {
        color: #8b93a7;
        font-size: 0.72rem;
        line-height: 1.5;
    }

    .sidebar-status {
        display: inline-block;
        margin-top: 9px;
        padding: 4px 8px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 700;
        background: rgba(34,197,94,0.12);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.18);
    }

    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:

    # -------------------------
    # Brand
    # -------------------------
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                ⚽ FIFA Analytics
            </div>
            <div class="sidebar-brand-subtitle">
                Player Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    # -------------------------
    # Navigation
    # -------------------------
    st.markdown(
        '<div class="sidebar-section-title">Workspace</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🤖 Player Prediction",
            "🔎 Player Explorer",
            "📊 Insights",
            "🧠 Model Performance",
            "ℹ️ About",
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    # -------------------------
    # Dataset information
    # -------------------------
    st.markdown(
        '<div class="sidebar-section-title">Dataset</div>',
        unsafe_allow_html=True
    )

    st.metric(
        label="Players",
        value=f"{len(df):,}"
    )

    st.caption("FIFA Players 20 Dataset")

   


# =========================================================
# HEADER
# =========================================================

st.title("⚽ FIFA Player Analytics")

st.caption(
    "Player performance analysis, scouting insights, "
    "clustering and XGBoost-based rating prediction."
)

st.divider()

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    avg_overall = df["overall"].mean()
    avg_potential = df["potential"].mean()
    top_rating = df["overall"].max()

    # ==============================
    # KPI CARDS
    # ==============================
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True):
            st.metric(
                label="Total Players",
                value=f"{len(df):,}"
            )

    with c2:
         with st.container(border=True):
            st.metric(
               label="Average Overall",
               value=f"{avg_overall:.2f}"
            )

    with c3:
        with st.container(border=True):    
            st.metric(
               label="Average Potential",
               value=f"{avg_potential:.2f}"
            )

    with c4:
        with st.container(border=True):
            st.metric(
               label="Top Rating",
               value=f"{top_rating:.0f}"
            )

    # ==============================
    # PERFORMANCE OVERVIEW
    # ==============================
    st.markdown("## 📊 Performance Overview")

    col1, col2 = st.columns(2)

    # ------------------------------
    # Overall Rating Distribution
    # ------------------------------
    with col1:

        counts = (
            df["overall"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        counts.columns = [
            "Overall",
            "Players"
        ]

        fig = px.bar(
            counts,
            x="Overall",
            y="Players",
            title="Overall Rating Distribution"
        )

        fig.update_layout(
            height=450,
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # ------------------------------
    # Average Overall by Age
    # ------------------------------
    with col2:

        age_stats = (
            df.groupby("age")
            .agg(
                avg_overall=("overall", "mean"),
                player_count=("overall", "count")
            )
            .reset_index()
        )

        #  Keep only ages with enough players
        age_stats = age_stats[
            age_stats["player_count"] >= 50
        ]

        fig = px.line(
            age_stats,
            x="age",
            y="avg_overall",
            markers=True,
            title="Average Overall by Age"
        )

        fig.update_layout(
            height=450,
            xaxis_title="Age",
            yaxis_title="Average Overall"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

# =========================================================
# PREDICTION
# =========================================================
elif page == "🤖 Player Prediction":

    st.markdown("## 🤖 AI Player Rating Prediction")

    st.write(
        "Enter the same five features used during model training."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            16,
            45,
            25,
        )

        height = st.slider(
            "Height (cm)",
            150,
            210,
            181,
        )

        weight = st.slider(
            "Weight (kg)",
            45,
            120,
            75,
        )

    with col2:

        potential = st.slider(
            "Potential",
            40,
            99,
            70,
        )

        experience = age - 18

        st.number_input(
            "Experience",
            value=experience,
            disabled=True,
        )

    st.info(
        "Experience is automatically calculated as Age - 18, "
        "matching the feature-engineering logic used during training."
    )

    input_data = pd.DataFrame(
        [
            [
                age,
                height,
                weight,
                potential,
                experience,
            ]
        ],
        columns=FEATURES,
    )

    if st.button(
        "⚡ Predict Overall Rating",
        type="primary",
        width="stretch",
    ):

        prediction = float(
            model.predict(input_data)[0]
        )

        prediction = np.clip(
            prediction,
            1,
            99
        )

        if prediction >= 85:

            category = "Elite Player ⭐⭐⭐"

        elif prediction >= 75:

            category = "High Rated Player ⭐⭐"

        elif prediction >= 65:

            category = "Good Player ⭐"

        else:

            category = "Developing Player"

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

        st.subheader("Prediction Inputs")

        st.dataframe(
            input_data,
            hide_index=True,
            width="stretch",
        )

        prediction_df = input_data.copy()

        prediction_df["predicted_overall"] = round(
            prediction,
            2,
        )

        prediction_df["category"] = category

        st.download_button(
            "⬇ Download Prediction",
            data=prediction_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="player_prediction.csv",
            mime="text/csv",
        )


# =========================================================
# PLAYER EXPLORER
# =========================================================
elif page == "🔎 Player Explorer":

    st.markdown("## 🔎 Player Explorer")

    search = st.text_input(
        "Search Player"
    )

    min_rating = st.slider(
        "Minimum Overall",
        40,
        95,
        70,
    )

    filtered = df[
        df["overall"] >= min_rating
    ].copy()

    if search:

        filtered = filtered[
            filtered["short_name"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False,
            )
        ]

    columns = [
        "short_name",
        "overall",
        "potential",
        "age",
        "height_cm",
        "weight_kg",
        "nationality",
        "club",
    ]

    columns = [
        c for c in columns
        if c in filtered.columns
    ]

    st.write(
        f"Players found: {len(filtered):,}"
    )

    st.dataframe(
        filtered[columns]
        .head(250),
        hide_index=True,
        width="stretch"
    )

    st.download_button(
        "⬇ Download Filtered Players",
        data=filtered[columns]
        .to_csv(index=False)
        .encode("utf-8"),
        file_name="filtered_players.csv",
        mime="text/csv",
    )


# =========================================================
# INSIGHTS
# =========================================================
elif page == "📊 Insights":

    st.markdown("## 📊 FIFA Insights")

    tab1, tab2, tab3 = st.tabs(
        [
            "Age & Performance",
            "Potential",
            "Market Analysis",
        ]
    )

    # -----------------------------------------------------
    # Age
    # -----------------------------------------------------
    with tab1:

        sample = (
            df[
                [
                    "age",
                    "overall",
                ]
            ]
            .dropna()
            .sample(
                min(5000, len(df)),
                random_state=42,
            )
        )

        fig = px.scatter(
            sample,
            x="age",
            y="overall",
            opacity=0.45,
            title="Age vs Overall Rating",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.markdown("## 🔍 Key Insights")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(
               "⭐ The dataset contains 18,278 FIFA players "
               "with an average overall rating of 66.24."
            )

        with col2:
            st.info(
              "📈 Player ratings generally increase through "
              " the prime-age range before gradually declining."
            )

        with col3:
            st.info(
               "🤖 The deployment XGBoost model achieved "
               "an R² score of 0.94968 on the clean test split."
            )

    # -----------------------------------------------------
    # Potential
    # -----------------------------------------------------
    with tab2:

        sample = (
            df[
                [
                    "potential",
                    "overall",
                ]
            ]
            .dropna()
            .sample(
                min(5000, len(df)),
                random_state=42,
            )
        )

        fig = px.scatter(
            sample,
            x="potential",
            y="overall",
            opacity=0.45,
            title="Potential vs Overall Rating",
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.markdown(
            """
            <div class="insight">

            <b>Insight:</b>
            Potential is strongly related to current overall
            rating and is particularly useful for identifying
            future talent.

            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # Market
    # -----------------------------------------------------
    with tab3:

        if "value_eur" in df.columns:

            sample = (
                df[
                    [
                        "value_eur",
                        "overall",
                    ]
                ]
                .dropna()
            )

            sample = sample[
                sample["value_eur"] > 0
            ].sample(
                min(5000, len(sample)),
                random_state=42,
            )

            fig = px.scatter(
                sample,
                x="value_eur",
                y="overall",
                log_x=True,
                opacity=0.45,
                title="Player Value vs Overall",
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "🧠 Model Performance":

    st.markdown("## 🧠 Model Performance")

    model_scores = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "XGBoost - Original Notebook",
                "XGBoost - Deployment Model",
                "Neural Network",
            ],
            "R2 Score": [
                0.791393,
                0.923355,
                0.949680,
                0.921643,
            ],
        }
    )

    fig = px.bar(
        model_scores,
        x="Model",
        y="R2 Score",
        text="R2 Score",
        title="Model Comparison",
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig.update_yaxes(
        range=[0, 1]
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.dataframe(
        model_scores,
        hide_index=True,
        width="stretch"
    )

    st.metric(
        "Deployment Model R²",
        "0.94968",
    )

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance = pd.DataFrame(
            {
                "Feature": FEATURES,
                "Importance": model.feature_importances_,
            }
        ).sort_values(
            "Importance"
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="XGBoost Feature Importance",
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.download_button(
        "⬇ Download Model Comparison",
        data=model_scores
        .to_csv(index=False)
        .encode("utf-8"),
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

        The deployed application uses an XGBoost regression
        model to predict FIFA overall rating from player
        characteristics.
        """
    )

    st.subheader("Technology Stack")

    st.write(
        """
        Python • Pandas • NumPy • Scikit-learn • XGBoost •
        Plotly • Streamlit
        """
    )

    st.subheader("Use Cases")

    st.write(
        """
        • Player scouting

        • Talent identification

        • Player benchmarking

        • Performance analysis

        • Football analytics
        """
    )

    st.success(
        "FIFA Player Analytics Dashboard is ready for deployment."
    )