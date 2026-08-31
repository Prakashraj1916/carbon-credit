import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Carbon Emission Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FORECAST_MODEL_PATH = BASE_DIR / "carbon_emission_forecasting_xgboost.pkl"

FORECAST_RESULTS_PATH = BASE_DIR / "carbon_forecast_results.csv"
PREDICTIONS_PATH = BASE_DIR / "xgboost_predictions.csv"
SHAP_PATH = BASE_DIR / "shap_feature_importance.csv"
FORECAST_IMPORTANCE_PATH = BASE_DIR / "forecast_feature_importance.csv"
NEXT_YEAR_PATH = BASE_DIR / "next_year_co2_prediction.csv"


# ============================================================
# CUSTOM CSS — ENVIRONMENTAL GREEN THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background-color: #F4F8F5;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ======================================================
       MAIN HEADER
       ====================================================== */

    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: #14532D;
        margin-bottom: 4px;
        letter-spacing: -1px;
    }

    .subtitle {
        font-size: 18px;
        color: #527064;
        margin-bottom: 30px;
    }


    /* ======================================================
       SECTION HEADINGS
       ====================================================== */

    .section-title {
        font-size: 28px;
        font-weight: 750;
        color: #166534;
        margin-top: 28px;
        margin-bottom: 18px;
    }


    /* ======================================================
       DASHBOARD CARDS
       ====================================================== */

    .metric-card {
        background: linear-gradient(
            135deg,
            #FFFFFF,
            #F0FDF4
        );

        padding: 24px;

        border-radius: 18px;

        border: 1px solid #BBE7C7;

        box-shadow:
            0 5px 18px rgba(20, 83, 45, 0.08);

        min-height: 125px;
    }

    .metric-title {
        color: #527064;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.7px;
    }

    .metric-value {
        color: #15803D;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }


    /* ======================================================
       INFORMATION BOX
       ====================================================== */

    .info-box {
        background: linear-gradient(
            135deg,
            #ECFDF5,
            #F0FDF4
        );

        padding: 20px;

        border-radius: 14px;

        border-left: 5px solid #16A34A;

        color: #28543D;

        margin-bottom: 22px;

        box-shadow:
            0 3px 12px rgba(22, 101, 52, 0.05);
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {

        background: linear-gradient(
            180deg,
            #064E3B 0%,
            #065F46 50%,
            #047857 100%
        );
    }


    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }


    section[data-testid="stSidebar"] .stRadio label {
        padding: 8px;
        border-radius: 8px;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {

        background: linear-gradient(
            135deg,
            #15803D,
            #16A34A
        );

        color: white;

        border-radius: 10px;

        border: none;

        font-weight: 700;

        padding: 10px 22px;

        transition: all 0.2s ease;
    }


    .stButton > button:hover {

        background: linear-gradient(
            135deg,
            #166534,
            #15803D
        );

        color: white;

        transform: translateY(-1px);

        box-shadow:
            0 5px 12px rgba(21, 128, 61, 0.25);
    }


    /* ======================================================
       DOWNLOAD BUTTONS
       ====================================================== */

    .stDownloadButton > button {

        background-color: #FFFFFF;

        color: #166534;

        border: 1px solid #86EFAC;

        border-radius: 10px;

        font-weight: 650;
    }


    .stDownloadButton > button:hover {

        background-color: #ECFDF5;

        border-color: #22C55E;

        color: #14532D;
    }


    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {

        border-radius: 12px;

        overflow: hidden;

        border: 1px solid #D1FAE5;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {

        text-align: center;

        color: #527064;

        padding: 20px;

        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    FORECAST_MODEL_PATH,
    FORECAST_RESULTS_PATH,
    PREDICTIONS_PATH,
    SHAP_PATH,
    FORECAST_IMPORTANCE_PATH,
    NEXT_YEAR_PATH
]


missing_files = [
    file.name
    for file in required_files
    if not file.exists()
]


if missing_files:

    st.error(
        "⚠️ Some required project files are missing."
    )

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- {file}")

    st.info(
        "Make sure all required files are uploaded "
        "to the same GitHub folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_forecast_model():

    model = joblib.load(
        FORECAST_MODEL_PATH
    )

    return model


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    forecast_results = pd.read_csv(
        FORECAST_RESULTS_PATH
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    shap_importance = pd.read_csv(
        SHAP_PATH
    )

    forecast_importance = pd.read_csv(
        FORECAST_IMPORTANCE_PATH
    )

    next_year = pd.read_csv(
        NEXT_YEAR_PATH
    )

    return (
        forecast_results,
        predictions,
        shap_importance,
        forecast_importance,
        next_year
    )


# ============================================================
# LOAD PROJECT
# ============================================================

try:

    forecast_model = load_forecast_model()

    (
        forecast_results,
        predictions,
        shap_importance,
        forecast_importance,
        next_year
    ) = load_data()

except Exception as error:

    st.error(
        "❌ An error occurred while loading the project."
    )

    st.exception(error)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🌍 AI-Based Carbon Emission Analysis
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        🌱 Intelligent Carbon Analytics • XGBoost • SHAP • CO₂ Forecasting
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "# 🌱 Carbon AI"
)

st.sidebar.write(
    "AI-powered environmental analytics and carbon emission forecasting."
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "🌿 Navigation",
    [
        "🏠 Dashboard",
        "🎯 CO₂ Prediction",
        "🔮 Future Forecast",
        "📊 Model Performance",
        "🧠 Explainable AI",
        "📈 Error Analysis",
        "📥 Download Results"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "🤖 Model: XGBoost\n\n"
    "🧠 Explainability: SHAP\n\n"
    "🔮 Forecasting: XGBoost\n\n"
    "🌍 Domain: Environmental AI"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">'
        '🌿 System Overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-title">
                    MACHINE LEARNING MODEL
                </div>

                <div class="metric-value">
                    XGBoost
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-title">
                    MODEL TYPE
                </div>

                <div class="metric-value">
                    Regression
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-title">
                    EXPLAINABLE AI
                </div>

                <div class="metric-value">
                    SHAP
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-title">
                    FORECASTING
                </div>

                <div class="metric-value">
                    Enabled
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # ABOUT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🌍 About the Project'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-box">

        <b>AI-Based Carbon Emission Analysis System</b>
        uses machine learning and explainable artificial
        intelligence to analyse carbon dioxide emissions.

        <br><br>

        The system uses <b>XGBoost</b> for carbon emission
        prediction, <b>SHAP</b> for model interpretation,
        and XGBoost-based forecasting to estimate future
        CO₂ emissions.

        <br><br>

        The objective is to provide data-driven insights
        that can support environmental monitoring,
        sustainability planning, and carbon management.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # NEXT YEAR FORECAST
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔮 Next-Year Carbon Forecast'
        '</div>',
        unsafe_allow_html=True
    )


    if (
        "Year" in next_year.columns
        and "Predicted_CO2" in next_year.columns
    ):

        forecast_year = int(
            next_year["Year"].iloc[0]
        )

        forecast_value = float(
            next_year["Predicted_CO2"].iloc[0]
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "📅 Forecast Year",
                forecast_year
            )


        with col2:

            st.metric(
                "🌫️ Predicted CO₂",
                f"{forecast_value:,.2f}"
            )


# ============================================================
# CO2 PREDICTION
# ============================================================

elif page == "🎯 CO₂ Prediction":

    st.markdown(
        '<div class="section-title">'
        '🎯 CO₂ Emission Prediction'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-box">

        Enter the required environmental and economic
        parameters to generate a CO₂ emission prediction.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.warning(
        "The interactive CO₂ prediction model is currently "
        "not available because the repository does not contain "
        "the separate final_xgboost_model.pkl file."
    )


    st.info(
        "Your forecasting model and previously generated "
        "prediction results are available in the other dashboard sections."
    )


# ============================================================
# FUTURE FORECAST
# ============================================================

elif page == "🔮 Future Forecast":

    st.markdown(
        '<div class="section-title">'
        '🔮 Future CO₂ Forecast'
        '</div>',
        unsafe_allow_html=True
    )


    if (
        "Year" in next_year.columns
        and "Predicted_CO2" in next_year.columns
    ):

        forecast_year = int(
            next_year["Year"].iloc[0]
        )

        forecast_value = float(
            next_year["Predicted_CO2"].iloc[0]
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "📅 Predicted Year",
                forecast_year
            )


        with col2:

            st.metric(
                "🌫️ Predicted CO₂",
                f"{forecast_value:,.2f}"
            )


    # ========================================================
    # HISTORICAL VS PREDICTED
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Historical vs Predicted CO₂'
        '</div>',
        unsafe_allow_html=True
    )


    fig = go.Figure()


    if (
        "Year" in predictions.columns
        and "Actual_CO2" in predictions.columns
        and "Predicted_CO2" in predictions.columns
    ):

        fig.add_trace(
            go.Scatter(
                x=predictions["Year"],
                y=predictions["Actual_CO2"],
                mode="lines",
                name="Actual CO₂"
            )
        )


        fig.add_trace(
            go.Scatter(
                x=predictions["Year"],
                y=predictions["Predicted_CO2"],
                mode="lines",
                name="Predicted CO₂"
            )
        )


        fig.add_trace(
            go.Scatter(
                x=[forecast_year],
                y=[forecast_value],
                mode="markers",
                marker=dict(
                    size=15
                ),
                name="Next-Year Forecast"
            )
        )


    fig.update_layout(
        template="plotly_white",
        title="CO₂ Emission Forecast",
        xaxis_title="Year",
        yaxis_title="CO₂ Emissions",
        hovermode="x unified"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="section-title">'
        '📊 XGBoost Model Performance'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # PREDICTION RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Prediction Results'
        '</div>',
        unsafe_allow_html=True
    )


    st.dataframe(
        predictions,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    if (
        "Actual_CO2" in predictions.columns
        and "Predicted_CO2" in predictions.columns
    ):

        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                y=predictions["Actual_CO2"],
                mode="lines",
                name="Actual CO₂"
            )
        )


        fig.add_trace(
            go.Scatter(
                y=predictions["Predicted_CO2"],
                mode="lines",
                name="Predicted CO₂"
            )
        )


        fig.update_layout(
            template="plotly_white",
            title="Actual vs Predicted CO₂",
            xaxis_title="Observation",
            yaxis_title="CO₂ Emissions",
            hovermode="x unified"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🌿 Forecast Feature Importance'
        '</div>',
        unsafe_allow_html=True
    )


    st.dataframe(
        forecast_importance,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif page == "🧠 Explainable AI":

    st.markdown(
        '<div class="section-title">'
        '🧠 Explainable AI — SHAP'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-box">

        <b>SHAP (SHapley Additive exPlanations)</b> helps
        explain how individual features influence the
        model's CO₂ emission predictions.

        <br><br>

        A higher mean absolute SHAP value indicates that
        a feature has a stronger influence on the model output.

        </div>
        """,
        unsafe_allow_html=True
    )


    if (
        "Feature" in shap_importance.columns
        and "Mean_Absolute_SHAP" in shap_importance.columns
    ):

        shap_sorted = (
            shap_importance
            .sort_values(
                "Mean_Absolute_SHAP",
                ascending=True
            )
        )


        fig = px.bar(
            shap_sorted,
            x="Mean_Absolute_SHAP",
            y="Feature",
            orientation="h",
            title="🌿 SHAP Feature Importance"
        )


        fig.update_layout(
            template="plotly_white",
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title=""
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.markdown(
        '<div class="section-title">'
        '📋 SHAP Importance Table'
        '</div>',
        unsafe_allow_html=True
    )


    st.dataframe(
        shap_importance,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

elif page == "📈 Error Analysis":

    st.markdown(
        '<div class="section-title">'
        '📈 Prediction Error Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    error_data = predictions.copy()


    # ========================================================
    # CREATE ERROR
    # ========================================================

    if (
        "Error" not in error_data.columns
        and "Actual_CO2" in error_data.columns
        and "Predicted_CO2" in error_data.columns
    ):

        error_data["Error"] = (
            error_data["Actual_CO2"]
            - error_data["Predicted_CO2"]
        )


    if "Error" in error_data.columns:

        error_data["Absolute_Error"] = (
            error_data["Error"].abs()
        )


        # ====================================================
        # ERROR METRICS
        # ====================================================

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "📏 Mean Absolute Error",
                f"{error_data['Absolute_Error'].mean():,.4f}"
            )


        with col2:

            st.metric(
                "⬆️ Maximum Error",
                f"{error_data['Absolute_Error'].max():,.4f}"
            )


        with col3:

            st.metric(
                "⬇️ Minimum Error",
                f"{error_data['Absolute_Error'].min():,.4f}"
            )


        # ====================================================
        # ERROR DISTRIBUTION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Error Distribution'
            '</div>',
            unsafe_allow_html=True
        )


        fig = px.histogram(
            error_data,
            x="Absolute_Error",
            nbins=30,
            title="Prediction Error Distribution"
        )


        fig.update_layout(
            template="plotly_white",
            xaxis_title="Absolute Error",
            yaxis_title="Frequency"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    else:

        st.warning(
            "Actual_CO2 and Predicted_CO2 columns are required "
            "for error analysis."
        )


    # ========================================================
    # ERROR TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Prediction Results'
        '</div>',
        unsafe_allow_html=True
    )


    st.dataframe(
        error_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

elif page == "📥 Download Results":

    st.markdown(
        '<div class="section-title">'
        '📥 Download Project Results'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-box">

        Download the generated machine learning,
        forecasting, feature importance, and SHAP
        results for further analysis.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    st.download_button(
        label="⬇️ Download Prediction Results",
        data=predictions.to_csv(index=False),
        file_name="xgboost_predictions.csv",
        mime="text/csv"
    )


    # ========================================================
    # SHAP
    # ========================================================

    st.download_button(
        label="🧠 Download SHAP Results",
        data=shap_importance.to_csv(index=False),
        file_name="shap_feature_importance.csv",
        mime="text/csv"
    )


    # ========================================================
    # FORECAST
    # ========================================================

    st.download_button(
        label="🔮 Download Forecast Results",
        data=forecast_results.to_csv(index=False),
        file_name="carbon_forecast_results.csv",
        mime="text/csv"
    )


    # ========================================================
    # NEXT YEAR
    # ========================================================

    st.download_button(
        label="📅 Download Next-Year Forecast",
        data=next_year.to_csv(index=False),
        file_name="next_year_co2_prediction.csv",
        mime="text/csv"
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.download_button(
        label="🌿 Download Feature Importance",
        data=forecast_importance.to_csv(index=False),
        file_name="forecast_feature_importance.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

        🌍 <b>AI-Based Carbon Emission Analysis System</b>

        <br><br>

        🌱 XGBoost &nbsp; • &nbsp;
        🧠 Explainable AI &nbsp; • &nbsp;
        🔮 Carbon Forecasting

        <br><br>

        Developed for Academic / Research Project

    </div>
    """,
    unsafe_allow_html=True
)
