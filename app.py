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

# All files are stored directly in the GitHub root directory.
FORECAST_MODEL_PATH = BASE_DIR / "carbon_emission_forecasting_xgboost.pkl"

FORECAST_RESULTS_PATH = BASE_DIR / "carbon_forecast_results.csv"
PREDICTIONS_PATH = BASE_DIR / "xgboost_predictions.csv"
SHAP_PATH = BASE_DIR / "shap_feature_importance.csv"
FORECAST_IMPORTANCE_PATH = BASE_DIR / "forecast_feature_importance.csv"
NEXT_YEAR_PATH = BASE_DIR / "next_year_co2_prediction.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #FFF8F0;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #7A3E00;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #765C4A;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        color: #7A3E00;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .metric-card {
        background-color: #FFFFFF;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #F0D7BE;
        box-shadow: 0 4px 12px rgba(100, 60, 20, 0.08);
    }

    .metric-title {
        color: #876F5C;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #A84F00;
        font-size: 28px;
        font-weight: 800;
    }

    .info-box {
        background-color: #FFF0DD;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #D97706;
        color: #5F3B20;
        margin-bottom: 20px;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFF1E2;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #7A3E00;
    }

    .stButton > button {
        background-color: #C65D00;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #9A4600;
        color: white;
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

    st.error("Some required project files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- {file}")

    st.info(
        "Make sure the required files are uploaded "
        "to the same GitHub folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD FORECAST MODEL
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
        "An error occurred while loading the project."
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
        XGBoost • Explainable AI • CO₂ Prediction • Future Forecasting
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
    "AI-powered carbon emission analysis system"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
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
    "Model: XGBoost\n\n"
    "Explainability: SHAP\n\n"
    "Forecasting: XGBoost"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">System Overview</div>',
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


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        The AI-Based Carbon Emission Analysis System uses
        machine learning techniques to analyse and predict
        carbon dioxide emissions.

        The system uses <b>XGBoost</b> for emission prediction,
        <b>SHAP</b> for model explainability, and an
        XGBoost-based forecasting model to estimate future
        carbon emissions.

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # NEXT YEAR FORECAST
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔮 Next-Year Forecast</div>',
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
                "Forecast Year",
                forecast_year
            )

        with col2:

            st.metric(
                "Predicted CO₂",
                f"{forecast_value:,.2f}"
            )


# ============================================================
# CO2 PREDICTION
# ============================================================

elif page == "🎯 CO₂ Prediction":

    st.markdown(
        '<div class="section-title">CO₂ Emission Prediction</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "The interactive CO₂ prediction model is not available "
        "because the repository currently does not contain "
        "the separate final_xgboost_model.pkl file."
    )

    st.info(
        "Your existing XGBoost forecast model and prediction "
        "results are available in the Future Forecast and "
        "Model Performance sections."
    )

    st.markdown(
        """
        <div class="info-box">

        To enable interactive prediction, upload the trained
        prediction model used during your final model training.
        The model must be compatible with the input features
        used during training.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FUTURE FORECAST
# ============================================================

elif page == "🔮 Future Forecast":

    st.markdown(
        '<div class="section-title">Future CO₂ Forecast</div>',
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
                "Predicted Year",
                forecast_year
            )

        with col2:

            st.metric(
                "Predicted CO₂",
                f"{forecast_value:,.2f}"
            )


    # --------------------------------------------------------
    # HISTORICAL VS PREDICTED
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Historical vs Predicted CO₂'
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
                marker=dict(size=14),
                name="Next-Year Forecast"
            )
        )

    fig.update_layout(
        template="simple_white",
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
        'XGBoost Model Performance'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Prediction Results'
        '</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        predictions,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

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
            template="simple_white",
            title="Actual vs Predicted CO₂",
            xaxis_title="Observation",
            yaxis_title="CO₂ Emissions",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # FORECAST FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Forecast Feature Importance'
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
        'Explainable AI — SHAP'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        SHAP (SHapley Additive exPlanations) identifies how
        strongly each feature influences the model prediction.

        A larger mean absolute SHAP value indicates greater
        influence on the model output.

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
            title="SHAP Feature Importance"
        )

        fig.update_layout(
            template="simple_white",
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.markdown(
        '<div class="section-title">'
        'SHAP Importance Table'
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
        'Prediction Error Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    error_data = predictions.copy()


    # --------------------------------------------------------
    # CREATE ERROR
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # ERROR METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Mean Absolute Error",
                f"{error_data['Absolute_Error'].mean():,.4f}"
            )

        with col2:

            st.metric(
                "Maximum Error",
                f"{error_data['Absolute_Error'].max():,.4f}"
            )

        with col3:

            st.metric(
                "Minimum Error",
                f"{error_data['Absolute_Error'].min():,.4f}"
            )


        # ----------------------------------------------------
        # ERROR DISTRIBUTION
        # ----------------------------------------------------

        fig = px.histogram(
            error_data,
            x="Absolute_Error",
            nbins=30,
            title="Prediction Error Distribution"
        )

        fig.update_layout(
            template="simple_white",
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


    # --------------------------------------------------------
    # ERROR TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Prediction Results'
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
        'Download Project Results'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Download the generated model outputs "
        "for further analysis."
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download Prediction Results",
        data=predictions.to_csv(index=False),
        file_name="xgboost_predictions.csv",
        mime="text/csv"
    )


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download SHAP Results",
        data=shap_importance.to_csv(index=False),
        file_name="shap_feature_importance.csv",
        mime="text/csv"
    )


    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download Forecast Results",
        data=forecast_results.to_csv(index=False),
        file_name="carbon_forecast_results.csv",
        mime="text/csv"
    )


    # --------------------------------------------------------
    # NEXT YEAR
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download Next-Year Forecast",
        data=next_year.to_csv(index=False),
        file_name="next_year_co2_prediction.csv",
        mime="text/csv"
    )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download Feature Importance",
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
    <div style="
        text-align:center;
        color:#876F5C;
        padding:15px;
    ">

    🌱 <b>AI-Based Carbon Emission Analysis System</b>

    <br>

    XGBoost • Explainable AI • Carbon Forecasting

    <br><br>

    Developed for Academic / Research Project

    </div>
    """,
    unsafe_allow_html=True
)
