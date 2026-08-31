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

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"


PREDICTION_MODEL_PATH = (
    MODEL_DIR / "final_xgboost_model.pkl"
)

FORECAST_MODEL_PATH = (
    MODEL_DIR / "carbon_emission_forecasting_xgboost.pkl"
)

METRICS_PATH = (
    RESULT_DIR / "xgboost_final_metrics.csv"
)

PREDICTIONS_PATH = (
    RESULT_DIR / "xgboost_prediction_results.csv"
)

SHAP_PATH = (
    RESULT_DIR / "shap_feature_importance.csv"
)

FORECAST_IMPORTANCE_PATH = (
    RESULT_DIR / "forecast_feature_importance.csv"
)

NEXT_YEAR_PATH = (
    RESULT_DIR / "next_year_co2_prediction.csv"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background-color: #FFF8F0;
    }

    /* Main container */
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #7A3E00;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        font-size: 18px;
        color: #765C4A;
        margin-bottom: 30px;
    }

    /* Section heading */
    .section-title {
        font-size: 28px;
        font-weight: 750;
        color: #7A3E00;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Cards */
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

    /* Information box */
    .info-box {
        background-color: #FFF0DD;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #D97706;
        color: #5F3B20;
        margin-bottom: 20px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFF1E2;
    }

    /* Sidebar title */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #7A3E00;
    }

    /* Buttons */
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
    PREDICTION_MODEL_PATH,
    FORECAST_MODEL_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    SHAP_PATH,
    FORECAST_IMPORTANCE_PATH,
    NEXT_YEAR_PATH
]


missing_files = [
    str(file)
    for file in required_files
    if not file.exists()
]


if missing_files:

    st.error("Some required project files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- {file}")

    st.info(
        "Make sure the models are inside the models folder "
        "and the CSV files are inside the results folder."
    )

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    prediction_model = joblib.load(
        PREDICTION_MODEL_PATH
    )

    forecast_model = joblib.load(
        FORECAST_MODEL_PATH
    )

    return prediction_model, forecast_model


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    metrics = pd.read_csv(
        METRICS_PATH
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
        metrics,
        predictions,
        shap_importance,
        forecast_importance,
        next_year
    )


# ============================================================
# LOAD PROJECT
# ============================================================

try:

    prediction_model, forecast_model = load_models()

    (
        metrics,
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
        '<div class="section-title">'
        'System Overview'
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


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'About the Project'
        '</div>',
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
    # NEXT YEAR
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🔮 Next-Year Forecast'
        '</div>',
        unsafe_allow_html=True
    )

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
        '<div class="section-title">'
        'CO₂ Emission Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        Enter the same features used during XGBoost model
        training. The model will calculate the predicted
        CO₂ emission.

        </div>
        """,
        unsafe_allow_html=True
    )


    # Get feature names from model

    try:

        feature_names = (
            prediction_model
            .get_booster()
            .feature_names
        )

    except Exception:

        feature_names = None


    if feature_names is None:

        st.error(
            "The trained model does not contain feature names."
        )

        st.info(
            "The model must be saved with the selected "
            "feature names used during training."
        )

        st.stop()


    # --------------------------------------------------------
    # INPUT FIELDS
    # --------------------------------------------------------

    user_inputs = {}

    input_columns = st.columns(2)

    for index, feature in enumerate(feature_names):

        with input_columns[index % 2]:

            user_inputs[feature] = st.number_input(
                label=str(feature),
                value=0.0,
                format="%.4f",
                key=f"input_{index}"
            )


    st.markdown("")


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict CO₂ Emission"
    ):

        try:

            input_data = pd.DataFrame(
                [user_inputs],
                columns=feature_names
            )

            prediction = prediction_model.predict(
                input_data
            )[0]

            st.success(
                "CO₂ prediction generated successfully."
            )

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-title">
                        PREDICTED CO₂ EMISSION
                    </div>

                    <div class="metric-value">
                        {prediction:,.4f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(error)


# ============================================================
# FUTURE FORECAST
# ============================================================

elif page == "🔮 Future Forecast":

    st.markdown(
        '<div class="section-title">'
        'Future CO₂ Forecast'
        '</div>',
        unsafe_allow_html=True
    )

    forecast_year = int(
        next_year["Year"].iloc[0]
    )

    forecast_value = float(
        next_year["Predicted_CO2"].iloc[0]
    )


    # --------------------------------------------------------
    # FORECAST METRICS
    # --------------------------------------------------------

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
    # HISTORICAL FORECAST GRAPH
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Historical vs Predicted CO₂'
        '</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure()


    if "Year" in predictions.columns:

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
                size=14
            ),
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
    # METRICS TABLE
    # --------------------------------------------------------

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # METRIC GRAPH
    # --------------------------------------------------------

    if (
        "Metric" in metrics.columns
        and "Value" in metrics.columns
    ):

        fig = px.bar(
            metrics,
            x="Metric",
            y="Value",
            title="Model Evaluation Metrics"
        )

        fig.update_layout(
            template="simple_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Actual vs Predicted CO₂'
        '</div>',
        unsafe_allow_html=True
    )


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
        and "Mean_Absolute_SHAP"
        in shap_importance.columns
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


    # Create error if necessary

    if (
        "Error" not in error_data.columns
        and "Actual_CO2" in error_data.columns
        and "Predicted_CO2" in error_data.columns
    ):

        error_data["Error"] = (
            error_data["Actual_CO2"]
            -
            error_data["Predicted_CO2"]
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
    # MODEL METRICS
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download Model Metrics",
        data=metrics.to_csv(index=False),
        file_name="xgboost_model_metrics.csv",
        mime="text/csv"
    )


    # --------------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download Prediction Results",
        data=predictions.to_csv(index=False),
        file_name="xgboost_prediction_results.csv",
        mime="text/csv"
    )


    # --------------------------------------------------------
    # SHAP RESULTS
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download SHAP Results",
        data=shap_importance.to_csv(index=False),
        file_name="shap_feature_importance.csv",
        mime="text/csv"
    )


    # --------------------------------------------------------
    # FORECAST RESULTS
    # --------------------------------------------------------

    st.download_button(
        label="⬇️ Download Forecast Results",
        data=next_year.to_csv(index=False),
        file_name="next_year_co2_prediction.csv",
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
