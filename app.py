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

    st.error("⚠️ Some required project files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- {file}")

    st.info(
        "Make sure all required files are uploaded "
        "to the same GitHub folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD FORECAST MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        FORECAST_MODEL_PATH
    )


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

    forecast_model = load_model()

    (
        forecast_results,
        predictions,
        shap_importance,
        forecast_importance,
        next_year
    ) = load_data()

except Exception as error:

    st.error(
        "❌ Error while loading project files."
    )

    st.exception(error)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌍 AI-Based Carbon Emission Analysis"
)

st.caption(
    "🌱 Intelligent Carbon Analytics • "
    "XGBoost • SHAP • CO₂ Forecasting"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌱 Carbon AI"
)

st.sidebar.write(
    "AI-powered environmental analytics "
    "and carbon emission forecasting."
)

st.sidebar.divider()

page = st.sidebar.radio(
    "🌿 Navigation",
    [
        "🏠 Dashboard",
        "🔮 Future Forecast",
        "📊 Model Performance",
        "🧠 Explainable AI",
        "📈 Error Analysis",
        "📥 Download Results"
    ]
)

st.sidebar.divider()

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

    st.header(
        "🌿 System Overview"
    )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🤖 Machine Learning Model",
            "XGBoost"
        )


    with col2:

        st.metric(
            "📊 Model Type",
            "Regression"
        )


    with col3:

        st.metric(
            "🧠 Explainable AI",
            "SHAP"
        )


    with col4:

        st.metric(
            "🔮 Forecasting",
            "Enabled"
        )


    # --------------------------------------------------------
    # ABOUT PROJECT
    # --------------------------------------------------------

    st.header(
        "🌍 About the Project"
    )

    st.info(
        """
        **AI-Based Carbon Emission Analysis System**

        This system uses machine learning and explainable
        artificial intelligence to analyse and forecast
        carbon dioxide emissions.

        **XGBoost** is used for carbon emission modelling,
        while **SHAP** provides model explainability.

        The forecasting component estimates future CO₂
        emissions using learned historical patterns.

        The system is designed to support environmental
        monitoring, sustainability planning, and
        carbon management.
        """
    )


    # --------------------------------------------------------
    # NEXT YEAR FORECAST
    # --------------------------------------------------------

    st.header(
        "🔮 Next-Year Carbon Forecast"
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


        st.success(
            f"🌱 The model predicts approximately "
            f"{forecast_value:,.2f} CO₂ units for "
            f"{forecast_year}."
        )


    else:

        st.warning(
            "Year and Predicted_CO2 columns "
            "are not available."
        )


# ============================================================
# FUTURE FORECAST
# ============================================================

elif page == "🔮 Future Forecast":

    st.header(
        "🔮 Future CO₂ Forecast"
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


    # --------------------------------------------------------
    # HISTORICAL VS PREDICTED
    # --------------------------------------------------------

    st.header(
        "📈 Historical vs Predicted CO₂"
    )


    if (
        "Year" in predictions.columns
        and "Actual_CO2" in predictions.columns
        and "Predicted_CO2" in predictions.columns
    ):

        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=predictions["Year"],
                y=predictions["Actual_CO2"],
                mode="lines+markers",
                name="Actual CO₂"
            )
        )


        fig.add_trace(
            go.Scatter(
                x=predictions["Year"],
                y=predictions["Predicted_CO2"],
                mode="lines+markers",
                name="Predicted CO₂"
            )
        )


        if (
            "Year" in next_year.columns
            and "Predicted_CO2" in next_year.columns
        ):

            fig.add_trace(
                go.Scatter(
                    x=[forecast_year],
                    y=[forecast_value],
                    mode="markers",
                    marker=dict(
                        size=16
                    ),
                    name="Next-Year Forecast"
                )
            )


        fig.update_layout(
            title="CO₂ Emission Forecast",
            xaxis_title="Year",
            yaxis_title="CO₂ Emissions",
            hovermode="x unified",
            template="plotly_white"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    else:

        st.warning(
            "Required columns for forecasting graph "
            "are not available."
        )


    # --------------------------------------------------------
    # FORECAST DATA
    # --------------------------------------------------------

    st.header(
        "📋 Forecast Results"
    )

    st.dataframe(
        forecast_results,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.header(
        "📊 XGBoost Model Performance"
    )


    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    if (
        "Actual_CO2" in predictions.columns
        and "Predicted_CO2" in predictions.columns
    ):

        actual = predictions["Actual_CO2"]

        predicted = predictions["Predicted_CO2"]


        mae = np.mean(
            np.abs(
                actual - predicted
            )
        )


        rmse = np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "📏 Mean Absolute Error",
                f"{mae:,.4f}"
            )


        with col2:

            st.metric(
                "📐 Root Mean Square Error",
                f"{rmse:,.4f}"
            )


    # --------------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------------

    st.header(
        "📋 Prediction Results"
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
            title="Actual vs Predicted CO₂",
            xaxis_title="Observation",
            yaxis_title="CO₂ Emissions",
            hovermode="x unified",
            template="plotly_white"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.header(
        "🌿 Forecast Feature Importance"
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

    st.header(
        "🧠 Explainable AI — SHAP"
    )


    st.info(
        """
        **SHAP (SHapley Additive exPlanations)** explains
        how individual features influence the model's
        CO₂ prediction.

        A higher mean absolute SHAP value indicates
        greater influence on the model output.
        """
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
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title="Feature",
            template="plotly_white"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.header(
        "📋 SHAP Importance Table"
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

    st.header(
        "📈 Prediction Error Analysis"
    )


    error_data = predictions.copy()


    if (
        "Actual_CO2" in error_data.columns
        and "Predicted_CO2" in error_data.columns
    ):

        error_data["Error"] = (
            error_data["Actual_CO2"]
            -
            error_data["Predicted_CO2"]
        )


        error_data["Absolute_Error"] = (
            error_data["Error"].abs()
        )


        # ----------------------------------------------------
        # ERROR METRICS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # ERROR DISTRIBUTION
        # ----------------------------------------------------

        st.header(
            "📊 Prediction Error Distribution"
        )


        fig = px.histogram(
            error_data,
            x="Absolute_Error",
            nbins=30,
            title="Prediction Error Distribution"
        )


        fig.update_layout(
            xaxis_title="Absolute Error",
            yaxis_title="Frequency",
            template="plotly_white"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # ERROR DATA
        # ----------------------------------------------------

        st.header(
            "📋 Error Analysis Results"
        )


        st.dataframe(
            error_data,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "Actual_CO2 and Predicted_CO2 columns "
            "are required for error analysis."
        )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

elif page == "📥 Download Results":

    st.header(
        "📥 Download Project Results"
    )


    st.info(
        "Download the generated machine learning, "
        "forecasting, feature importance and SHAP results."
    )


    st.download_button(
        label="⬇️ Download Prediction Results",
        data=predictions.to_csv(index=False),
        file_name="xgboost_predictions.csv",
        mime="text/csv"
    )


    st.download_button(
        label="🧠 Download SHAP Results",
        data=shap_importance.to_csv(index=False),
        file_name="shap_feature_importance.csv",
        mime="text/csv"
    )


    st.download_button(
        label="🔮 Download Forecast Results",
        data=forecast_results.to_csv(index=False),
        file_name="carbon_forecast_results.csv",
        mime="text/csv"
    )


    st.download_button(
        label="📅 Download Next-Year Forecast",
        data=next_year.to_csv(index=False),
        file_name="next_year_co2_prediction.csv",
        mime="text/csv"
    )


    st.download_button(
        label="🌿 Download Feature Importance",
        data=forecast_importance.to_csv(index=False),
        file_name="forecast_feature_importance.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌍 AI-Based Carbon Emission Analysis System | "
    "XGBoost • Explainable AI • Carbon Forecasting"
)
