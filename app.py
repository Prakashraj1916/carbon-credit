# ============================================================
# NEXT YEAR CARBON FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔮 Next-Year Carbon Forecast'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
        🌱 <b>Future Carbon Emission Forecast</b><br><br>
        The XGBoost forecasting model estimates the expected
        CO₂ emission for the upcoming year based on the
        learned historical emission patterns.
    </div>
    """,
    unsafe_allow_html=True
)


# Check required columns

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


    # --------------------------------------------------------
    # FORECAST CARDS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            """
            <div style="
                background: linear-gradient(
                    135deg,
                    #ECFDF5,
                    #D1FAE5
                );
                padding: 25px;
                border-radius: 18px;
                border: 1px solid #86EFAC;
                text-align: center;
                box-shadow: 0 5px 15px rgba(22,101,52,0.10);
            ">
                <div style="
                    font-size: 15px;
                    font-weight: 700;
                    color: #166534;
                    margin-bottom: 8px;
                ">
                    📅 FORECAST YEAR
                </div>

                <div style="
                    font-size: 38px;
                    font-weight: 800;
                    color: #15803D;
                ">
                    """
            + str(forecast_year)
            + """
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div style="
                background: linear-gradient(
                    135deg,
                    #F0FDFA,
                    #CCFBF1
                );
                padding: 25px;
                border-radius: 18px;
                border: 1px solid #5EEAD4;
                text-align: center;
                box-shadow: 0 5px 15px rgba(13,148,136,0.10);
            ">
                <div style="
                    font-size: 15px;
                    font-weight: 700;
                    color: #115E59;
                    margin-bottom: 8px;
                ">
                    🌫️ PREDICTED CO₂
                </div>

                <div style="
                    font-size: 38px;
                    font-weight: 800;
                    color: #0F766E;
                ">
                    """
            + f"{forecast_value:,.2f}"
            + """
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # FORECAST SUMMARY
    # --------------------------------------------------------

    st.markdown("")

    st.success(
        f"🌱 The XGBoost model predicts approximately "
        f"**{forecast_value:,.2f} CO₂ units** for **{forecast_year}**."
    )

else:

    st.warning(
        "The next-year forecast file does not contain "
        "the required 'Year' and 'Predicted_CO2' columns."
    )

