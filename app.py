import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hydro Generation Forecast",
    page_icon="💧",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* KPI value */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
    }

    /* KPI label */
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    /* KPI container */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 16px;
        min-height: 115px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

@st.cache_data
def load_historical():

    df = pd.read_csv(
        "data/HydroGen_Model_Ready_Dataset.csv"
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    return df


# ============================================================
# LOAD FORECAST DATA
# ============================================================

@st.cache_data
def load_forecast():

    df = pd.read_csv(
        "outputs/next_year_forecast.csv"
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

historical = load_historical()

forecast = load_forecast()


# ============================================================
# HEADER
# ============================================================

st.title(
    "💧 Hydro Generation Forecasting System"
)

st.caption(
    "Generation forecasting using historical generation, "
    "rainfall and time-series features"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Forecast Controls"
)

plants = sorted(
    historical["Site"]
    .dropna()
    .unique()
)

selected_site = st.sidebar.selectbox(
    "Select Plant",
    plants
)


# ============================================================
# FILTER DATA
# ============================================================

historical_site = historical[
    historical["Site"] == selected_site
].copy()

forecast_site = forecast[
    forecast["Site"] == selected_site
].copy()


# ============================================================
# CHECK FORECAST
# ============================================================

if forecast_site.empty:

    st.error(
        f"No forecast data available for {selected_site}."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_forecast = (
    forecast_site[
        "Forecast_Generation_kWh"
    ]
    .sum()
)

average_forecast = (
    forecast_site[
        "Forecast_Generation_kWh"
    ]
    .mean()
)

max_forecast = (
    forecast_site[
        "Forecast_Generation_kWh"
    ]
    .max()
)

total_rainfall = (
    forecast_site[
        "Forecast_Rainfall_mm"
    ]
    .sum()
)


# ============================================================
# FORMAT FUNCTIONS
# ============================================================

def format_total_generation(value):

    if value >= 1_000_000_000:

        return (
            f"{value / 1_000_000_000:,.2f} GWh"
        )

    elif value >= 1_000_000:

        return (
            f"{value / 1_000_000:,.2f} GWh"
        )

    else:

        return (
            f"{value:,.0f} kWh"
        )


def format_daily_generation(value):

    if value >= 1_000_000:

        return (
            f"{value / 1_000_000:,.2f} GWh"
        )

    else:

        return (
            f"{value:,.0f} kWh"
        )


# ============================================================
# FORECAST SUMMARY
# ============================================================

st.subheader(
    "📊 Forecast Summary"
)

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# KPI 1
# ------------------------------------------------------------

with col1:

    st.metric(
        label="Forecast Generation",
        value=format_total_generation(
            total_forecast
        ),
        help="Total predicted generation for the forecast period."
    )


# ------------------------------------------------------------
# KPI 2
# ------------------------------------------------------------

with col2:

    st.metric(
        label="Average Daily Generation",
        value=format_daily_generation(
            average_forecast
        ),
        help="Average predicted generation per day."
    )


# ------------------------------------------------------------
# KPI 3
# ------------------------------------------------------------

with col3:

    st.metric(
        label="Maximum Forecast",
        value=format_daily_generation(
            max_forecast
        ),
        help="Highest predicted daily generation."
    )


# ------------------------------------------------------------
# KPI 4
# ------------------------------------------------------------

with col4:

    st.metric(
        label="Forecast Rainfall",
        value=f"{total_rainfall:,.1f} mm",
        help="Total predicted rainfall for the forecast period."
    )


# ============================================================
# FORECAST PERIOD
# ============================================================

forecast_start = forecast_site["Date"].min()
forecast_end = forecast_site["Date"].max()
forecast_days = forecast_site["Date"].nunique()

st.info(
    f"Forecast period: "
    f"{forecast_start.strftime('%d %B %Y')} "
    f"to "
    f"{forecast_end.strftime('%d %B %Y')} "
    f"({forecast_days} days)"
)


# ============================================================
# FUTURE GENERATION FORECAST
# ============================================================

st.subheader(
    "📈 Future Generation Forecast"
)

fig = px.line(
    forecast_site,
    x="Date",
    y="Forecast_Generation_kWh",
    title=f"{selected_site} - Generation Forecast"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Generation (kWh)",
    hovermode="x unified",
    template="plotly_white"
)

fig.update_traces(
    line=dict(width=2)
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# RAINFALL FORECAST
# ============================================================

st.subheader(
    "🌧️ Forecast Rainfall"
)

rain_fig = px.line(
    forecast_site,
    x="Date",
    y="Forecast_Rainfall_mm",
    title=f"{selected_site} - Forecast Rainfall"
)

rain_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Rainfall (mm)",
    hovermode="x unified",
    template="plotly_white"
)

rain_fig.update_traces(
    line=dict(width=2)
)

st.plotly_chart(
    rain_fig,
    use_container_width=True
)


# ============================================================
# MONTHLY FORECAST
# ============================================================

st.subheader(
    "📊 Monthly Generation Forecast"
)

monthly = (
    forecast_site
    .assign(
        Year=lambda x:
            x["Date"].dt.year,

        Month_Number=lambda x:
            x["Date"].dt.month,

        Month=lambda x:
            x["Date"].dt.strftime("%b")
    )
    .groupby(
        [
            "Year",
            "Month_Number",
            "Month"
        ],
        sort=True
    )[
        "Forecast_Generation_kWh"
    ]
    .sum()
    .reset_index()
)

monthly = monthly.sort_values(
    [
        "Year",
        "Month_Number"
    ]
)

monthly_fig = px.bar(
    monthly,
    x="Month",
    y="Forecast_Generation_kWh",
    text_auto=".2s",
    title=f"{selected_site} - Monthly Forecast Generation"
)

monthly_fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Generation (kWh)",
    template="plotly_white"
)

st.plotly_chart(
    monthly_fig,
    use_container_width=True
)


# ============================================================
# HISTORICAL GENERATION
# ============================================================

st.subheader(
    "📉 Historical Generation"
)

history_plot = (
    historical_site
    .sort_values("Date")
    .tail(1000)
)

hist_fig = px.line(
    history_plot,
    x="Date",
    y="Generation_kWh",
    title=f"{selected_site} - Historical Generation"
)

hist_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Generation (kWh)",
    hovermode="x unified",
    template="plotly_white"
)

hist_fig.update_traces(
    line=dict(width=1.8)
)

st.plotly_chart(
    hist_fig,
    use_container_width=True
)


# ============================================================
# FORECAST DETAILS
# ============================================================

st.subheader(
    "📋 Forecast Details"
)

display_df = forecast_site.copy()

display_df[
    "Forecast_Generation_kWh"
] = display_df[
    "Forecast_Generation_kWh"
].round(2)

display_df[
    "Forecast_Rainfall_mm"
] = display_df[
    "Forecast_Rainfall_mm"
].round(2)


# Rename columns

display_df = display_df.rename(
    columns={
        "Site":
            "Plant",

        "Date":
            "Date",

        "Forecast_Rainfall_mm":
            "Forecast Rainfall (mm)",

        "Forecast_Generation_kWh":
            "Forecast Generation (kWh)"
    }
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD
# ============================================================

st.subheader(
    "⬇️ Download Forecast"
)

csv = forecast_site.to_csv(
    index=False
)

st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv,
    file_name=f"{selected_site}_forecast.csv",
    mime="text/csv"
)