import streamlit as st
import pandas as pd


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

    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: 500 !important;
    }

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
        df["Date"],
        errors="coerce"
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
        df["Date"],
        errors="coerce"
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

try:

    historical = load_historical()
    forecast = load_forecast()

except FileNotFoundError as e:

    st.error(
        "Required data file was not found."
    )

    st.code(str(e))

    st.info(
        """
        Please make sure your GitHub repository contains:

        data/HydroGen_Model_Ready_Dataset.csv

        and

        outputs/next_year_forecast.csv
        """
    )

    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

historical_required = [
    "Site",
    "Date",
    "Generation_kWh"
]

forecast_required = [
    "Site",
    "Date",
    "Forecast_Generation_kWh",
    "Forecast_Rainfall_mm"
]

missing_historical = [
    col for col in historical_required
    if col not in historical.columns
]

missing_forecast = [
    col for col in forecast_required
    if col not in forecast.columns
]


if missing_historical:

    st.error(
        "Historical dataset is missing required columns:"
    )

    st.write(missing_historical)

    st.stop()


if missing_forecast:

    st.error(
        "Forecast dataset is missing required columns:"
    )

    st.write(missing_forecast)

    st.stop()


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


if len(plants) == 0:

    st.error(
        "No plant/site information was found."
    )

    st.stop()


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


forecast_site = forecast_site.sort_values(
    "Date"
)


historical_site = historical_site.sort_values(
    "Date"
)


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
            f"{value / 1_000_000_000:,.2f} TWh"
        )

    elif value >= 1_000_000:

        return (
            f"{value / 1_000_000:,.2f} GWh"
        )

    elif value >= 1_000:

        return (
            f"{value / 1_000:,.2f} MWh"
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

    elif value >= 1_000:

        return (
            f"{value / 1_000:,.2f} MWh"
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


with col1:

    st.metric(
        label="Forecast Generation",
        value=format_total_generation(
            total_forecast
        ),
        help="Total predicted generation for the forecast period."
    )


with col2:

    st.metric(
        label="Average Daily Generation",
        value=format_daily_generation(
            average_forecast
        ),
        help="Average predicted generation per day."
    )


with col3:

    st.metric(
        label="Maximum Forecast",
        value=format_daily_generation(
            max_forecast
        ),
        help="Highest predicted daily generation."
    )


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


generation_chart = (
    forecast_site[
        [
            "Date",
            "Forecast_Generation_kWh"
        ]
    ]
    .set_index("Date")
)


st.line_chart(
    generation_chart,
    use_container_width=True
)


# ============================================================
# RAINFALL FORECAST
# ============================================================

st.subheader(
    "🌧️ Forecast Rainfall"
)


rainfall_chart = (
    forecast_site[
        [
            "Date",
            "Forecast_Rainfall_mm"
        ]
    ]
    .set_index("Date")
)


st.line_chart(
    rainfall_chart,
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
        ]
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


monthly_display = monthly.copy()

monthly_display["Period"] = (
    monthly_display["Month"]
    + " "
    + monthly_display["Year"].astype(str)
)


monthly_chart = (
    monthly_display[
        [
            "Period",
            "Forecast_Generation_kWh"
        ]
    ]
    .set_index("Period")
)


st.bar_chart(
    monthly_chart,
    use_container_width=True
)


# ============================================================
# MONTHLY SUMMARY TABLE
# ============================================================

st.subheader(
    "📋 Monthly Forecast Summary"
)


monthly_table = monthly_display[
    [
        "Period",
        "Forecast_Generation_kWh"
    ]
].copy()


monthly_table[
    "Forecast_Generation_kWh"
] = monthly_table[
    "Forecast_Generation_kWh"
].round(2)


monthly_table = monthly_table.rename(
    columns={
        "Period": "Month",
        "Forecast_Generation_kWh":
            "Forecast Generation (kWh)"
    }
)


st.dataframe(
    monthly_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# HISTORICAL GENERATION
# ============================================================

st.subheader(
    "📉 Historical Generation"
)


history_plot = (
    historical_site[
        [
            "Date",
            "Generation_kWh"
        ]
    ]
    .tail(1000)
    .set_index("Date")
)


st.line_chart(
    history_plot,
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


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "Hydro Generation Forecasting System | "
    "Generation & Rainfall Analysis"
)
