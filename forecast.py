import pandas as pd
import numpy as np
import joblib
import os

DATA_PATH = "data/HydroGen_Model_Ready_Dataset.csv"
MODEL_DIR = "models"

OUTPUT_PATH = "outputs/next_year_forecast.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Site", "Date"]
).reset_index(drop=True)

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

FEATURES = [
    "Rainfall_Input_mm",

    "Lag1_Generation_kWh",
    "Lag7_Generation_kWh",
    "Lag14_Generation_kWh",
    "Lag30_Generation_kWh",

    "Rolling7_Generation_kWh",
    "Rolling30_Generation_kWh",

    "Lag1_Rainfall_mm",
    "Lag7_Rainfall_mm",
    "Lag14_Rainfall_mm",
    "Lag30_Rainfall_mm",

    "Rolling7_Rainfall_mm",
    "Rolling30_Rainfall_mm",

    "Month",
    "DayOfYear",
    "WeekOfYear",

    "Sin_DayOfYear",
    "Cos_DayOfYear",

    "Rainfall_Coverage_30D"
]

# --------------------------------------------------
# CREATE SEASONAL RAINFALL PROFILE
# --------------------------------------------------

df["Month"] = df["Date"].dt.month

rainfall_profile = (
    df.groupby(
        ["Site", "Month"]
    )["Rainfall_Input_mm"]
    .median()
    .reset_index()
)

# --------------------------------------------------
# FORECAST FUNCTION
# --------------------------------------------------

def forecast_site(
    site,
    horizon=365
):

    site_df = df[
        df["Site"] == site
    ].copy()

    site_df = site_df.sort_values(
        "Date"
    ).reset_index(drop=True)

    model_path = (
        f"{MODEL_DIR}/{site}_xgb_model.pkl"
    )

    if not os.path.exists(model_path):

        print(
            f"No model found for {site}"
        )

        return pd.DataFrame()

    model = joblib.load(
        model_path
    )

    # Future dates

    last_date = site_df["Date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    # History used for lag calculations

    history = site_df[
        [
            "Date",
            "Generation_kWh",
            "Rainfall_Input_mm"
        ]
    ].copy()

    predictions = []

    # --------------------------------------------------
    # RECURSIVE FORECAST
    # --------------------------------------------------

    for future_date in future_dates:

        # ------------------------------------------
        # RAINFALL ESTIMATE
        # ------------------------------------------

        month = future_date.month

        rainfall_row = rainfall_profile[
            (
                rainfall_profile["Site"] == site
            )
            &
            (
                rainfall_profile["Month"] == month
            )
        ]

        if len(rainfall_row) > 0:

            rainfall = float(
                rainfall_row[
                    "Rainfall_Input_mm"
                ].iloc[0]
            )

        else:

            rainfall = float(
                history[
                    "Rainfall_Input_mm"
                ].tail(30).median()
            )

        # ------------------------------------------
        # GENERATION LAGS
        # ------------------------------------------

        def get_generation_lag(days):

            target_date = (
                future_date
                -
                pd.Timedelta(days=days)
            )

            values = history.loc[
                history["Date"] == target_date,
                "Generation_kWh"
            ]

            if len(values) > 0:
                return float(values.iloc[0])

            return float(
                history[
                    "Generation_kWh"
                ].tail(days).mean()
            )

        lag1 = get_generation_lag(1)

        lag7 = get_generation_lag(7)

        lag14 = get_generation_lag(14)

        lag30 = get_generation_lag(30)

        # ------------------------------------------
        # ROLLING GENERATION
        # ------------------------------------------

        rolling7 = (
            history[
                "Generation_kWh"
            ]
            .tail(7)
            .mean()
        )

        rolling30 = (
            history[
                "Generation_kWh"
            ]
            .tail(30)
            .mean()
        )

        # ------------------------------------------
        # RAINFALL LAGS
        # ------------------------------------------

        def get_rainfall_lag(days):

            target_date = (
                future_date
                -
                pd.Timedelta(days=days)
            )

            values = history.loc[
                history["Date"] == target_date,
                "Rainfall_Input_mm"
            ]

            if len(values) > 0:

                return float(
                    values.iloc[0]
                )

            return rainfall

        lag_rain1 = get_rainfall_lag(1)

        lag_rain7 = get_rainfall_lag(7)

        lag_rain14 = get_rainfall_lag(14)

        lag_rain30 = get_rainfall_lag(30)

        rolling_rain7 = (
            history[
                "Rainfall_Input_mm"
            ]
            .tail(7)
            .mean()
        )

        rolling_rain30 = (
            history[
                "Rainfall_Input_mm"
            ]
            .tail(30)
            .mean()
        )

        # ------------------------------------------
        # SEASONAL FEATURES
        # ------------------------------------------

        day_of_year = (
            future_date.dayofyear
        )

        week_of_year = (
            future_date.isocalendar().week
        )

        sin_day = np.sin(
            2 * np.pi *
            day_of_year / 365.25
        )

        cos_day = np.cos(
            2 * np.pi *
            day_of_year / 365.25
        )

        # ------------------------------------------
        # MODEL INPUT
        # ------------------------------------------

        X_future = pd.DataFrame([{

            "Rainfall_Input_mm":
                rainfall,

            "Lag1_Generation_kWh":
                lag1,

            "Lag7_Generation_kWh":
                lag7,

            "Lag14_Generation_kWh":
                lag14,

            "Lag30_Generation_kWh":
                lag30,

            "Rolling7_Generation_kWh":
                rolling7,

            "Rolling30_Generation_kWh":
                rolling30,

            "Lag1_Rainfall_mm":
                lag_rain1,

            "Lag7_Rainfall_mm":
                lag_rain7,

            "Lag14_Rainfall_mm":
                lag_rain14,

            "Lag30_Rainfall_mm":
                lag_rain30,

            "Rolling7_Rainfall_mm":
                rolling_rain7,

            "Rolling30_Rainfall_mm":
                rolling_rain30,

            "Month":
                future_date.month,

            "DayOfYear":
                day_of_year,

            "WeekOfYear":
                week_of_year,

            "Sin_DayOfYear":
                sin_day,

            "Cos_DayOfYear":
                cos_day,

            "Rainfall_Coverage_30D":
                1.0

        }])

        # ------------------------------------------
        # PREDICT
        # ------------------------------------------

        prediction = model.predict(
            X_future[FEATURES]
        )[0]

        prediction = max(
            0,
            prediction
        )

        # ------------------------------------------
        # STORE
        # ------------------------------------------

        predictions.append({

            "Site":
                site,

            "Date":
                future_date,

            "Forecast_Rainfall_mm":
                rainfall,

            "Forecast_Generation_kWh":
                prediction

        })

        # ------------------------------------------
        # ADD PREDICTION TO HISTORY
        # ------------------------------------------

        history = pd.concat(
            [
                history,
                pd.DataFrame([{
                    "Date":
                        future_date,

                    "Generation_kWh":
                        prediction,

                    "Rainfall_Input_mm":
                        rainfall
                }])
            ],
            ignore_index=True
        )

    return pd.DataFrame(
        predictions
    )


# --------------------------------------------------
# ALL PLANTS
# --------------------------------------------------

all_forecasts = []

for site in sorted(
    df["Site"].unique()
):

    print(
        f"Forecasting {site}..."
    )

    result = forecast_site(
        site,
        horizon=365
    )

    if not result.empty:

        all_forecasts.append(
            result
        )

# --------------------------------------------------
# COMBINE ALL FORECASTS
# --------------------------------------------------

forecast_df = pd.concat(
    all_forecasts,
    ignore_index=True
)

forecast_df["Date"] = pd.to_datetime(
    forecast_df["Date"]
)

# --------------------------------------------------
# 1. SAVE DAILY FORECAST
# --------------------------------------------------

forecast_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nDaily forecast saved to: {OUTPUT_PATH}"
)


# ==================================================
# FORECAST SUMMARIES
# ==================================================

# --------------------------------------------------
# 2. ADD YEAR AND MONTH
# --------------------------------------------------

forecast_df["Year"] = (
    forecast_df["Date"].dt.year
)

forecast_df["Month"] = (
    forecast_df["Date"].dt.month
)

forecast_df["Month_Name"] = (
    forecast_df["Date"].dt.strftime("%B")
)


# --------------------------------------------------
# 3. MONTHLY FORECAST SUMMARY
# --------------------------------------------------

monthly_forecast = (
    forecast_df
    .groupby(
        [
            "Site",
            "Year",
            "Month",
            "Month_Name"
        ]
    )
    .agg(
        Forecast_Generation_kWh=(
            "Forecast_Generation_kWh",
            "sum"
        ),

        Average_Daily_Generation_kWh=(
            "Forecast_Generation_kWh",
            "mean"
        ),

        Maximum_Daily_Generation_kWh=(
            "Forecast_Generation_kWh",
            "max"
        ),

        Minimum_Daily_Generation_kWh=(
            "Forecast_Generation_kWh",
            "min"
        ),

        Forecast_Rainfall_mm=(
            "Forecast_Rainfall_mm",
            "sum"
        ),

        Average_Daily_Rainfall_mm=(
            "Forecast_Rainfall_mm",
            "mean"
        )
    )
    .reset_index()
)

monthly_forecast.to_csv(
    "outputs/monthly_forecast.csv",
    index=False
)

print(
    "Monthly forecast saved to: "
    "outputs/monthly_forecast.csv"
)


# --------------------------------------------------
# 4. YEARLY FORECAST SUMMARY
# --------------------------------------------------

yearly_forecast = (
    forecast_df
    .groupby(
        ["Site", "Year"]
    )
    .agg(
        Total_Forecast_Generation_kWh=(
            "Forecast_Generation_kWh",
            "sum"
        ),

        Average_Daily_Generation_kWh=(
            "Forecast_Generation_kWh",
            "mean"
        ),

        Maximum_Daily_Generation_kWh=(
            "Forecast_Generation_kWh",
            "max"
        ),

        Minimum_Daily_Generation_kWh=(
            "Forecast_Generation_kWh",
            "min"
        ),

        Total_Forecast_Rainfall_mm=(
            "Forecast_Rainfall_mm",
            "sum"
        ),

        Average_Daily_Rainfall_mm=(
            "Forecast_Rainfall_mm",
            "mean"
        )
    )
    .reset_index()
)

yearly_forecast.to_csv(
    "outputs/yearly_forecast.csv",
    index=False
)

print(
    "Yearly forecast saved to: "
    "outputs/yearly_forecast.csv"
)


# --------------------------------------------------
# 5. PLANT FORECAST SUMMARY
# --------------------------------------------------

plant_forecast_summary = (
    forecast_df
    .groupby("Site")
    .agg(
        Forecast_Days=(
            "Date",
            "count"
        ),

        Forecast_Start_Date=(
            "Date",
            "min"
        ),

        Forecast_End_Date=(
            "Date",
            "max"
        ),

        Total_Forecast_Generation_kWh=(
            "Forecast_Generation_kWh",
            "sum"
        ),

        Average_Daily_Generation_kWh=(
            "Forecast_Generation_kWh",
            "mean"
        ),

        Maximum_Daily_Generation_kWh=(
            "Forecast_Generation_kWh",
            "max"
        ),

        Minimum_Daily_Generation_kWh=(
            "Forecast_Generation_kWh",
            "min"
        ),

        Total_Forecast_Rainfall_mm=(
            "Forecast_Rainfall_mm",
            "sum"
        ),

        Average_Daily_Rainfall_mm=(
            "Forecast_Rainfall_mm",
            "mean"
        )
    )
    .reset_index()
)

plant_forecast_summary.to_csv(
    "outputs/plant_forecast_summary.csv",
    index=False
)

print(
    "Plant forecast summary saved to: "
    "outputs/plant_forecast_summary.csv"
)


# --------------------------------------------------
# 6. OVERALL FORECAST SUMMARY
# --------------------------------------------------

overall_forecast = pd.DataFrame({

    "Forecast_Start_Date": [
        forecast_df["Date"].min()
    ],

    "Forecast_End_Date": [
        forecast_df["Date"].max()
    ],

    "Number_of_Plants": [
        forecast_df["Site"].nunique()
    ],

    "Forecast_Days": [
        forecast_df["Date"].nunique()
    ],

    "Total_Forecast_Generation_kWh": [
        forecast_df[
            "Forecast_Generation_kWh"
        ].sum()
    ],

    "Average_Daily_Generation_kWh": [
        forecast_df
        .groupby("Date")[
            "Forecast_Generation_kWh"
        ]
        .sum()
        .mean()
    ],

    "Maximum_Daily_Generation_kWh": [
        forecast_df
        .groupby("Date")[
            "Forecast_Generation_kWh"
        ]
        .sum()
        .max()
    ],

    "Total_Forecast_Rainfall_mm": [
        forecast_df[
            "Forecast_Rainfall_mm"
        ].sum()
    ]
})

overall_forecast.to_csv(
    "outputs/overall_forecast_summary.csv",
    index=False
)

print(
    "Overall forecast summary saved to: "
    "outputs/overall_forecast_summary.csv"
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)

print("FORECAST SUMMARY")

print("=" * 70)

print("\nPlant Forecast Summary:")

print(
    plant_forecast_summary[
        [
            "Site",
            "Forecast_Days",
            "Total_Forecast_Generation_kWh",
            "Total_Forecast_Rainfall_mm"
        ]
    ]
)

print("\nYearly Forecast:")

print(
    yearly_forecast
)

print("\nMonthly Forecast:")

print(
    monthly_forecast.head(20)
)

print("\nOverall Forecast:")

print(
    overall_forecast
)

print("\nForecast completed successfully.")