import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/HydroGen_Model_Ready_Dataset.csv"

MODEL_DIR = "models"

OUTPUT_DIR = "outputs"

OUTPUT_PATH = (
    f"{OUTPUT_DIR}/next_year_forecast.csv"
)

HORIZON = 365

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading historical data...")

df = pd.read_csv(
    DATA_PATH
)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Site",
    "Date",
    "Generation_kWh",
    "Rainfall_Input_mm"
]

missing_columns = [
    col
    for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=[
        "Site",
        "Date"
    ]
).copy()


df = df.sort_values(
    [
        "Site",
        "Date"
    ]
).reset_index(
    drop=True
)


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# CREATE HISTORICAL RAINFALL PROFILE
#
# Future rainfall is unknown.
#
# Therefore, use historical median rainfall for the
# corresponding plant + month as a baseline.
# ============================================================

df["Month"] = (
    df["Date"].dt.month
)


rainfall_profile = (

    df.groupby(
        [
            "Site",
            "Month"
        ]
    )[
        "Rainfall_Input_mm"
    ]

    .median()

    .reset_index()

)


# ============================================================
# FORECAST FUNCTION
# ============================================================

def forecast_site(
    site,
    horizon=365
):

    print("\n" + "=" * 70)

    print(
        f"FORECASTING SITE: {site}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # SITE DATA
    # --------------------------------------------------------

    site_df = df[
        df["Site"] == site
    ].copy()


    site_df = site_df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )


    if site_df.empty:

        print(
            f"No historical data found for {site}"
        )

        return pd.DataFrame()


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model_path = (
        f"{MODEL_DIR}/{site}_xgb_model.pkl"
    )


    if not os.path.exists(
        model_path
    ):

        print(
            f"Model not found: {model_path}"
        )

        return pd.DataFrame()


    model = joblib.load(
        model_path
    )


    # --------------------------------------------------------
    # LAST ACTUAL DATE
    # --------------------------------------------------------

    last_date = (
        site_df["Date"].max()
    )


    print(
        f"Last actual date: "
        f"{last_date.strftime('%d %B %Y')}"
    )


    # --------------------------------------------------------
    # FUTURE DATES
    # --------------------------------------------------------

    future_dates = pd.date_range(

        start=(
            last_date
            +
            pd.Timedelta(days=1)
        ),

        periods=horizon,

        freq="D"

    )


    print(
        f"Forecast starts: "
        f"{future_dates.min().strftime('%d %B %Y')}"
    )

    print(
        f"Forecast ends: "
        f"{future_dates.max().strftime('%d %B %Y')}"
    )


    # --------------------------------------------------------
    # HISTORY
    #
    # This history is continuously expanded with predictions.
    # Therefore the model can perform recursive forecasting.
    # --------------------------------------------------------

    history = site_df[
        [
            "Date",
            "Generation_kWh",
            "Rainfall_Input_mm"
        ]
    ].copy()


    history = history.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )


    predictions = []


    # ========================================================
    # RECURSIVE FORECAST
    # ========================================================

    for future_date in future_dates:


        # ====================================================
        # 1. FUTURE RAINFALL
        # ====================================================

        month = (
            future_date.month
        )


        rainfall_row = rainfall_profile[
            (
                rainfall_profile["Site"]
                == site
            )
            &
            (
                rainfall_profile["Month"]
                == month
            )
        ]


        if not rainfall_row.empty:

            rainfall = float(
                rainfall_row[
                    "Rainfall_Input_mm"
                ].iloc[0]
            )

        else:

            rainfall = float(
                history[
                    "Rainfall_Input_mm"
                ]
                .tail(30)
                .median()
            )


        # Protect against NaN

        if pd.isna(rainfall):

            rainfall = 0.0


        # ====================================================
        # 2. GENERATION LAGS
        # ====================================================

        def get_generation_lag(
            days
        ):

            target_date = (
                future_date
                -
                pd.Timedelta(
                    days=days
                )
            )


            values = history.loc[
                history["Date"]
                == target_date,
                "Generation_kWh"
            ]


            if not values.empty:

                return float(
                    values.iloc[-1]
                )


            # Fallback if exact date does not exist

            fallback = (
                history[
                    "Generation_kWh"
                ]
                .tail(days)
                .mean()
            )


            if pd.isna(fallback):

                fallback = (
                    history[
                        "Generation_kWh"
                    ]
                    .mean()
                )


            return float(
                fallback
            )


        lag1 = get_generation_lag(
            1
        )

        lag7 = get_generation_lag(
            7
        )

        lag14 = get_generation_lag(
            14
        )

        lag30 = get_generation_lag(
            30
        )


        # ====================================================
        # 3. GENERATION ROLLING FEATURES
        #
        # IMPORTANT:
        # Only previous observations are used.
        # ====================================================

        previous_generation = (
            history[
                "Generation_kWh"
            ]
            .dropna()
        )


        rolling7 = float(
            previous_generation
            .tail(7)
            .mean()
        )


        rolling30 = float(
            previous_generation
            .tail(30)
            .mean()
        )


        # ====================================================
        # 4. RAINFALL LAGS
        # ====================================================

        def get_rainfall_lag(
            days
        ):

            target_date = (
                future_date
                -
                pd.Timedelta(
                    days=days
                )
            )


            values = history.loc[
                history["Date"]
                == target_date,
                "Rainfall_Input_mm"
            ]


            if not values.empty:

                value = float(
                    values.iloc[-1]
                )

                if not pd.isna(value):

                    return value


            return rainfall


        lag_rain1 = (
            get_rainfall_lag(1)
        )

        lag_rain7 = (
            get_rainfall_lag(7)
        )

        lag_rain14 = (
            get_rainfall_lag(14)
        )

        lag_rain30 = (
            get_rainfall_lag(30)
        )


        # ====================================================
        # 5. RAINFALL ROLLING FEATURES
        #
        # MATCH TRAINING:
        #
        # Rolling7_Rainfall = SUM
        # Rolling30_Rainfall = SUM
        # ====================================================

        previous_rainfall = (
            history[
                "Rainfall_Input_mm"
            ]
            .dropna()
        )


        rolling_rain7 = float(
            previous_rainfall
            .tail(7)
            .sum()
        )


        rolling_rain30 = float(
            previous_rainfall
            .tail(30)
            .sum()
        )


        # ====================================================
        # 6. RAINFALL COVERAGE
        # ====================================================

        rainfall_count_30 = (
            history[
                "Rainfall_Input_mm"
            ]
            .tail(30)
            .notna()
            .sum()
        )


        rainfall_coverage_30d = (
            rainfall_count_30 / 30
        )


        # ====================================================
        # 7. CALENDAR FEATURES
        # ====================================================

        day_of_year = (
            future_date.dayofyear
        )


        week_of_year = int(
            future_date.isocalendar().week
        )


        # ====================================================
        # 8. CYCLICAL FEATURES
        # ====================================================

        sin_day = np.sin(
            2
            *
            np.pi
            *
            day_of_year
            /
            365.25
        )


        cos_day = np.cos(
            2
            *
            np.pi
            *
            day_of_year
            /
            365.25
        )


        # ====================================================
        # 9. CREATE MODEL INPUT
        # ====================================================

        X_future = pd.DataFrame(
            [{
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
                    rainfall_coverage_30d
            }]
        )


        # ====================================================
        # 10. PREDICT
        # ====================================================

        prediction = model.predict(
            X_future[
                FEATURES
            ]
        )[0]


        # Prevent negative generation

        prediction = max(
            0,
            float(prediction)
        )


        # ====================================================
        # 11. STORE FORECAST
        # ====================================================

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


        # ====================================================
        # 12. ADD PREDICTION TO HISTORY
        #
        # This is what makes the forecast recursive.
        # ====================================================

        history = pd.concat(

            [
                history,

                pd.DataFrame(
                    [{
                        "Date":
                            future_date,

                        "Generation_kWh":
                            prediction,

                        "Rainfall_Input_mm":
                            rainfall
                    }]
                )
            ],

            ignore_index=True

        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            len(predictions) == 1
            or
            len(predictions) % 30 == 0
        ):

            print(
                f"{future_date.strftime('%d %b %Y')}"
                f" → "
                f"{prediction:,.2f} kWh"
            )


    # ========================================================
    # RETURN
    # ========================================================

    return pd.DataFrame(
        predictions
    )


# ============================================================
# FORECAST ALL PLANTS
# ============================================================

all_forecasts = []


sites = sorted(
    df["Site"]
    .dropna()
    .unique()
)


print("\n")
print("=" * 70)
print("STARTING 365-DAY FORECAST")
print("=" * 70)


for site in sites:

    result = forecast_site(
        site=site,
        horizon=HORIZON
    )


    if not result.empty:

        all_forecasts.append(
            result
        )


# ============================================================
# COMBINE FORECASTS
# ============================================================

if not all_forecasts:

    raise RuntimeError(
        "No forecasts were generated. "
        "Check whether model .pkl files exist."
    )


forecast_df = pd.concat(
    all_forecasts,
    ignore_index=True
)


forecast_df["Date"] = pd.to_datetime(
    forecast_df["Date"]
)


forecast_df = forecast_df.sort_values(
    [
        "Site",
        "Date"
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE DAILY FORECAST
# ============================================================

forecast_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nDaily forecast saved to:"
    f"\n{OUTPUT_PATH}"
)


# ============================================================
# FORECAST SUMMARIES
# ============================================================

forecast_df["Year"] = (
    forecast_df["Date"].dt.year
)


forecast_df["Month"] = (
    forecast_df["Date"].dt.month
)


forecast_df["Month_Name"] = (
    forecast_df["Date"].dt.strftime(
        "%B"
    )
)


# ============================================================
# MONTHLY FORECAST
# ============================================================

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
    f"{OUTPUT_DIR}/monthly_forecast.csv",
    index=False
)


print(
    "Monthly forecast saved."
)


# ============================================================
# YEARLY FORECAST
# ============================================================

yearly_forecast = (

    forecast_df

    .groupby(
        [
            "Site",
            "Year"
        ]
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
    f"{OUTPUT_DIR}/yearly_forecast.csv",
    index=False
)


print(
    "Yearly forecast saved."
)


# ============================================================
# PLANT FORECAST SUMMARY
# ============================================================

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
    f"{OUTPUT_DIR}/plant_forecast_summary.csv",
    index=False
)


print(
    "Plant forecast summary saved."
)


# ============================================================
# OVERALL FORECAST SUMMARY
# ============================================================

daily_total = (

    forecast_df

    .groupby("Date")[
        "Forecast_Generation_kWh"
    ]

    .sum()

)


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
        daily_total.mean()
    ],

    "Maximum_Daily_Generation_kWh": [
        daily_total.max()
    ],

    "Total_Forecast_Rainfall_mm": [
        forecast_df[
            "Forecast_Rainfall_mm"
        ].sum()
    ]

})


overall_forecast.to_csv(
    f"{OUTPUT_DIR}/overall_forecast_summary.csv",
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FORECAST SUMMARY")
print("=" * 70)


print(
    "\nPlant Forecast Summary:"
)


print(
    plant_forecast_summary[
        [
            "Site",
            "Forecast_Days",
            "Total_Forecast_Generation_kWh",
            "Total_Forecast_Rainfall_mm"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nYearly Forecast:"
)


print(
    yearly_forecast.to_string(
        index=False
    )
)


print(
    "\nOverall Forecast:"
)


print(
    overall_forecast.to_string(
        index=False
    )
)


print("\n")
print(
    "Forecast completed successfully."
)
