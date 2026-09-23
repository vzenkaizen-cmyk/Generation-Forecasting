import pandas as pd
import numpy as np
import os
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = "data/HydroGen_Model_Ready_Dataset.csv"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.sort_values(
    ["Site", "Date"]
).reset_index(drop=True)


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
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "Site",
        "Date",
        "Generation_kWh"
    ]
).copy()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(site_df):

    site_df = site_df.copy()

    site_df = site_df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # GENERATION LAGS
    # --------------------------------------------------------

    site_df["Lag1_Generation_kWh"] = (
        site_df["Generation_kWh"].shift(1)
    )

    site_df["Lag7_Generation_kWh"] = (
        site_df["Generation_kWh"].shift(7)
    )

    site_df["Lag14_Generation_kWh"] = (
        site_df["Generation_kWh"].shift(14)
    )

    site_df["Lag30_Generation_kWh"] = (
        site_df["Generation_kWh"].shift(30)
    )


    # --------------------------------------------------------
    # GENERATION ROLLING FEATURES
    #
    # IMPORTANT:
    # shift(1) means today's generation is NOT included.
    # --------------------------------------------------------

    site_df["Rolling7_Generation_kWh"] = (
        site_df["Generation_kWh"]
        .shift(1)
        .rolling(
            window=7,
            min_periods=3
        )
        .mean()
    )

    site_df["Rolling30_Generation_kWh"] = (
        site_df["Generation_kWh"]
        .shift(1)
        .rolling(
            window=30,
            min_periods=7
        )
        .mean()
    )


    # --------------------------------------------------------
    # RAINFALL LAGS
    # --------------------------------------------------------

    site_df["Lag1_Rainfall_mm"] = (
        site_df["Rainfall_Input_mm"].shift(1)
    )

    site_df["Lag7_Rainfall_mm"] = (
        site_df["Rainfall_Input_mm"].shift(7)
    )

    site_df["Lag14_Rainfall_mm"] = (
        site_df["Rainfall_Input_mm"].shift(14)
    )

    site_df["Lag30_Rainfall_mm"] = (
        site_df["Rainfall_Input_mm"].shift(30)
    )


    # --------------------------------------------------------
    # RAINFALL ROLLING FEATURES
    # --------------------------------------------------------

    site_df["Rolling7_Rainfall_mm"] = (
        site_df["Rainfall_Input_mm"]
        .shift(1)
        .rolling(
            window=7,
            min_periods=3
        )
        .sum()
    )

    site_df["Rolling30_Rainfall_mm"] = (
        site_df["Rainfall_Input_mm"]
        .shift(1)
        .rolling(
            window=30,
            min_periods=7
        )
        .sum()
    )


    # --------------------------------------------------------
    # CALENDAR FEATURES
    # --------------------------------------------------------

    site_df["Month"] = (
        site_df["Date"].dt.month
    )

    site_df["DayOfYear"] = (
        site_df["Date"].dt.dayofyear
    )

    site_df["WeekOfYear"] = (
        site_df["Date"].dt.isocalendar().week
        .astype(int)
    )


    # --------------------------------------------------------
    # CYCLICAL SEASONAL FEATURES
    # --------------------------------------------------------

    site_df["Sin_DayOfYear"] = np.sin(
        2 * np.pi *
        site_df["DayOfYear"] / 365.25
    )

    site_df["Cos_DayOfYear"] = np.cos(
        2 * np.pi *
        site_df["DayOfYear"] / 365.25
    )


    # --------------------------------------------------------
    # RAINFALL COVERAGE
    # --------------------------------------------------------

    site_df["Rainfall_Coverage_30D"] = (
        site_df["Rainfall_Input_mm"]
        .shift(1)
        .rolling(
            window=30,
            min_periods=1
        )
        .count()
        / 30
    )


    return site_df


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

TARGET = "Generation_kWh"


# ============================================================
# MODEL PARAMETERS
# ============================================================

MODEL_PARAMETERS = {

    "n_estimators": 1000,

    "learning_rate": 0.03,

    "max_depth": 5,

    "min_child_weight": 5,

    "subsample": 0.8,

    "colsample_bytree": 0.8,

    "reg_alpha": 0.05,

    "reg_lambda": 1.0,

    "objective": "reg:squarederror",

    "random_state": 42,

    "n_jobs": -1
}


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# TRAIN EACH PLANT
# ============================================================

for site in sorted(df["Site"].unique()):

    print("\n")
    print("=" * 70)
    print(f"TRAINING MODEL: {site}")
    print("=" * 70)


    # --------------------------------------------------------
    # FILTER PLANT
    # --------------------------------------------------------

    site_df = df[
        df["Site"] == site
    ].copy()


    site_df = site_df.sort_values(
        "Date"
    ).reset_index(drop=True)


    print(
        "Original rows:",
        len(site_df)
    )


    # --------------------------------------------------------
    # CREATE FEATURES
    # --------------------------------------------------------

    site_df = create_features(
        site_df
    )


    # --------------------------------------------------------
    # REMOVE ROWS WHERE FEATURES
    # ARE NOT AVAILABLE
    # --------------------------------------------------------

    model_df = site_df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()


    print(
        "Usable rows:",
        len(model_df)
    )


    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if len(model_df) < 100:

        print(
            f"Skipping {site}: "
            f"not enough usable observations."
        )

        continue


    # ========================================================
    # TIME-BASED TRAIN / TEST SPLIT
    # ========================================================

    split_index = int(
        len(model_df) * 0.80
    )


    train = model_df.iloc[
        :split_index
    ].copy()


    test = model_df.iloc[
        split_index:
    ].copy()


    X_train = train[FEATURES]

    y_train = train[TARGET]


    X_test = test[FEATURES]

    y_test = test[TARGET]


    print(
        "Training rows:",
        len(train)
    )

    print(
        "Testing rows:",
        len(test)
    )

    print(
        "Training period:",
        train["Date"].min().date(),
        "→",
        train["Date"].max().date()
    )

    print(
        "Testing period:",
        test["Date"].min().date(),
        "→",
        test["Date"].max().date()
    )


    # ========================================================
    # VALIDATION MODEL
    # ========================================================

    validation_model = XGBRegressor(
        **MODEL_PARAMETERS
    )


    validation_model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_test,
                y_test
            )
        ],
        verbose=False
    )


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    predictions = validation_model.predict(
        X_test
    )


    # Prevent negative generation

    predictions = np.maximum(
        predictions,
        0
    )


    # ========================================================
    # METRICS
    # ========================================================

    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    actual_values = y_test.values

    prediction_values = predictions


    non_zero = (
        actual_values != 0
    )


    if np.sum(non_zero) > 0:

        mape = np.mean(
            np.abs(
                (
                    actual_values[non_zero]
                    -
                    prediction_values[non_zero]
                )
                /
                actual_values[non_zero]
            )
        ) * 100

    else:

        mape = np.nan


    # ========================================================
    # PRINT PERFORMANCE
    # ========================================================

    print("\nMODEL PERFORMANCE")

    print(
        f"MAE  : {mae:,.2f} kWh"
    )

    print(
        f"RMSE : {rmse:,.2f} kWh"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    print(
        f"MAPE : {mape:.2f}%"
    )


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    test_results = test[
        [
            "Site",
            "Date",
            TARGET
        ]
    ].copy()


    test_results[
        "Predicted_Generation_kWh"
    ] = predictions


    test_results[
        "Error_kWh"
    ] = (
        test_results[TARGET]
        -
        test_results[
            "Predicted_Generation_kWh"
        ]
    )


    test_results[
        "Absolute_Error_kWh"
    ] = np.abs(
        test_results[
            "Error_kWh"
        ]
    )


    test_results[
        "APE_pct"
    ] = np.where(
        test_results[TARGET] != 0,

        (
            test_results[
                "Absolute_Error_kWh"
            ]
            /
            test_results[TARGET]
        ) * 100,

        np.nan
    )


    test_results.to_csv(
        f"{OUTPUT_DIR}/{site}_test_predictions.csv",
        index=False
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance_df = pd.DataFrame({

        "Feature":
            FEATURES,

        "Importance":
            validation_model.feature_importances_

    })


    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )


    importance_df.to_csv(
        f"{OUTPUT_DIR}/{site}_feature_importance.csv",
        index=False
    )


    # ========================================================
    # TRAIN FINAL MODEL USING ALL DATA
    #
    # IMPORTANT:
    # The final forecasting model should use ALL
    # historical observations available.
    # ========================================================

    print(
        "\nTraining final model using all historical data..."
    )


    X_full = model_df[
        FEATURES
    ]

    y_full = model_df[
        TARGET
    ]


    final_model = XGBRegressor(
        **MODEL_PARAMETERS
    )


    final_model.fit(
        X_full,
        y_full,
        verbose=False
    )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    model_path = (
        f"{MODEL_DIR}/{site}_xgb_model.pkl"
    )


    joblib.dump(
        final_model,
        model_path
    )


    print(
        f"Final model saved: {model_path}"
    )


    # ========================================================
    # SAVE FEATURE LIST
    # ========================================================

    feature_path = (
        f"{MODEL_DIR}/{site}_features.pkl"
    )


    joblib.dump(
        FEATURES,
        feature_path
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results.append({

        "Site":
            site,

        "Training_Rows":
            len(train),

        "Testing_Rows":
            len(test),

        "MAE":
            mae,

        "RMSE":
            rmse,

        "R2":
            r2,

        "MAPE":
            mape

    })


# ============================================================
# MODEL PERFORMANCE SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    f"{OUTPUT_DIR}/model_performance.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("ALL MODELS COMPLETED")
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


print("\n")
print(
    "Models saved in:",
    MODEL_DIR
)


print(
    "Performance saved in:",
    OUTPUT_DIR
)
