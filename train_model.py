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

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_PATH = "data/HydroGen_Model_Ready_Dataset.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

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

TARGET = "Generation_kWh"

# --------------------------------------------------
# MODEL RESULTS
# --------------------------------------------------

results = []

# --------------------------------------------------
# TRAIN MODEL FOR EACH PLANT
# --------------------------------------------------

for site in sorted(df["Site"].unique()):

    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {site}")
    print("=" * 60)

    site_df = df[
        df["Site"] == site
    ].copy()

    site_df = site_df.sort_values("Date")

    # Remove invalid rows
    site_df = site_df.dropna(
        subset=FEATURES + [TARGET]
    )

    # --------------------------------------------------
    # TIME BASED SPLIT
    # --------------------------------------------------

    split_index = int(
        len(site_df) * 0.80
    )

    train = site_df.iloc[:split_index]

    test = site_df.iloc[split_index:]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("Training rows:", len(train))
    print("Testing rows:", len(test))

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_test, y_test)
        ],
        verbose=False
    )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    predictions = model.predict(X_test)

    predictions = np.maximum(
        predictions,
        0
    )

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

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

    # MAPE
    non_zero = y_test != 0

    if non_zero.sum() > 0:

        mape = np.mean(
            np.abs(
                (
                    y_test[non_zero]
                    -
                    predictions[non_zero]
                )
                /
                y_test[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    model_path = (
        f"{MODEL_DIR}/{site}_xgb_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved: {model_path}"
    )

    # --------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------

    test_results = test[
        ["Site", "Date", TARGET]
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
        test_results["Error_kWh"]
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
        f"outputs/{site}_test_predictions.csv",
        index=False
    )

    results.append({
        "Site": site,
        "Training_Rows": len(train),
        "Testing_Rows": len(test),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    })

# --------------------------------------------------
# SAVE MODEL PERFORMANCE
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/model_performance.csv",
    index=False
)

print("\n" + "=" * 60)
print("ALL MODELS COMPLETED")
print("=" * 60)

print(results_df)
