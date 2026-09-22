import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/HydroGen_Model_Ready_Dataset.csv"

os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

print("=" * 70)
print("HYDRO GENERATION DATA ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nNumber of plants:")
print(df["Site"].nunique())

print("\nPlants:")
print(sorted(df["Site"].unique()))

# --------------------------------------------------
# BASIC STATISTICS
# --------------------------------------------------

print("\nGeneration statistics:")
print(
    df["Generation_kWh"].describe()
)

print("\nRainfall statistics:")
print(
    df["Rainfall_Input_mm"].describe()
)

# --------------------------------------------------
# PLANT SUMMARY
# --------------------------------------------------

plant_summary = df.groupby("Site").agg(
    Records=("Date", "count"),
    Start_Date=("Date", "min"),
    End_Date=("Date", "max"),
    Total_Generation_kWh=("Generation_kWh", "sum"),
    Average_Generation_kWh=("Generation_kWh", "mean"),
    Maximum_Generation_kWh=("Generation_kWh", "max"),
    Average_Rainfall_mm=("Rainfall_Input_mm", "mean"),
    Total_Rainfall_mm=("Rainfall_Input_mm", "sum"),
    Average_Plant_Factor=("Plant_Factor_pct", "mean")
).reset_index()

print("\nPlant summary:")
print(plant_summary)

plant_summary.to_csv(
    "outputs/plant_summary.csv",
    index=False
)

# --------------------------------------------------
# YEARLY ANALYSIS
# --------------------------------------------------

yearly = df.groupby(["Site", "Year"]).agg(
    Generation_kWh=("Generation_kWh", "sum"),
    Average_Generation_kWh=("Generation_kWh", "mean"),
    Rainfall_mm=("Rainfall_Input_mm", "sum"),
    Average_Plant_Factor=("Plant_Factor_pct", "mean")
).reset_index()

yearly.to_csv(
    "outputs/yearly_analysis.csv",
    index=False
)

# --------------------------------------------------
# MONTHLY ANALYSIS
# --------------------------------------------------

monthly = df.groupby(
    ["Site", "Year", "Month"]
).agg(
    Generation_kWh=("Generation_kWh", "sum"),
    Rainfall_mm=("Rainfall_Input_mm", "sum"),
    Average_Plant_Factor=("Plant_Factor_pct", "mean")
).reset_index()

monthly.to_csv(
    "outputs/monthly_analysis.csv",
    index=False
)

# --------------------------------------------------
# GENERATION TREND
# --------------------------------------------------

for site in df["Site"].unique():

    site_df = df[df["Site"] == site].sort_values("Date")

    plt.figure(figsize=(14, 5))

    plt.plot(
        site_df["Date"],
        site_df["Generation_kWh"]
    )

    plt.title(
        f"{site} - Daily Generation"
    )

    plt.xlabel("Date")
    plt.ylabel("Generation (kWh)")

    plt.tight_layout()

    plt.savefig(
        f"outputs/{site}_generation_trend.png",
        dpi=150
    )

    plt.close()

# --------------------------------------------------
# RAINFALL TREND
# --------------------------------------------------

for site in df["Site"].unique():

    site_df = df[df["Site"] == site].sort_values("Date")

    plt.figure(figsize=(14, 5))

    plt.plot(
        site_df["Date"],
        site_df["Rainfall_Input_mm"]
    )

    plt.title(
        f"{site} - Rainfall"
    )

    plt.xlabel("Date")
    plt.ylabel("Rainfall (mm)")

    plt.tight_layout()

    plt.savefig(
        f"outputs/{site}_rainfall_trend.png",
        dpi=150
    )

    plt.close()

# --------------------------------------------------
# GENERATION VS RAINFALL
# --------------------------------------------------

correlation = df[
    [
        "Generation_kWh",
        "Rainfall_Input_mm",
        "Lag1_Generation_kWh",
        "Lag7_Generation_kWh",
        "Rolling7_Generation_kWh"
    ]
].corr()

print("\nCorrelation:")
print(correlation)

correlation.to_csv(
    "outputs/correlation.csv"
)

# --------------------------------------------------
# MONTHLY SEASONAL PATTERN
# --------------------------------------------------

seasonal = df.groupby(
    ["Site", "Month"]
).agg(
    Average_Generation=("Generation_kWh", "mean"),
    Average_Rainfall=("Rainfall_Input_mm", "mean")
).reset_index()

seasonal.to_csv(
    "outputs/seasonal_analysis.csv",
    index=False
)

print("\nAnalysis completed.")
