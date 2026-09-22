# 💧 Hydro Generation Forecasting System

A machine learning-based hydroelectric power generation forecasting system developed using historical **power generation** and **rainfall** data.

The system analyses historical plant-level generation patterns, rainfall patterns, seasonal behaviour, and lagged generation features to forecast future hydroelectric power generation using **XGBoost Regression**.

The forecasting results are presented through an interactive **Streamlit web application**.

---

## 📌 Project Overview

Hydropower generation is influenced by several factors, particularly rainfall, historical generation behaviour, seasonal patterns, and recent plant performance.

This project aims to develop a data-driven forecasting system that can:

- Analyse historical hydro generation data
- Analyse rainfall patterns
- Identify seasonal generation patterns
- Engineer time-series features
- Train plant-specific XGBoost models
- Evaluate forecasting performance
- Generate future generation forecasts
- Generate next-year generation predictions
- Display forecasts through an interactive Streamlit dashboard
- Allow users to download forecast results

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyse historical generation and rainfall data.
2. Identify relationships between rainfall and hydro generation.
3. Develop machine learning models for generation forecasting.
4. Evaluate model performance using appropriate forecasting metrics.
5. Forecast future generation for individual hydro plants.
6. Provide an interactive dashboard for analysing forecast results.
7. Support generation planning and monitoring through data-driven predictions.

---

## 🏗️ System Architecture

```text
                 Historical Data
                       │
          ┌────────────┴────────────┐
          │                         │
     Generation Data           Rainfall Data
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
               Data Preparation
                       │
                       ▼
             Feature Engineering
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     Generation     Rainfall      Seasonal
       Lags           Lags         Features
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
                XGBoost Model
                       │
                       ▼
              Model Evaluation
                       │
                       ▼
              Future Forecasting
                       │
                       ▼
              Forecast CSV Files
                       │
                       ▼
              Streamlit Dashboard
