import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.plant_locations import PLANTS
from agent.weather_tool import get_weather_forecast
from agent.generation_tool import run_hydro_forecast


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hydro Forecasting Agent",
    page_icon="💧",
    layout="wide",
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GET OPENWEATHER API KEY
# ============================================================

def get_openweather_api_key():
    """
    Get OpenWeather API key.

    Local:
        .env

    Streamlit Cloud:
        st.secrets
    """

    # --------------------------------------------------------
    # First: local .env
    # --------------------------------------------------------

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if api_key:
        return api_key.strip()

    # --------------------------------------------------------
    # Second: Streamlit Secrets
    # --------------------------------------------------------

    try:
        api_key = st.secrets.get("OPENWEATHER_API_KEY")

        if api_key:
            return str(api_key).strip()

    except Exception:
        # No secrets.toml exists locally.
        pass

    return None


OPENWEATHER_API_KEY = get_openweather_api_key()


# Make API key available to weather_tool.py
if OPENWEATHER_API_KEY:
    os.environ["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY


# ============================================================
# HEADER
# ============================================================

st.title("💧 Hydro Forecasting Agent")

st.caption(
    "OpenWeather rainfall forecast → XGBoost generation forecast"
)


# ============================================================
# API KEY VALIDATION
# ============================================================

if not OPENWEATHER_API_KEY:

    st.error(
        "⚠️ OpenWeather API key has not been configured."
    )

    st.markdown(
        """
### For local development

Create a file called:

```text
.env
