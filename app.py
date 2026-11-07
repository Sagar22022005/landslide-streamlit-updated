import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json


# ============================================================
# LOAD MODEL AND FEATURE NAMES
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("landslide_susceptibility_model.pkl")


@st.cache_data
def load_feature_names():
    with open("feature_names.json", "r") as f:
        return json.load(f)


model = load_model()
feature_names = load_feature_names()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Landslide Susceptibility Prediction",
    page_icon="⛰️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⛰️ Landslide Susceptibility Prediction System")

st.write(
    "Enter the environmental and terrain characteristics "
    "to predict landslide susceptibility."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🌍 Environmental & Terrain Inputs")

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    elevation_m = st.number_input(
        "Elevation (m)",
        min_value=0.0,
        value=500.0
    )

    slope_deg = st.number_input(
        "Slope (°)",
        min_value=0.0,
        max_value=90.0,
        value=20.0
    )

    annual_rainfall_mm = st.number_input(
        "Annual Rainfall (mm)",
        min_value=0.0,
        value=1800.0
    )

    clay_content_pct = st.number_input(
        "Clay Content (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0
    )

    sand_content_pct = st.number_input(
        "Sand Content (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    silt_content_pct = st.number_input(
        "Silt Content (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0
    )

    vol_water_10kpa_pct = st.number_input(
        "Volumetric Water - 10 kPa (%)",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    vol_water_33kpa_pct = st.number_input(
        "Volumetric Water - 33 kPa (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    vol_water_1500kpa_pct = st.number_input(
        "Volumetric Water - 1500 kPa (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

    distance_to_road_m = st.number_input(
        "Distance to Road (m)",
        min_value=0.0,
        value=500.0
    )

    soil_moisture = st.number_input(
        "Soil Moisture",
        min_value=0.0,
        value=0.20
    )

    st.subheader("🌳 Land Cover")

    land_cover = st.selectbox(
        "Select Land Cover",
        [
            "Tree cover",
            "Grassland",
            "Cropland",
            "Built-up",
            "Bare land"
        ]
    )


# ============================================================
# CONVERT LAND COVER TO ONE-HOT ENCODING
# ============================================================

tree_cover = 1 if land_cover == "Tree cover" else 0
grassland = 1 if land_cover == "Grassland" else 0
cropland = 1 if land_cover == "Cropland" else 0
built_up = 1 if land_cover == "Built-up" else 0
bare_land = 1 if land_cover == "Bare land" else 0


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

texture_sum = clay_content_pct + sand_content_pct + silt_content_pct

if abs(texture_sum - 100.0) > 1.0:
    st.warning(
        f"Clay + sand + silt = {texture_sum:.1f}%. "
        "Soil texture fractions normally sum to 100%."
    )

if st.button(
    "🔍 Predict Landslide Susceptibility",
    type="primary"
):

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame([{
        "elevation_m": elevation_m,
        "slope_deg": slope_deg,
        "annual_rainfall_mm": annual_rainfall_mm,
        "clay_content_pct": clay_content_pct,
        "sand_content_pct": sand_content_pct,
        "silt_content_pct": silt_content_pct,
        "vol_water_10kpa_pct": vol_water_10kpa_pct,
        "vol_water_33kpa_pct": vol_water_33kpa_pct,
        "vol_water_1500kpa_pct": vol_water_1500kpa_pct,
        "distance_to_road_m": distance_to_road_m,
        "soil_moisture": soil_moisture,
        "tree_cover": tree_cover,
        "grassland": grassland,
        "cropland": cropland,
        "built_up": built_up,
        "bare_land": bare_land
    }])

    # --------------------------------------------------------
    # ENSURE CORRECT FEATURE ORDER
    # --------------------------------------------------------

    input_data = input_data[feature_names]

    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    st.subheader("🎯 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Landslide Susceptibility: HIGH"
        )

    else:

        st.success(
            "✅ Landslide Susceptibility: LOW"
        )

    st.metric(
        "Landslide Probability",
        f"{probability * 100:.2f}%"
    )