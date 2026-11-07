
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Landslide Susceptibility Prediction",
    page_icon="⛰️",
    layout="wide"
)


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


try:
    model = load_model()
    feature_names = load_feature_names()

except Exception as e:
    st.error(f"Error loading model or feature names: {e}")
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("⛰️ Landslide Susceptibility Prediction System")

st.write(
    "An ML-based system to estimate landslide susceptibility "
    "using environmental and terrain characteristics."
)

st.info(
    "Select environmental parameters below to estimate "
    "landslide susceptibility."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):
    st.write("**Algorithm:** Extra Trees Classifier")
    st.write("**Number of Estimators:** 400")
    st.write("**Random State:** 42")
    st.write("**Model:** ExtraTreesClassifier")


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🌍 Environmental & Terrain Inputs")

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    st.subheader("Terrain and Soil Properties")

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

    st.subheader("Water Content and Land Cover")

    vol_water_1500kpa_pct = st.number_input(
        "Volumetric Water - 1500 kPa (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

    distance_to_road_m = st.number_input(
        "Distance to Nearest Road (m)",
        min_value=0.0,
        value=500.0
    )

    soil_moisture = st.number_input(
        "Soil Moisture (m³/m³)",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
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
# LAND COVER ONE-HOT ENCODING
# ============================================================

tree_cover = 1 if land_cover == "Tree cover" else 0
grassland = 1 if land_cover == "Grassland" else 0
cropland = 1 if land_cover == "Cropland" else 0
built_up = 1 if land_cover == "Built-up" else 0
bare_land = 1 if land_cover == "Bare land" else 0


# ============================================================
# PREDICTION SECTION
# ============================================================

st.divider()

st.header("🎯 Landslide Prediction")


texture_sum = (
    clay_content_pct
    + sand_content_pct
    + silt_content_pct
)

if abs(texture_sum - 100.0) > 1.0:
    st.warning(
        f"Clay + Sand + Silt = {texture_sum:.1f}%. "
        "Soil texture fractions normally sum to approximately 100%."
    )


if st.button(
    "🔍 Predict Landslide Susceptibility",
    type="primary",
    use_container_width=True
):

    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

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


    # ========================================================
    # VALIDATE FEATURE NAMES
    # ========================================================

    missing_features = [
        feature for feature in feature_names
        if feature not in input_data.columns
    ]

    if missing_features:

        st.error(
            f"Missing features in input data: {missing_features}"
        )
        st.stop()


    # ========================================================
    # ENSURE CORRECT FEATURE ORDER
    # ========================================================

    input_data = input_data[feature_names]


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        positive_class_index = list(model.classes_).index(1)

        probability = probabilities[positive_class_index]


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.subheader("📊 Prediction Results")

        if prediction == 1:

            st.error(
                "⚠️ High Landslide Susceptibility"
            )

        else:

            st.success(
                "✅ Low Landslide Susceptibility"
            )


        st.metric(
            "Predicted Landslide Probability",
            f"{probability * 100:.2f}%"
        )


        # ====================================================
        # PROBABILITY VISUALIZATION
        # ====================================================

        st.write("### Susceptibility Probability")

        st.progress(float(probability))

        st.caption(
            "Probability represents the model's estimated "
            "likelihood of class 1, not a guaranteed real-world risk."
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        with st.expander("View Input Parameters"):

            st.dataframe(
                input_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(f"Prediction failed: {e}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Landslide Susceptibility Prediction | "
    "Machine Learning Project | IIT Indore"
)
