# ⛰️ Landslide Susceptibility Prediction System

A Streamlit app that predicts landslide susceptibility from terrain,
rainfall, soil and land-cover characteristics.

## Live app

<!-- paste your Streamlit Cloud URL here after deploying -->

## Model

ExtraTreesClassifier (400 trees) trained on 16 features:

- **Terrain:** elevation, slope
- **Climate:** annual rainfall
- **Soil:** clay / sand / silt content, volumetric water at 10, 33 and
  1500 kPa, soil moisture
- **Anthropogenic:** distance to road
- **Land cover:** tree cover, grassland, cropland, built-up, bare land
  (one-hot encoded)

Outputs a binary susceptibility class plus a probability score.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit interface and prediction logic |
| `landslide_susceptibility_model.pkl` | Trained model (compressed) |
| `feature_names.json` | Feature order expected by the model |
| `requirements.txt` | Python dependencies |
