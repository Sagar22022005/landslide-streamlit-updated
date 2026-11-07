# ⛰️ Landslide Susceptibility Prediction System

A Streamlit application that predicts landslide susceptibility using terrain, rainfall, soil, and land-cover characteristics.

## 🌐 Live App

**[Click here to access the Landslide Susceptibility Prediction System](https://zcayloyq5nueccqsxuzhzf.streamlit.app/)**

## 🤖 Model

ExtraTreesClassifier (400 trees) trained on 16 features.

### Features

- **Terrain:** Elevation, slope
- **Climate:** Annual rainfall
- **Soil:** Clay, sand, silt content, volumetric water at 10, 33 and 1500 kPa, soil moisture
- **Anthropogenic:** Distance to nearest road
- **Land Cover:** Tree cover, grassland, cropland, built-up, bare land (one-hot encoded)

### Prediction Output

The application provides:

- Binary susceptibility classification (High / Low)
- Predicted landslide probability

## 💻 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Files

| File                                 | Purpose                                  |
| ------------------------------------ | ---------------------------------------- |
| `app.py`                             | Streamlit interface and prediction logic |
| `landslide_susceptibility_model.pkl` | Trained Extra Trees model                |
| `feature_names.json`                 | Feature order expected by the model      |
| `requirements.txt`                   | Python dependencies                      |

## 🛠️ Technologies Used

Python, Scikit-learn, Pandas, NumPy, Joblib, Streamlit.

## 🎓 Project

Landslide Susceptibility Prediction and Geospatial Risk Analysis Using Machine Learning.

IIT Indore
