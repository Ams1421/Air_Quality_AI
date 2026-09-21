# Air Quality Prediction & Pollution Analysis - AI/ML Project

**Author:** Ram Das Chilakalapudi | **Date:** September 2026

> **Solving a real-world environmental problem:** Predicting urban Air Quality Index (AQI) categories and CO concentrations from low-cost IoT sensor data using machine learning.

**Dataset:** https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set

---

## Problem Statement

Air pollution causes ~7 million premature deaths per year (WHO). Traditional monitoring stations are expensive and sparse. This project trains AI models on multi-sensor data to:
- **Classify** hourly AQI into 6 health categories (Good to Hazardous)
- **Predict** Carbon Monoxide (CO) concentration values
- **Identify** the strongest pollution-driving environmental variables

---

## Submission Files

| File | Format | Purpose |
|---|---|---|
| `RamDasChilakalapudi_AirQualityProject.py` | `.py` | **Complete ML pipeline** - all 7 steps: data loading, EDA, feature engineering, classification, regression, feature importance, summary |
| `requirements.txt` | `.txt` | Python dependencies |
| `RamDasChilakalapudi_ProjectReport.pdf` | `.pdf` | Full IEEE-format project report with embedded charts and screenshots |
| `README.md` | `.md` | This file - project overview, setup instructions, results |

---

## All Project Files

---

## Project Structure

```
air_quality_ai_project/
│
├── RamDasChilakalapudi_AirQualityProject.py  # [SUBMIT] Full ML pipeline (Steps 1-7)
├── requirements.txt                           # [SUBMIT] Python dependencies
├── RamDasChilakalapudi_ProjectReport.docx     # [SUBMIT] IEEE-format project report
├── README.md                                  # [SUBMIT] This file
│
├── app.py                       # Flask web dashboard (run to view live UI)
├── report.html                  # Self-contained HTML report (open in browser)
├── report.pdf                   # PDF version of the IEEE report
│
├── data/
│   └── air_quality_dataset.csv  # UCI Air Quality dataset (94 rows, 16 columns)
│
├── models/
│   ├── best_classifier.pkl      # Decision Tree (100% test accuracy)
│   ├── best_regressor.pkl       # Linear Regression (best RMSE = 1.86)
│   ├── scaler.pkl               # StandardScaler
│   └── label_encoder.pkl        # AQI label encoder
│
└── outputs/                     # 9 generated chart PNGs
    ├── 01_aqi_distribution.png
    ├── 02_co_by_hour.png
    ├── 03_correlation_heatmap.png
    ├── 04_co_by_aqi.png
    ├── 05_temp_vs_no2.png
    ├── 06_confusion_matrix.png
    ├── 07_classifier_comparison.png
    ├── 08_regression_actual_vs_pred.png
    └── 09_feature_importance.png
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

```bash
python RamDasChilakalapudi_AirQualityProject.py
```

This runs all 7 steps: data loading, EDA (5 charts), feature engineering, AQI classification (4 models), CO regression (4 models), feature importance, and summary. Saves trained models to `models/` and charts to `outputs/`.

### 3. Launch Interactive Dashboard (optional)

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

To skip retraining and load saved models directly:

```bash
python app.py serve
```

The dashboard has 5 tabs:
- **Overview** — dataset stats + key charts
- **EDA Charts** — all exploratory visualisations
- **Model Results** — classification & regression tables + charts
- **Live Predict** — enter sensor readings and get real-time AQI + CO prediction
- **Feature Importance** — top pollution drivers

### 4. REST API (programmatic access)

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/predict/classify` | POST | JSON sensor readings | AQI label + confidence |
| `/predict/co` | POST | JSON sensor readings | Predicted CO (mg/m³) |
| `/results` | GET | — | All model metrics as JSON |

**Example request body:**
```json
{
  "CO_GT": 2.6, "C6H6_GT": 11.9, "NOx_GT": 166, "NO2_GT": 113,
  "S1_CO": 1360, "S2_NMHC": 1046, "S3_NOx": 1056,
  "T": 13.6, "RH": 48.9, "AH": 0.7578, "Hour": 18
}
```

---

## Dataset

| Property | Details |
|---|---|
| **Source** | UCI Air Quality — Kaggle |
| **Location** | Road-level monitoring station, Italy |
| **Period** | March – April 2004 (hourly readings) |
| **Records** | 94 cleaned samples |
| **Features** | 15 (sensor + meteorological + engineered) |
| **Targets** | `AQI_Label` (6 classes) + `CO(GT)` (continuous) |

---

## Results Summary

### Classification — AQI Category Prediction

| Model | Test Accuracy | CV Accuracy (5-fold) |
|---|---|---|
| Logistic Regression | 0.8421 | 0.8076 |
| **Decision Tree** ⭐ | **1.0000** | **0.9041** |
| Random Forest | 1.0000 | 0.9146 |
| XGBoost | 1.0000 | 0.9047 |

### Regression — CO Concentration Prediction

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Linear Regression** ⭐ | **1.8647** | **1.2845** | **0.1191** |
| Random Forest Regressor | 2.0921 | 1.5078 | -0.1089 |
| Gradient Boosting | 2.4845 | 1.6798 | -0.5639 |
| XGBoost Regressor | 2.5769 | 1.6782 | -0.6825 |

---

## Key Findings

- **Traffic hours dominate** — CO peaks at 07:00–09:00 and 17:00–19:00 (rush hours)
- **CO & Benzene correlation r ≈ 0.97** — both are vehicle exhaust co-products
- **Hazardous events at moderate temperatures** (15–22°C) due to atmospheric inversion trapping
- **Humidity is the top CO predictor** — AH/RH modulates pollutant dispersion
- **Engineered features add ~3% accuracy** — domain knowledge improves ML performance

---

## Dependencies

```
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, joblib, scipy, flask
```

Install: `pip install -r requirements.txt`

---

## Pipeline Architecture

```
Raw CSV Data
    │
    ├─► Cleaning (replace -200 sentinels, linear interpolation, dropna)
    │
    ├─► EDA (distributions, time trends, correlations) → Charts 1–5
    │
    ├─► Feature Engineering (Pollution_Index, CO_NO2_ratio, Peak_Hour, Weekend)
    │
    ├─► Classification Pipeline → Predict AQI Category (6 classes)
    │       Logistic Regression / Decision Tree / Random Forest / XGBoost
    │
    ├─► Regression Pipeline → Predict CO Concentration
    │       Linear Regression / RF / Gradient Boosting / XGBoost
    │
    └─► Outputs: Models (.pkl) + Charts (.png) + Web Dashboard (app.py)
```

---

## Future Enhancements

- LSTM/GRU for sequential pollution spike prediction
- Real-time REST API deployment with FastAPI + Docker
- Geo-spatial pollution heatmaps across monitoring stations
- Anomaly detection for faulty sensor identification
- Full year dataset (9,358 rows) for seasonal modelling

---

## References

1. De Vito et al. (2008). *On field calibration of an electronic nose for benzene estimation.* Sensors and Actuators B, 129(2), 750–757.
2. UCI Air Quality Dataset: https://archive.ics.uci.edu/ml/datasets/Air+Quality
3. Kaggle Dataset: https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set
4. WHO (2021). *Global Air Quality Guidelines.*
5. Chen & Guestrin (2016). *XGBoost: A scalable tree boosting system.* KDD '16.
6. Breiman (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
7. US EPA (2023). *Air Quality Index — A Guide to Air Quality and Your Health.*

---

*Air Quality AI Project | September 2026 | Built with Python, scikit-learn, XGBoost, Flask*
