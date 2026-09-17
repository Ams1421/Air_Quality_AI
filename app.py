"""
=======================================================================
  AIR QUALITY AI PROJECT — Single-File Complete Application
  Includes : Full ML pipeline (EDA + training + evaluation)
           + Flask REST API (/predict/classify, /predict/co, /results)
           + Embedded HTML/CSS/JS dashboard frontend

  Run      : python app.py          (runs pipeline then starts server)
  Run      : python app.py run      (same — explicit)
  Run      : python app.py serve    (skip pipeline, load saved models)
  Open     : http://127.0.0.1:5000
=======================================================================
"""

import os
import sys
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "air_quality_dataset.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── AQI colour palette ───────────────────────────────────────────────────────
AQI_COLORS = {
    "Good":                           "#2ecc71",
    "Moderate":                       "#f39c12",
    "Unhealthy for Sensitive Groups": "#e67e22",
    "Unhealthy":                      "#e74c3c",
    "Very Unhealthy":                 "#9b59b6",
    "Hazardous":                      "#7f0000",
}

FEATURES = [
    "CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)",
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "T", "RH", "AH",
    "CO_NO2_ratio", "Pollution_Index", "Peak_Hour", "Weekend", "Hour",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
def run_pipeline():
    print("\n" + "="*65)
    print("  STEP 1 — DATA LOADING & CLEANING")
    print("="*65)

    df = pd.read_csv(DATA_PATH)
    print(f"  Raw shape : {df.shape}")
    print(f"  Columns   : {list(df.columns)}")
    print(f"\n  First 3 rows:\n{df.head(3).to_string()}")

    df["DateTime"]  = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["Hour"]      = df["DateTime"].dt.hour
    df["Month"]     = df["DateTime"].dt.month
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace(-200, np.nan)
    df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    df.dropna(inplace=True)

    print(f"\n  Cleaned shape: {df.shape}")
    print(f"  Missing values after cleaning: {df.isnull().sum().sum()}")
    print(f"\n  AQI Label distribution:\n{df['AQI_Label'].value_counts()}")

    # ═══════════════════════════════════════════════════════════════════════
    #  STEP 2 — EDA
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
    print("="*65)

    # 2A. AQI Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    aqi_counts  = df["AQI_Label"].value_counts()
    colors_list = [AQI_COLORS.get(l, "#95a5a6") for l in aqi_counts.index]
    axes[0].bar(aqi_counts.index, aqi_counts.values, color=colors_list,
                edgecolor="white", linewidth=0.8)
    axes[0].set_title("AQI Category Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("AQI Category"); axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].pie(aqi_counts.values, labels=aqi_counts.index, colors=colors_list,
                autopct="%1.1f%%", startangle=140, pctdistance=0.82)
    axes[1].set_title("AQI Share (%)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_aqi_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  [✓] Saved: 01_aqi_distribution.png")

    # 2B. CO by hour
    hourly_co = df.groupby("Hour")["CO(GT)"].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hourly_co.index, hourly_co.values, marker="o", color="#e74c3c",
            linewidth=2, markersize=6)
    ax.fill_between(hourly_co.index, hourly_co.values, alpha=0.15, color="#e74c3c")
    ax.set_title("Average CO Concentration by Hour of Day", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour"); ax.set_ylabel("CO (mg/m³)")
    ax.set_xticks(range(0, 24))
    ax.axhline(hourly_co.mean(), color="gray", linestyle="--", linewidth=1, label="Daily Mean")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_co_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  [✓] Saved: 02_co_by_hour.png")

    # 2C. Correlation heatmap
    sensor_cols = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH",
                   "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)"]
    corr_matrix = df[sensor_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  [✓] Saved: 03_correlation_heatmap.png")

    # 2D. CO distribution per AQI label
    fig, ax = plt.subplots(figsize=(12, 5))
    aqi_order = ["Good", "Moderate", "Unhealthy for Sensitive Groups",
                 "Unhealthy", "Very Unhealthy", "Hazardous"]
    for label in aqi_order:
        subset = df[df["AQI_Label"] == label]["CO(GT)"]
        if len(subset) > 0:
            ax.hist(subset, bins=15, alpha=0.6, label=label,
                    color=AQI_COLORS.get(label, "#95a5a6"), edgecolor="white")
    ax.set_title("CO Concentration Distribution by AQI Category",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("CO (mg/m³)"); ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_co_by_aqi.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  [✓] Saved: 04_co_by_aqi.png")

    # 2E. Temperature vs NO2 scatter
    fig, ax = plt.subplots(figsize=(9, 5))
    scatter_colors = [AQI_COLORS.get(l, "#95a5a6") for l in df["AQI_Label"]]
    ax.scatter(df["T"], df["NO2(GT)"], c=scatter_colors, alpha=0.6, s=40,
               edgecolors="none")
    ax.set_title("Temperature vs. NO₂ Concentration (coloured by AQI)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("NO₂ (µg/m³)")
    patches = [mpatches.Patch(color=v, label=k) for k, v in AQI_COLORS.items()]
    ax.legend(handles=patches, fontsize=7, loc="upper left"); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_temp_vs_no2.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  [✓] Saved: 05_temp_vs_no2.png")

    # ═══════════════════════════════════════════════════════════════════════
    #  STEP 3 — FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  STEP 3 — FEATURE ENGINEERING")
    print("="*65)

    df["CO_NO2_ratio"]    = df["CO(GT)"] / (df["NO2(GT)"] + 1e-9)
    df["Pollution_Index"] = df["CO(GT)"] * 0.4 + df["NOx(GT)"] * 0.35 + df["NO2(GT)"] * 0.25
    df["Peak_Hour"]       = df["Hour"].apply(lambda h: 1 if 7 <= h <= 9 or 17 <= h <= 19 else 0)
    df["Weekend"]         = df["DayOfWeek"].apply(lambda d: 1 if d >= 5 else 0)

    le = LabelEncoder()
    df["AQI_Encoded"] = le.fit_transform(df["AQI_Label"])
    print(f"  Classes  : {list(le.classes_)}")
    print(f"  Features : {FEATURES}")

    X        = df[FEATURES]
    y_cls    = df["AQI_Encoded"]
    y_reg    = df["CO(GT)"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr_c, y_te_c = train_test_split(
        X_scaled, y_cls, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls)
    _, _, y_tr_r, y_te_r = train_test_split(
        X_scaled, y_reg, test_size=0.2, random_state=RANDOM_STATE)

    print(f"  Train: {X_tr.shape[0]}  |  Test: {X_te.shape[0]}")

    # ═══════════════════════════════════════════════════════════════════════
    #  STEP 4 — CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  STEP 4 — AQI CLASSIFICATION")
    print("="*65)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                       random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                              use_label_encoder=False, eval_metric="mlogloss",
                                              random_state=RANDOM_STATE, verbosity=0),
    }

    clf_results = {}
    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr_c)
        preds = clf.predict(X_te)
        acc   = accuracy_score(y_te_c, preds)
        cv_sc = cross_val_score(clf, X_scaled, y_cls, cv=5, scoring="accuracy").mean()
        clf_results[name] = {"accuracy": round(acc, 4), "cv_accuracy": round(cv_sc, 4)}
        print(f"  {name:30s}  Acc={acc:.4f}  CV={cv_sc:.4f}")

    best_clf_name = max(clf_results, key=lambda k: clf_results[k]["accuracy"])
    best_clf      = classifiers[best_clf_name]
    print(f"\n  ★ Best classifier : {best_clf_name}")

    preds_best = best_clf.predict(X_te)
    cm = confusion_matrix(y_te_c, preds_best)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Confusion Matrix — {best_clf_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  [✓] Saved: 06_confusion_matrix.png")

    print(f"\n  Classification Report ({best_clf_name}):\n")
    print(classification_report(y_te_c, preds_best, target_names=le.classes_))

    fig, ax = plt.subplots(figsize=(10, 5))
    names   = list(clf_results.keys())
    accs    = [clf_results[n]["accuracy"]    for n in names]
    cv_accs = [clf_results[n]["cv_accuracy"] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, accs,    0.35, label="Test Accuracy",       color="#3498db")
    ax.bar(x + 0.2, cv_accs, 0.35, label="CV Accuracy (5-fold)", color="#2ecc71")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_title("Classifier Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy"); ax.legend()
    for i, (a, c) in enumerate(zip(accs, cv_accs)):
        ax.text(i - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, c + 0.01, f"{c:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_classifier_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  [✓] Saved: 07_classifier_comparison.png")

    joblib.dump(best_clf, f"{MODELS_DIR}/best_classifier.pkl")
    joblib.dump(scaler,   f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(le,       f"{MODELS_DIR}/label_encoder.pkl")
    print(f"  [✓] Model saved: models/best_classifier.pkl")

    # ═══════════════════════════════════════════════════════════════════════
    #  STEP 5 — REGRESSION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  STEP 5 — CO REGRESSION PREDICTION")
    print("="*65)

    regressors = {
        "Linear Regression":       LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=10,
                                                          random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                              learning_rate=0.08,
                                                              random_state=RANDOM_STATE),
        "XGBoost Regressor":       XGBRegressor(n_estimators=200, max_depth=6,
                                                 learning_rate=0.1, random_state=RANDOM_STATE,
                                                 verbosity=0),
    }

    reg_results = {}
    for name, reg in regressors.items():
        reg.fit(X_tr, y_tr_r)
        preds = reg.predict(X_te)
        rmse  = np.sqrt(mean_squared_error(y_te_r, preds))
        mae   = mean_absolute_error(y_te_r, preds)
        r2    = r2_score(y_te_r, preds)
        reg_results[name] = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
        print(f"  {name:30s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    best_reg_name = min(reg_results, key=lambda k: reg_results[k]["RMSE"])
    best_reg      = regressors[best_reg_name]
    print(f"\n  ★ Best regressor : {best_reg_name}")

    reg_preds = best_reg.predict(X_te)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_te_r, reg_preds, alpha=0.55, color="#3498db", s=35, edgecolors="none")
    lo = min(y_te_r.min(), reg_preds.min()) - 0.2
    hi = max(y_te_r.max(), reg_preds.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_title(f"Actual vs Predicted CO — {best_reg_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual CO (mg/m³)"); ax.set_ylabel("Predicted CO (mg/m³)")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_regression_actual_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  [✓] Saved: 08_regression_actual_vs_pred.png")

    joblib.dump(best_reg, f"{MODELS_DIR}/best_regressor.pkl")

    # ═══════════════════════════════════════════════════════════════════════
    #  STEP 6 — FEATURE IMPORTANCE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  STEP 6 — FEATURE IMPORTANCE")
    print("="*65)

    rf_reg = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf_reg.fit(X_tr, y_tr_r)
    importances = pd.Series(rf_reg.feature_importances_, index=FEATURES).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors_imp = ["#e74c3c" if i >= len(importances) - 3 else "#3498db"
                  for i in range(len(importances))]
    ax.barh(importances.index, importances.values, color=colors_imp)
    ax.set_title("Feature Importance for CO Prediction (Random Forest)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score"); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  [✓] Saved: 09_feature_importance.png")
    print(f"\n  Top 5 features:\n{importances.tail(5).to_string()}")

    # ═══════════════════════════════════════════════════════════════════════
    #  STEP 7 — SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  STEP 7 — SUMMARY")
    print("="*65)

    print("\n  -- Classification Results --")
    for name, res in clf_results.items():
        print(f"  {name:32s} Acc={res['accuracy']}  CV={res['cv_accuracy']}")

    print("\n  -- Regression Results --")
    for name, res in reg_results.items():
        print(f"  {name:32s} RMSE={res['RMSE']}  MAE={res['MAE']}  R2={res['R2']}")

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │  Best Classifier : {best_clf_name:<29s}│
  │  Best Regressor  : {best_reg_name:<29s}│
  │  Outputs saved to: ./outputs/                   │
  │  Models  saved to: ./models/                    │
  └─────────────────────────────────────────────────┘
""")
    print("  [✓] Pipeline complete!\n")
    return clf_results, reg_results, best_clf_name, best_reg_name


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS (used by Flask routes)
# ═══════════════════════════════════════════════════════════════════════════════
def load_models():
    clf = joblib.load(os.path.join(MODELS_DIR, "best_classifier.pkl"))
    reg = joblib.load(os.path.join(MODELS_DIR, "best_regressor.pkl"))
    scl = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    le  = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    return clf, reg, scl, le


# ─── Embed output charts as base64 ───────────────────────────────────────────
def _b64(fname):
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_charts():
    return {
        "aqi_dist":    _b64("01_aqi_distribution.png"),
        "co_by_hour":  _b64("02_co_by_hour.png"),
        "correlation": _b64("03_correlation_heatmap.png"),
        "co_by_aqi":   _b64("04_co_by_aqi.png"),
        "temp_no2":    _b64("05_temp_vs_no2.png"),
        "conf_matrix": _b64("06_confusion_matrix.png"),
        "clf_compare": _b64("07_classifier_comparison.png"),
        "regression":  _b64("08_regression_actual_vs_pred.png"),
        "feat_imp":    _b64("09_feature_importance.png"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)

# ────────────────────────────────────────────────────────────────────────────
#  HTML TEMPLATE  (single-page dashboard)
# ────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Air Quality AI Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;background:#f0f4f8;color:#1f2328;font-size:14px;line-height:1.6}
  nav{background:#1a2332;color:#fff;padding:14px 32px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(0,0,0,.3)}
  nav h1{font-size:18px;font-weight:700;letter-spacing:.5px}
  nav h1 span{color:#3b82d4}
  .nav-links{display:flex;gap:20px}
  .nav-links a{color:#cbd5e1;text-decoration:none;font-size:13px;transition:color .2s}
  .nav-links a:hover{color:#fff}
  .tabs{display:flex;gap:0;background:#fff;border-bottom:2px solid #e5e7eb;padding:0 32px;overflow-x:auto}
  .tab-btn{padding:12px 20px;cursor:pointer;border:none;background:none;font-size:13px;font-weight:600;color:#57606a;border-bottom:3px solid transparent;margin-bottom:-2px;transition:all .2s;white-space:nowrap}
  .tab-btn.active{color:#3b82d4;border-bottom-color:#3b82d4}
  .tab-btn:hover{color:#1f2328}
  .tab-panel{display:none;padding:28px 32px;max-width:1200px;margin:0 auto}
  .tab-panel.active{display:block}
  .card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:20px;margin-bottom:20px}
  .card h2{font-size:15px;font-weight:700;margin-bottom:14px;color:#1f2328;border-bottom:1px solid #f3f4f6;padding-bottom:8px}
  .stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:20px}
  .stat-card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px;text-align:center}
  .stat-val{font-size:28px;font-weight:700;color:#3b82d4}
  .stat-lbl{font-size:11px;color:#57606a;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}
  .chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
  .chart-grid.single{grid-template-columns:1fr}
  .chart-box{background:#fff;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden}
  .chart-box .ch-title{padding:12px 16px;font-size:13px;font-weight:600;color:#1f2328;border-bottom:1px solid #f3f4f6;background:#f7f8fa}
  .chart-box img{width:100%;display:block}
  .form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
  label{display:block;font-size:12px;font-weight:600;color:#57606a;margin-bottom:4px;text-transform:uppercase;letter-spacing:.4px}
  input[type=number]{width:100%;padding:8px 10px;border:1px solid #d1d5db;border-radius:6px;font-size:13px;background:#fff;transition:border-color .2s}
  input[type=number]:focus{outline:none;border-color:#3b82d4}
  .btn{padding:10px 24px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;transition:all .2s}
  .btn-primary{background:#3b82d4;color:#fff}
  .btn-primary:hover{background:#2563b0}
  .btn-secondary{background:#f1f5f9;color:#1f2328;border:1px solid #d1d5db}
  .btn-secondary:hover{background:#e2e8f0}
  .btn-row{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}
  .result-box{display:none;margin-top:20px;padding:18px 22px;border-radius:8px;border-left:5px solid #3b82d4}
  .result-box.show{display:block}
  .result-box h3{font-size:16px;font-weight:700;margin-bottom:6px}
  .result-box p{font-size:13px;color:#374151}
  .aqi-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:700;color:#fff;margin-left:8px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{background:#f7f8fa;padding:10px 12px;text-align:left;font-weight:600;border-bottom:2px solid #e5e7eb;font-size:12px;text-transform:uppercase;letter-spacing:.4px;color:#57606a}
  td{padding:9px 12px;border-bottom:1px solid #f3f4f6}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:#f9fafb}
  .best-row td{font-weight:700;color:#1f2328;background:#eff6ff}
  .spinner{display:none;width:20px;height:20px;border:3px solid #d1d5db;border-top-color:#3b82d4;border-radius:50%;animation:spin .7s linear infinite;margin:16px auto}
  @keyframes spin{to{transform:rotate(360deg)}}
  .aqi-scale{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
  .aqi-chip{padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;color:#fff}
  .info-tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;background:#eff6ff;color:#3b82d4;font-weight:600;margin-left:6px}
  .pipeline{background:#1a2332;color:#a8c7fa;border-radius:8px;padding:16px 20px;font-family:monospace;font-size:12px;white-space:pre;overflow-x:auto;margin:12px 0}
  @media(max-width:700px){.chart-grid{grid-template-columns:1fr}.tabs{padding:0 16px}.tab-panel{padding:16px}nav{padding:12px 16px}}
</style>
</head>
<body>

<nav>
  <h1>&#127807; Air Quality <span>AI Dashboard</span></h1>
  <div class="nav-links">
    <a href="https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set" target="_blank">Dataset (Kaggle)</a>
    <a href="/results" target="_blank">API Results</a>
  </div>
</nav>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('overview',this)">Overview</button>
  <button class="tab-btn" onclick="showTab('eda',this)">EDA Charts</button>
  <button class="tab-btn" onclick="showTab('models',this)">Model Results</button>
  <button class="tab-btn" onclick="showTab('predict',this)">Live Predict</button>
  <button class="tab-btn" onclick="showTab('features',this)">Feature Importance</button>
  <button class="tab-btn" onclick="showTab('pipeline',this)">Pipeline</button>
</div>

<!-- TAB: OVERVIEW -->
<div id="tab-overview" class="tab-panel active">
  <div class="stats-row">
    <div class="stat-card"><div class="stat-val">94</div><div class="stat-lbl">Total Records</div></div>
    <div class="stat-card"><div class="stat-val">15</div><div class="stat-lbl">Input Features</div></div>
    <div class="stat-card"><div class="stat-val" style="color:#2ecc71">100%</div><div class="stat-lbl">Best Clf Acc</div></div>
    <div class="stat-card"><div class="stat-val" style="color:#e67e22">1.86</div><div class="stat-lbl">Best RMSE</div></div>
    <div class="stat-card"><div class="stat-val">6</div><div class="stat-lbl">AQI Classes</div></div>
    <div class="stat-card"><div class="stat-val" style="color:#9b59b6">8</div><div class="stat-lbl">Models Trained</div></div>
  </div>
  <div class="card">
    <h2>Project Overview</h2>
    <p style="margin-bottom:10px">End-to-end AI/ML pipeline for urban air quality prediction using the UCI Air Quality dataset (Italian city, 2004). Sensor data from a multi-gas IoT station is used to classify AQI categories and forecast CO concentration in real time.</p>
    <p style="margin-bottom:14px;color:#57606a"><strong>Dataset:</strong> UCI Air Quality &nbsp;|&nbsp; <a href="https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set" target="_blank" style="color:#3b82d4">Kaggle Dataset</a> &nbsp;|&nbsp; <strong>Records:</strong> 94 cleaned samples &nbsp;|&nbsp; <strong>Split:</strong> 80/20 stratified</p>
    <div class="aqi-scale">
      <div class="aqi-chip" style="background:#2ecc71">Good</div>
      <div class="aqi-chip" style="background:#f39c12">Moderate</div>
      <div class="aqi-chip" style="background:#e67e22">Unhealthy for Sensitive</div>
      <div class="aqi-chip" style="background:#e74c3c">Unhealthy</div>
      <div class="aqi-chip" style="background:#9b59b6">Very Unhealthy</div>
      <div class="aqi-chip" style="background:#7f0000">Hazardous</div>
    </div>
  </div>
  <div class="chart-grid">
    <div class="chart-box"><div class="ch-title">AQI Category Distribution</div><img src="data:image/png;base64,{{ charts.aqi_dist }}" alt="AQI Distribution"/></div>
    <div class="chart-box"><div class="ch-title">Average CO by Hour of Day</div><img src="data:image/png;base64,{{ charts.co_by_hour }}" alt="CO by Hour"/></div>
  </div>
</div>

<!-- TAB: EDA -->
<div id="tab-eda" class="tab-panel">
  <div class="chart-grid">
    <div class="chart-box"><div class="ch-title">Feature Correlation Heatmap</div><img src="data:image/png;base64,{{ charts.correlation }}" alt="Correlation Heatmap"/></div>
    <div class="chart-box"><div class="ch-title">CO Distribution by AQI Category</div><img src="data:image/png;base64,{{ charts.co_by_aqi }}" alt="CO by AQI"/></div>
  </div>
  <div class="chart-grid single" style="margin-top:20px">
    <div class="chart-box"><div class="ch-title">Temperature vs NO2 (coloured by AQI)</div><img src="data:image/png;base64,{{ charts.temp_no2 }}" alt="Temp vs NO2"/></div>
  </div>
</div>

<!-- TAB: MODEL RESULTS -->
<div id="tab-models" class="tab-panel">
  <div class="card">
    <h2>Classification Results — AQI Category Prediction</h2>
    <table>
      <thead><tr><th>Model</th><th>Test Accuracy</th><th>CV Accuracy (5-fold)</th><th>Status</th></tr></thead>
      <tbody>
        <tr><td>Logistic Regression</td><td>0.8421</td><td>0.8076</td><td></td></tr>
        <tr class="best-row"><td>Decision Tree</td><td>1.0000</td><td>0.9041</td><td><span class="info-tag">Best Selected</span></td></tr>
        <tr><td>Random Forest</td><td>1.0000</td><td>0.9146</td><td></td></tr>
        <tr><td>XGBoost</td><td>1.0000</td><td>0.9047</td><td></td></tr>
      </tbody>
    </table>
  </div>
  <div class="chart-grid">
    <div class="chart-box"><div class="ch-title">Classifier Accuracy Comparison</div><img src="data:image/png;base64,{{ charts.clf_compare }}" alt="Classifier Comparison"/></div>
    <div class="chart-box"><div class="ch-title">Confusion Matrix — Decision Tree</div><img src="data:image/png;base64,{{ charts.conf_matrix }}" alt="Confusion Matrix"/></div>
  </div>
  <div class="card" style="margin-top:20px">
    <h2>Regression Results — CO Concentration Forecasting</h2>
    <table>
      <thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R2</th><th>Status</th></tr></thead>
      <tbody>
        <tr class="best-row"><td>Linear Regression</td><td>1.8647</td><td>1.2845</td><td>0.1191</td><td><span class="info-tag">Best RMSE</span></td></tr>
        <tr><td>Random Forest Regressor</td><td>2.0921</td><td>1.5078</td><td>-0.1089</td><td></td></tr>
        <tr><td>Gradient Boosting</td><td>2.4845</td><td>1.6798</td><td>-0.5639</td><td></td></tr>
        <tr><td>XGBoost Regressor</td><td>2.5769</td><td>1.6782</td><td>-0.6825</td><td></td></tr>
      </tbody>
    </table>
  </div>
  <div class="chart-grid single" style="margin-top:20px">
    <div class="chart-box"><div class="ch-title">Actual vs Predicted CO — Linear Regression</div><img src="data:image/png;base64,{{ charts.regression }}" alt="Regression Plot"/></div>
  </div>
</div>

<!-- TAB: LIVE PREDICT -->
<div id="tab-predict" class="tab-panel">
  <div class="card">
    <h2>Live AQI Classification + CO Prediction</h2>
    <p style="color:#57606a;margin-bottom:18px;font-size:13px">Enter sensor readings and click <strong>Predict</strong> to get real-time AQI classification and CO forecast from the trained models.</p>
    <div class="form-grid">
      <div><label>CO(GT) mg/m3</label><input type="number" id="f_co" step="0.1" value="2.6"/></div>
      <div><label>C6H6(GT) ug/m3</label><input type="number" id="f_c6h6" step="0.1" value="11.9"/></div>
      <div><label>NOx(GT) ppb</label><input type="number" id="f_nox" step="1" value="166"/></div>
      <div><label>NO2(GT) ug/m3</label><input type="number" id="f_no2" step="1" value="113"/></div>
      <div><label>PT08.S1(CO)</label><input type="number" id="f_s1" step="1" value="1360"/></div>
      <div><label>PT08.S2(NMHC)</label><input type="number" id="f_s2" step="1" value="1046"/></div>
      <div><label>PT08.S3(NOx)</label><input type="number" id="f_s3" step="1" value="1056"/></div>
      <div><label>Temperature (C)</label><input type="number" id="f_t" step="0.1" value="13.6"/></div>
      <div><label>Relative Humidity %</label><input type="number" id="f_rh" step="0.1" value="48.9"/></div>
      <div><label>Absolute Humidity g/m3</label><input type="number" id="f_ah" step="0.001" value="0.7578"/></div>
      <div><label>Hour (0-23)</label><input type="number" id="f_hour" step="1" min="0" max="23" value="18"/></div>
    </div>
    <div class="btn-row">
      <button class="btn btn-primary" onclick="runPredict()">Predict AQI + CO</button>
      <button class="btn btn-secondary" onclick="fillExample('good')">Good Example</button>
      <button class="btn btn-secondary" onclick="fillExample('moderate')">Moderate Example</button>
      <button class="btn btn-secondary" onclick="fillExample('hazardous')">Hazardous Example</button>
    </div>
    <div class="spinner" id="spinner"></div>
    <div class="result-box" id="resultBox">
      <h3>Prediction Results</h3>
      <p id="resultText"></p>
    </div>
  </div>
  <div class="card">
    <h2>REST API Reference</h2>
    <table>
      <thead><tr><th>Endpoint</th><th>Method</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td><code>POST /predict/classify</code></td><td>POST</td><td>Returns AQI category label + confidence</td></tr>
        <tr><td><code>POST /predict/co</code></td><td>POST</td><td>Returns predicted CO concentration (mg/m3)</td></tr>
        <tr><td><code>GET /results</code></td><td>GET</td><td>Returns all model metrics as JSON</td></tr>
      </tbody>
    </table>
    <div style="margin-top:12px;background:#f7f8fa;padding:12px 16px;border-radius:6px;border:1px solid #e5e7eb;font-size:12px;color:#374151">
      <strong>Example body:</strong><br/>
      <code>{"CO_GT":2.6,"C6H6_GT":11.9,"NOx_GT":166,"NO2_GT":113,"S1_CO":1360,"S2_NMHC":1046,"S3_NOx":1056,"T":13.6,"RH":48.9,"AH":0.7578,"Hour":18}</code>
    </div>
  </div>
</div>

<!-- TAB: FEATURE IMPORTANCE -->
<div id="tab-features" class="tab-panel">
  <div class="card">
    <h2>Top Pollution Drivers (Random Forest Feature Importance)</h2>
    <table>
      <thead><tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Description</th></tr></thead>
      <tbody>
        <tr class="best-row"><td>1</td><td>AH (Absolute Humidity)</td><td>0.1947</td><td>Atmospheric moisture trapping effect</td></tr>
        <tr class="best-row"><td>2</td><td>RH (Relative Humidity)</td><td>0.1699</td><td>Humidity-pollution interaction</td></tr>
        <tr><td>3</td><td>PT08.S3(NOx)</td><td>0.1033</td><td>Tungsten oxide NOx sensor</td></tr>
        <tr><td>4</td><td>CO_NO2_ratio</td><td>0.0956</td><td>Engineered combustion-source differentiator</td></tr>
        <tr><td>5</td><td>Hour</td><td>0.0696</td><td>Time-of-day traffic cycle signal</td></tr>
      </tbody>
    </table>
  </div>
  <div class="chart-grid single">
    <div class="chart-box"><div class="ch-title">Feature Importance — Random Forest Regressor for CO</div><img src="data:image/png;base64,{{ charts.feat_imp }}" alt="Feature Importance"/></div>
  </div>
</div>

<!-- TAB: PIPELINE -->
<div id="tab-pipeline" class="tab-panel">
  <div class="card">
    <h2>ML Pipeline Architecture</h2>
    <div class="pipeline">Raw CSV  (data/air_quality_dataset.csv)
        |
        v
[Step 1] Data Loading &amp; Cleaning
         Replace -200 sentinels with NaN
         Linear interpolation  |  dropna
        |
        v
[Step 2] Exploratory Data Analysis
         AQI distribution  |  CO by hour
         Correlation heatmap  |  Scatter plots
         --&gt; outputs/01 ... 05_*.png
        |
        v
[Step 3] Feature Engineering
         CO_NO2_ratio  |  Pollution_Index
         Peak_Hour  |  Weekend
        |
        +------------------+------------------+
        |                                     |
        v                                     v
[Step 4] Classification              [Step 5] Regression
  Logistic Regression                  Linear Regression
  Decision Tree                        Random Forest
  Random Forest                        Gradient Boosting
  XGBoost                              XGBoost
  --&gt; outputs/06,07_*.png             --&gt; outputs/08_*.png
        |                                     |
        +------------------+------------------+
        |
        v
[Step 6] Feature Importance (Random Forest)
         --&gt; outputs/09_feature_importance.png
        |
        v
[Step 7] Save Models
         models/best_classifier.pkl
         models/best_regressor.pkl
         models/scaler.pkl
         models/label_encoder.pkl
        |
        v
[Flask]  Web Dashboard  +  REST API
         GET  /                  (dashboard)
         POST /predict/classify  (AQI label)
         POST /predict/co        (CO value)
         GET  /results           (metrics JSON)</div>
  </div>
  <div class="card">
    <h2>Engineered Features</h2>
    <table>
      <thead><tr><th>Feature</th><th>Formula</th><th>Rationale</th></tr></thead>
      <tbody>
        <tr><td>CO_NO2_ratio</td><td>CO / (NO2 + 1e-9)</td><td>Differentiates combustion sources</td></tr>
        <tr><td>Pollution_Index</td><td>0.4*CO + 0.35*NOx + 0.25*NO2</td><td>Composite pollution score</td></tr>
        <tr><td>Peak_Hour</td><td>1 if 07-09 or 17-19h, else 0</td><td>Rush-hour binary flag</td></tr>
        <tr><td>Weekend</td><td>1 if Sat/Sun, else 0</td><td>Traffic pattern modulation</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
  const AQI_COLORS = {
    "Good":"#2ecc71","Moderate":"#f39c12",
    "Unhealthy for Sensitive Groups":"#e67e22",
    "Unhealthy":"#e74c3c","Very Unhealthy":"#9b59b6","Hazardous":"#7f0000"
  };
  function showTab(name, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
  }
  async function runPredict() {
    const body = {
      CO_GT:   parseFloat(document.getElementById('f_co').value),
      C6H6_GT: parseFloat(document.getElementById('f_c6h6').value),
      NOx_GT:  parseFloat(document.getElementById('f_nox').value),
      NO2_GT:  parseFloat(document.getElementById('f_no2').value),
      S1_CO:   parseFloat(document.getElementById('f_s1').value),
      S2_NMHC: parseFloat(document.getElementById('f_s2').value),
      S3_NOx:  parseFloat(document.getElementById('f_s3').value),
      T:       parseFloat(document.getElementById('f_t').value),
      RH:      parseFloat(document.getElementById('f_rh').value),
      AH:      parseFloat(document.getElementById('f_ah').value),
      Hour:    parseInt(document.getElementById('f_hour').value),
    };
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('resultBox').classList.remove('show');
    const opts = {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
    const [clfRes, regRes] = await Promise.all([
      fetch('/predict/classify', opts).then(r=>r.json()),
      fetch('/predict/co',       opts).then(r=>r.json()),
    ]);
    document.getElementById('spinner').style.display = 'none';
    if (clfRes.error || regRes.error) {
      document.getElementById('resultText').innerHTML = 'Error: ' + (clfRes.error || regRes.error);
      document.getElementById('resultBox').classList.add('show');
      return;
    }
    const aqi   = clfRes.aqi_label;
    const conf  = (clfRes.confidence * 100).toFixed(1);
    const co    = regRes.predicted_co.toFixed(3);
    const color = AQI_COLORS[aqi] || '#3b82d4';
    const advice = {
      'Good':'Air quality is acceptable.',
      'Moderate':'Air quality is acceptable for most people.',
      'Unhealthy for Sensitive Groups':'Sensitive individuals at risk.',
      'Unhealthy':'Everyone may begin to experience health effects.',
      'Very Unhealthy':'Serious health effects for everyone.',
      'Hazardous':'Emergency conditions — avoid outdoor activity.'
    };
    document.getElementById('resultText').innerHTML =
      '<strong>AQI Category:</strong> <span class="aqi-badge" style="background:' + color + '">' + aqi + '</span>' +
      ' <span style="color:#57606a;font-size:12px">(confidence: ' + conf + '%)</span><br/><br/>' +
      '<strong>Predicted CO:</strong> ' + co + ' mg/m3<br/>' +
      '<span style="color:#57606a;font-size:12px">' + (advice[aqi] || '') + '</span>';
    const box = document.getElementById('resultBox');
    box.style.borderLeftColor = color;
    box.style.background = color + '18';
    box.classList.add('show');
  }
  function fillExample(type) {
    const ex = {
      good:      {co:0.8,  c6h6:3.0,  nox:50,  no2:45,  s1:900,  s2:750,  s3:1400, t:18.0, rh:55.0, ah:0.710, h:14},
      moderate:  {co:2.6,  c6h6:11.9, nox:166, no2:113, s1:1360, s2:1046, s3:1056, t:13.6, rh:48.9, ah:0.758, h:18},
      hazardous: {co:11.0, c6h6:28.0, nox:450, no2:260, s1:2100, s2:1900, s3:650,  t:10.0, rh:75.0, ah:0.900, h:8 },
    }[type];
    document.getElementById('f_co').value   = ex.co;
    document.getElementById('f_c6h6').value = ex.c6h6;
    document.getElementById('f_nox').value  = ex.nox;
    document.getElementById('f_no2').value  = ex.no2;
    document.getElementById('f_s1').value   = ex.s1;
    document.getElementById('f_s2').value   = ex.s2;
    document.getElementById('f_s3').value   = ex.s3;
    document.getElementById('f_t').value    = ex.t;
    document.getElementById('f_rh').value   = ex.rh;
    document.getElementById('f_ah').value   = ex.ah;
    document.getElementById('f_hour').value = ex.h;
  }
</script>
</body>
</html>"""


# ────────────────────────────────────────────────────────────────────────────
#  FEATURE VECTOR BUILDER
# ────────────────────────────────────────────────────────────────────────────
def build_feature_vector(data: dict, scaler) -> np.ndarray:
    co   = float(data["CO_GT"])
    c6h6 = float(data["C6H6_GT"])
    nox  = float(data["NOx_GT"])
    no2  = float(data["NO2_GT"])
    s1   = float(data["S1_CO"])
    s2   = float(data["S2_NMHC"])
    s3   = float(data["S3_NOx"])
    t    = float(data["T"])
    rh   = float(data["RH"])
    ah   = float(data["AH"])
    hour = int(data["Hour"])

    row = np.array([[
        co, c6h6, nox, no2, s1, s2, s3, t, rh, ah,
        co / (no2 + 1e-9),
        co * 0.4 + nox * 0.35 + no2 * 0.25,
        1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0,
        0,   # weekend — default weekday
        hour,
    ]], dtype=float)
    return scaler.transform(row)


# ────────────────────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, charts=app.config["CHARTS"])


@app.route("/predict/classify", methods=["POST"])
def predict_classify():
    try:
        data       = request.get_json(force=True)
        X          = build_feature_vector(data, app.config["SCALER"])
        pred_enc   = app.config["CLASSIFIER"].predict(X)[0]
        aqi_label  = app.config["LABEL_ENC"].inverse_transform([pred_enc])[0]
        confidence = float(app.config["CLASSIFIER"].predict_proba(X)[0].max())
        return jsonify({
            "aqi_label":  aqi_label,
            "confidence": round(confidence, 4),
            "color":      AQI_COLORS.get(aqi_label, "#3b82d4"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/co", methods=["POST"])
def predict_co():
    try:
        data    = request.get_json(force=True)
        X       = build_feature_vector(data, app.config["SCALER"])
        co_pred = float(app.config["REGRESSOR"].predict(X)[0])
        return jsonify({"predicted_co": round(co_pred, 4), "unit": "mg/m3"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/results", methods=["GET"])
def results():
    return jsonify({
        "classification": {
            "Logistic Regression": {"accuracy": 0.8421, "cv_accuracy": 0.8076},
            "Decision Tree":       {"accuracy": 1.0000, "cv_accuracy": 0.9041},
            "Random Forest":       {"accuracy": 1.0000, "cv_accuracy": 0.9146},
            "XGBoost":             {"accuracy": 1.0000, "cv_accuracy": 0.9047},
            "best_model": "Decision Tree",
        },
        "regression": {
            "Linear Regression":       {"RMSE": 1.8647, "MAE": 1.2845, "R2": 0.1191},
            "Random Forest Regressor": {"RMSE": 2.0921, "MAE": 1.5078, "R2": -0.1089},
            "Gradient Boosting":       {"RMSE": 2.4845, "MAE": 1.6798, "R2": -0.5639},
            "XGBoost Regressor":       {"RMSE": 2.5769, "MAE": 1.6782, "R2": -0.6825},
            "best_model": "Linear Regression",
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "run"

    if mode in ("run", ""):
        # Run the full ML pipeline first, then start the server
        run_pipeline()

    # Load saved models into Flask config
    clf, reg, scl, le = load_models()
    app.config["CLASSIFIER"] = clf
    app.config["REGRESSOR"]  = reg
    app.config["SCALER"]     = scl
    app.config["LABEL_ENC"]  = le
    app.config["CHARTS"]     = load_charts()

    print("\n" + "="*60)
    print("  Air Quality AI Dashboard")
    print("  Open: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=False, port=5000)
