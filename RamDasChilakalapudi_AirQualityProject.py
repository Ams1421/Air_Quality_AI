"""
=======================================================================
  AIR QUALITY PREDICTION & POLLUTION ANALYSIS USING MACHINE LEARNING
  Author   : Ram Das Chilakalapudi
  Date     : September 2026
  Dataset  : https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set

  Description:
    Single-file complete application containing:
      - Full ML pipeline (Steps 1-7: data loading, EDA, feature
        engineering, AQI classification, CO regression,
        feature importance, summary)
      - Flask REST API  (/predict/classify, /predict/co, /results)
      - Embedded HTML/CSS/JS interactive web dashboard

  How to run:
    python RamDasChilakalapudi_AirQualityProject.py         <- full pipeline + dashboard
    python RamDasChilakalapudi_AirQualityProject.py serve   <- dashboard only (saved models)

  Then open: http://127.0.0.1:5000
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
  :root{
    --navy:#0f172a;--navy2:#1e293b;--navy3:#334155;
    --blue:#3b82f6;--blue2:#2563eb;--blue-light:#eff6ff;
    --green:#22c55e;--orange:#f59e0b;--red:#ef4444;--purple:#a855f7;
    --bg:#f1f5f9;--surface:#ffffff;--border:#e2e8f0;--border2:#f1f5f9;
    --text:#0f172a;--muted:#64748b;--muted2:#94a3b8;
    --radius:12px;--radius-sm:8px;--shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.05);
    --shadow-md:0 4px 12px rgba(0,0,0,.1);
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html{scroll-behavior:smooth}
  body{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.65}

  /* ── NAVBAR ── */
  nav{background:var(--navy);padding:0 32px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:200;height:58px;box-shadow:0 2px 12px rgba(0,0,0,.3)}
  .nav-brand{display:flex;align-items:center;gap:10px}
  .nav-brand .logo{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0}
  .nav-brand h1{font-size:16px;font-weight:700;color:#fff;letter-spacing:.2px}
  .nav-brand h1 span{color:#93c5fd}
  .nav-right{display:flex;align-items:center;gap:6px}
  .nav-pill{display:inline-flex;align-items:center;gap:5px;padding:6px 12px;border-radius:20px;font-size:12px;font-weight:600;color:#cbd5e1;text-decoration:none;border:1px solid rgba(255,255,255,.1);transition:all .2s}
  .nav-pill:hover{background:rgba(255,255,255,.1);color:#fff;border-color:rgba(255,255,255,.2)}
  .nav-pill.active-pill{background:var(--blue);color:#fff;border-color:var(--blue)}
  .nav-dot{width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block}

  /* ── SIDEBAR + LAYOUT ── */
  .shell{display:flex;min-height:calc(100vh - 58px)}
  .sidebar{width:220px;background:var(--surface);border-right:1px solid var(--border);padding:20px 0;flex-shrink:0;position:sticky;top:58px;height:calc(100vh - 58px);overflow-y:auto}
  .sidebar-section{padding:6px 16px;font-size:10px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.8px;margin-top:12px}
  .sidebar-item{display:flex;align-items:center;gap:10px;padding:9px 20px;font-size:13px;font-weight:500;color:var(--muted);cursor:pointer;border:none;background:none;width:100%;text-align:left;border-left:3px solid transparent;transition:all .15s}
  .sidebar-item:hover{background:var(--bg);color:var(--text)}
  .sidebar-item.active{color:var(--blue);background:var(--blue-light);border-left-color:var(--blue);font-weight:600}
  .sidebar-icon{font-size:15px;width:20px;text-align:center;flex-shrink:0}
  .main{flex:1;min-width:0;padding:28px 32px;max-width:1100px}

  /* ── PANELS ── */
  .panel{display:none}.panel.active{display:block}

  /* ── PAGE HEADER ── */
  .page-hdr{margin-bottom:24px}
  .page-hdr h2{font-size:20px;font-weight:700;color:var(--text);margin-bottom:4px}
  .page-hdr p{font-size:13px;color:var(--muted)}
  .breadcrumb{font-size:11px;color:var(--muted2);margin-bottom:8px;display:flex;align-items:center;gap:5px}
  .breadcrumb span{color:var(--muted)}

  /* ── STAT CARDS ── */
  .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:24px}
  .stat-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px 16px;position:relative;overflow:hidden;transition:box-shadow .2s}
  .stat-card:hover{box-shadow:var(--shadow-md)}
  .stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:var(--radius) var(--radius) 0 0}
  .stat-card.blue::before{background:var(--blue)}
  .stat-card.green::before{background:var(--green)}
  .stat-card.orange::before{background:var(--orange)}
  .stat-card.purple::before{background:var(--purple)}
  .stat-card.red::before{background:var(--red)}
  .stat-num{font-size:26px;font-weight:800;line-height:1;margin-bottom:5px}
  .stat-num.blue{color:var(--blue)}.stat-num.green{color:var(--green)}.stat-num.orange{color:var(--orange)}.stat-num.purple{color:var(--purple)}
  .stat-lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;font-weight:600}
  .stat-sub{font-size:11px;color:var(--muted2);margin-top:3px}

  /* ── CARDS ── */
  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:20px;overflow:hidden;box-shadow:var(--shadow)}
  .card-hdr{padding:14px 20px;border-bottom:1px solid var(--border2);display:flex;align-items:center;justify-content:space-between;background:#fafbfc}
  .card-hdr h3{font-size:14px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:8px}
  .card-hdr .hdr-icon{font-size:15px}
  .card-body{padding:20px}
  .card-body.no-pad{padding:0}

  /* ── CHART GRID ── */
  .chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px}
  .chart-grid.one{grid-template-columns:1fr}
  .chart-grid.three{grid-template-columns:1fr 1fr 1fr}
  .chart-wrap{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow);transition:box-shadow .2s}
  .chart-wrap:hover{box-shadow:var(--shadow-md)}
  .chart-title{padding:10px 16px;font-size:12px;font-weight:600;color:var(--muted);border-bottom:1px solid var(--border2);background:#fafbfc;display:flex;align-items:center;gap:6px}
  .chart-title::before{content:'';width:8px;height:8px;border-radius:2px;background:var(--blue);flex-shrink:0}
  .chart-wrap img{width:100%;display:block}

  /* ── TABLES ── */
  .tbl-wrap{overflow-x:auto;border-radius:var(--radius-sm)}
  table{width:100%;border-collapse:collapse;font-size:13px}
  thead tr{background:#f8fafc}
  th{padding:10px 14px;text-align:left;font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid var(--border);white-space:nowrap}
  td{padding:10px 14px;border-bottom:1px solid var(--border2);color:var(--text);vertical-align:middle}
  tbody tr:last-child td{border-bottom:none}
  tbody tr:hover td{background:#f8fafc}
  .best-row td{font-weight:700;background:var(--blue-light)!important}
  .tag{display:inline-flex;align-items:center;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700}
  .tag-blue{background:var(--blue-light);color:var(--blue2)}
  .tag-green{background:#f0fdf4;color:#16a34a}
  .tag-orange{background:#fffbeb;color:#d97706}
  .bar-cell{display:flex;align-items:center;gap:8px}
  .bar-bg{flex:1;height:6px;background:#e2e8f0;border-radius:3px;min-width:60px}
  .bar-fill{height:6px;border-radius:3px;background:var(--blue)}

  /* ── AQI CHIPS ── */
  .aqi-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
  .aqi-chip{padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;color:#fff;letter-spacing:.2px}

  /* ── FORM ── */
  .section-label{font-size:11px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid var(--border2)}
  .form-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:14px;margin-bottom:18px}
  .field{display:flex;flex-direction:column;gap:5px}
  .field label{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
  .field-hint{font-size:10px;color:var(--muted2)}
  .field input[type=number]{padding:9px 12px;border:1.5px solid var(--border);border-radius:var(--radius-sm);font-size:13px;color:var(--text);background:var(--surface);transition:all .2s;outline:none;width:100%}
  .field input[type=number]:hover{border-color:#94a3b8}
  .field input[type=number]:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(59,130,246,.12)}
  .btn-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  .btn{display:inline-flex;align-items:center;gap:6px;padding:9px 20px;border:none;border-radius:var(--radius-sm);cursor:pointer;font-size:13px;font-weight:600;transition:all .2s;white-space:nowrap}
  .btn-primary{background:var(--blue);color:#fff}
  .btn-primary:hover{background:var(--blue2);transform:translateY(-1px);box-shadow:0 4px 12px rgba(59,130,246,.3)}
  .btn-primary:active{transform:translateY(0)}
  .btn-ghost{background:transparent;color:var(--muted);border:1.5px solid var(--border)}
  .btn-ghost:hover{background:var(--bg);color:var(--text);border-color:#94a3b8}
  .btn-outline-green{background:#f0fdf4;color:#16a34a;border:1.5px solid #bbf7d0}
  .btn-outline-green:hover{background:#dcfce7}
  .btn-outline-orange{background:#fffbeb;color:#d97706;border:1.5px solid #fde68a}
  .btn-outline-orange:hover{background:#fef3c7}
  .btn-outline-red{background:#fef2f2;color:#dc2626;border:1.5px solid #fecaca}
  .btn-outline-red:hover{background:#fee2e2}

  /* ── PREDICTION RESULT ── */
  .pred-result{display:none;margin-top:22px;border-radius:var(--radius);overflow:hidden;border:1px solid var(--border);box-shadow:var(--shadow)}
  .pred-result.show{display:block}
  .pred-banner{padding:16px 20px;display:flex;align-items:center;gap:12px}
  .pred-banner .aqi-big{font-size:18px;font-weight:800;color:#fff}
  .pred-banner .aqi-sub{font-size:12px;color:rgba(255,255,255,.8);margin-top:2px}
  .pred-body{padding:16px 20px;background:var(--surface);display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .pred-metric{display:flex;flex-direction:column;gap:3px}
  .pred-metric .label{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
  .pred-metric .value{font-size:22px;font-weight:800;color:var(--text)}
  .pred-metric .unit{font-size:11px;color:var(--muted2)}
  .pred-advice{grid-column:1/-1;padding:10px 14px;background:var(--bg);border-radius:var(--radius-sm);font-size:13px;color:var(--muted);border-left:3px solid var(--blue)}
  .conf-bar{margin-top:6px;height:5px;border-radius:3px;background:#e2e8f0;overflow:hidden}
  .conf-fill{height:100%;border-radius:3px;background:var(--green);transition:width .6s ease}

  /* ── SPINNER ── */
  .spinner{display:none;width:24px;height:24px;border:3px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:spin .7s linear infinite;margin:20px auto}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* ── CODE BLOCK ── */
  .code-block{background:var(--navy);border-radius:var(--radius-sm);padding:14px 16px;font-family:'Cascadia Code','Fira Code',monospace;font-size:12px;color:#e2e8f0;overflow-x:auto;line-height:1.7;border:1px solid var(--navy3)}
  .code-block .kw{color:#93c5fd}.code-block .str{color:#86efac}.code-block .cm{color:#64748b}

  /* ── PIPELINE ── */
  .pipeline-box{background:var(--navy);border-radius:var(--radius);padding:20px 24px;font-family:'Cascadia Code','Fira Code',Consolas,monospace;font-size:12px;color:#94a3b8;white-space:pre;overflow-x:auto;line-height:1.9;border:1px solid var(--navy3)}
  .pipeline-box .step{color:#93c5fd;font-weight:700}
  .pipeline-box .arrow{color:#64748b}
  .pipeline-box .out{color:#86efac}

  /* ── PROGRESS BARS ── */
  .progress-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
  .progress-label{font-size:12px;color:var(--muted);width:160px;flex-shrink:0;font-weight:500}
  .progress-track{flex:1;height:8px;background:var(--border2);border-radius:4px;overflow:hidden}
  .progress-fill{height:100%;border-radius:4px;transition:width 1s ease}
  .progress-val{font-size:12px;font-weight:700;color:var(--text);width:42px;text-align:right;flex-shrink:0}

  /* ── INFO CARDS ── */
  .info-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
  .info-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px;box-shadow:var(--shadow)}
  .info-card .ic-top{display:flex;align-items:center;gap:8px;margin-bottom:8px}
  .info-card .ic-icon{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0}
  .info-card .ic-title{font-size:13px;font-weight:700;color:var(--text)}
  .info-card p{font-size:12px;color:var(--muted);line-height:1.6}

  /* ── API TABLE ── */
  .method-badge{display:inline-flex;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:.4px}
  .method-post{background:#fef3c7;color:#92400e}
  .method-get{background:#d1fae5;color:#065f46}

  /* ── FOOTER ── */
  .page-footer{margin-top:40px;padding-top:20px;border-top:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;flex-wrap:gap;gap:10px}
  .page-footer p{font-size:12px;color:var(--muted2)}
  .page-footer a{color:var(--blue);text-decoration:none;font-weight:600}

  /* ── DIVIDER ── */
  .divider{height:1px;background:var(--border2);margin:20px 0}

  /* ── RESPONSIVE ── */
  @media(max-width:900px){
    .sidebar{display:none}
    .main{padding:20px 16px}
    .chart-grid{grid-template-columns:1fr}
    .chart-grid.three{grid-template-columns:1fr}
    .info-grid{grid-template-columns:1fr}
    .pred-body{grid-template-columns:1fr}
  }
  @media(max-width:600px){
    nav{padding:0 16px}
    .stats-grid{grid-template-columns:1fr 1fr}
    .form-grid{grid-template-columns:1fr 1fr}
    .btn-row{flex-direction:column;align-items:stretch}
    .btn{justify-content:center}
  }
</style>
</head>
<body>

<!-- ══ NAVBAR ══════════════════════════════════════════════════════════════ -->
<nav>
  <div class="nav-brand">
    <div class="logo">&#127807;</div>
    <h1>Air Quality <span>AI Dashboard</span></h1>
  </div>
  <div class="nav-right">
    <span style="display:flex;align-items:center;gap:5px;font-size:12px;color:#94a3b8;margin-right:8px">
      <span class="nav-dot"></span> Live
    </span>
    <a class="nav-pill" href="https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set" target="_blank">Dataset</a>
    <a class="nav-pill active-pill" href="/results" target="_blank">API</a>
  </div>
</nav>

<!-- ══ SHELL ════════════════════════════════════════════════════════════════ -->
<div class="shell">

<!-- ── SIDEBAR ── -->
<aside class="sidebar">
  <div class="sidebar-section">Explore</div>
  <button class="sidebar-item active" onclick="showPanel('overview',this)">
    <span class="sidebar-icon">&#128202;</span> Overview
  </button>
  <button class="sidebar-item" onclick="showPanel('eda',this)">
    <span class="sidebar-icon">&#128269;</span> EDA Charts
  </button>
  <div class="sidebar-section">Models</div>
  <button class="sidebar-item" onclick="showPanel('models',this)">
    <span class="sidebar-icon">&#129302;</span> Model Results
  </button>
  <button class="sidebar-item" onclick="showPanel('features',this)">
    <span class="sidebar-icon">&#128200;</span> Feature Importance
  </button>
  <div class="sidebar-section">Tools</div>
  <button class="sidebar-item" onclick="showPanel('predict',this)">
    <span class="sidebar-icon">&#9889;</span> Live Predict
  </button>
  <button class="sidebar-item" onclick="showPanel('pipeline',this)">
    <span class="sidebar-icon">&#9881;</span> Pipeline
  </button>
</aside>

<!-- ── MAIN CONTENT ── -->
<main class="main">

<!-- ════ PANEL: OVERVIEW ════════════════════════════════════════════════════ -->
<div id="panel-overview" class="panel active">
  <div class="page-hdr">
    <div class="breadcrumb"><span>Dashboard</span> / Overview</div>
    <h2>Project Overview</h2>
    <p>End-to-end AI/ML pipeline for urban air quality prediction — UCI Air Quality dataset, Italian city 2004</p>
  </div>

  <div class="stats-grid">
    <div class="stat-card blue">
      <div class="stat-num blue">94</div>
      <div class="stat-lbl">Total Records</div>
      <div class="stat-sub">Cleaned samples</div>
    </div>
    <div class="stat-card blue">
      <div class="stat-num blue">15</div>
      <div class="stat-lbl">Input Features</div>
      <div class="stat-sub">Sensor + engineered</div>
    </div>
    <div class="stat-card green">
      <div class="stat-num green">100%</div>
      <div class="stat-lbl">Best Clf Accuracy</div>
      <div class="stat-sub">Decision Tree</div>
    </div>
    <div class="stat-card orange">
      <div class="stat-num orange">1.86</div>
      <div class="stat-lbl">Best Reg RMSE</div>
      <div class="stat-sub">Linear Regression</div>
    </div>
    <div class="stat-card purple">
      <div class="stat-num purple">6</div>
      <div class="stat-lbl">AQI Classes</div>
      <div class="stat-sub">Good to Hazardous</div>
    </div>
    <div class="stat-card blue">
      <div class="stat-num blue">8</div>
      <div class="stat-lbl">Models Trained</div>
      <div class="stat-sub">4 clf + 4 reg</div>
    </div>
  </div>

  <div class="info-grid">
    <div class="info-card">
      <div class="ic-top">
        <div class="ic-icon" style="background:#eff6ff">&#128202;</div>
        <div class="ic-title">About the Dataset</div>
      </div>
      <p>Multi-sensor IoT station data from an Italian city, March–April 2004. 9,358 raw hourly readings; 94 representative samples used.
      <br/><br/>
      <a href="https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set" target="_blank" style="color:#3b82f6;font-weight:600">Kaggle Dataset &#8599;</a></p>
    </div>
    <div class="info-card">
      <div class="ic-top">
        <div class="ic-icon" style="background:#f0fdf4">&#129302;</div>
        <div class="ic-title">Model Performance</div>
      </div>
      <p>Classification: 3 of 4 models achieve 100% test accuracy. Best CV score: Random Forest at 91.5%.<br/><br/>
      Regression: Linear Regression achieves best RMSE of 1.86 mg/m&#179; on CO prediction.</p>
    </div>
  </div>

  <div class="card">
    <div class="card-hdr"><h3><span class="hdr-icon">&#127912;</span> AQI Scale (EPA Standard)</h3></div>
    <div class="card-body">
      <div class="aqi-row">
        <span class="aqi-chip" style="background:#22c55e">Good</span>
        <span class="aqi-chip" style="background:#f59e0b">Moderate</span>
        <span class="aqi-chip" style="background:#f97316">Unhealthy for Sensitive Groups</span>
        <span class="aqi-chip" style="background:#ef4444">Unhealthy</span>
        <span class="aqi-chip" style="background:#a855f7">Very Unhealthy</span>
        <span class="aqi-chip" style="background:#7f0000">Hazardous</span>
      </div>
    </div>
  </div>

  <div class="chart-grid">
    <div class="chart-wrap">
      <div class="chart-title">AQI Category Distribution</div>
      <img src="data:image/png;base64,{{ charts.aqi_dist }}" alt="AQI Distribution"/>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">Average CO by Hour of Day</div>
      <img src="data:image/png;base64,{{ charts.co_by_hour }}" alt="CO by Hour"/>
    </div>
  </div>

  <div class="page-footer">
    <p>Data: <a href="https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set" target="_blank">kaggle.com/datasets/fedesoriano/air-quality-data-set</a></p>
    <p>Air Quality AI Project &nbsp;&middot;&nbsp; September 2026</p>
  </div>
</div>

<!-- ════ PANEL: EDA ══════════════════════════════════════════════════════════ -->
<div id="panel-eda" class="panel">
  <div class="page-hdr">
    <div class="breadcrumb"><span>Dashboard</span> / EDA Charts</div>
    <h2>Exploratory Data Analysis</h2>
    <p>Visualising sensor distributions, temporal patterns and feature correlations</p>
  </div>

  <div class="chart-grid">
    <div class="chart-wrap">
      <div class="chart-title">Feature Correlation Heatmap</div>
      <img src="data:image/png;base64,{{ charts.correlation }}" alt="Correlation"/>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">CO Distribution by AQI Category</div>
      <img src="data:image/png;base64,{{ charts.co_by_aqi }}" alt="CO by AQI"/>
    </div>
  </div>
  <div class="chart-grid one">
    <div class="chart-wrap">
      <div class="chart-title">Temperature vs NO&#8322; Concentration (coloured by AQI)</div>
      <img src="data:image/png;base64,{{ charts.temp_no2 }}" alt="Temp vs NO2"/>
    </div>
  </div>

  <div class="card">
    <div class="card-hdr"><h3><span class="hdr-icon">&#128161;</span> Key EDA Findings</h3></div>
    <div class="card-body">
      <div class="progress-row"><span class="progress-label">CO ↔ Benzene (C6H6)</span><div class="progress-track"><div class="progress-fill" style="width:97%;background:#3b82f6"></div></div><span class="progress-val">r=0.97</span></div>
      <div class="progress-row"><span class="progress-label">CO ↔ NOx</span><div class="progress-track"><div class="progress-fill" style="width:94%;background:#8b5cf6"></div></div><span class="progress-val">r=0.94</span></div>
      <div class="progress-row"><span class="progress-label">S1(CO) ↔ S2(NMHC)</span><div class="progress-track"><div class="progress-fill" style="width:95%;background:#22c55e"></div></div><span class="progress-val">r=0.95</span></div>
      <div class="progress-row"><span class="progress-label">Temp ↔ Humidity (inverse)</span><div class="progress-track"><div class="progress-fill" style="width:82%;background:#f59e0b"></div></div><span class="progress-val">r=-0.82</span></div>
    </div>
  </div>
</div>

<!-- ════ PANEL: MODEL RESULTS ════════════════════════════════════════════════ -->
<div id="panel-models" class="panel">
  <div class="page-hdr">
    <div class="breadcrumb"><span>Dashboard</span> / Model Results</div>
    <h2>Model Evaluation Results</h2>
    <p>Performance comparison across all 8 trained models — classification and regression tasks</p>
  </div>

  <div class="card">
    <div class="card-hdr">
      <h3><span class="hdr-icon">&#127937;</span> Classification — AQI Category Prediction</h3>
      <span class="tag tag-blue">80/20 stratified split</span>
    </div>
    <div class="card-body no-pad">
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr><th>Model</th><th>Test Accuracy</th><th>CV Accuracy (5-fold)</th><th>Accuracy Bar</th><th>Status</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Logistic Regression</td><td>0.8421</td><td>0.8076</td>
              <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:84.2%;background:#94a3b8"></div></div><span style="font-size:11px;color:#64748b">84.2%</span></div></td>
              <td></td>
            </tr>
            <tr class="best-row">
              <td><strong>Decision Tree</strong></td><td><strong>1.0000</strong></td><td>0.9041</td>
              <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:100%;background:#22c55e"></div></div><span style="font-size:11px;color:#16a34a;font-weight:700">100%</span></div></td>
              <td><span class="tag tag-green">Best Selected</span></td>
            </tr>
            <tr>
              <td>Random Forest</td><td>1.0000</td><td>0.9146</td>
              <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:100%;background:#3b82f6"></div></div><span style="font-size:11px;color:#2563eb;font-weight:700">100%</span></div></td>
              <td><span class="tag tag-blue">Best CV</span></td>
            </tr>
            <tr>
              <td>XGBoost</td><td>1.0000</td><td>0.9047</td>
              <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:100%;background:#3b82f6"></div></div><span style="font-size:11px;color:#2563eb;font-weight:700">100%</span></div></td>
              <td></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="chart-grid">
    <div class="chart-wrap">
      <div class="chart-title">Classifier Accuracy Comparison</div>
      <img src="data:image/png;base64,{{ charts.clf_compare }}" alt="Classifier Comparison"/>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">Confusion Matrix — Decision Tree</div>
      <img src="data:image/png;base64,{{ charts.conf_matrix }}" alt="Confusion Matrix"/>
    </div>
  </div>

  <div class="card">
    <div class="card-hdr">
      <h3><span class="hdr-icon">&#128200;</span> Regression — CO Concentration Forecasting</h3>
      <span class="tag tag-orange">Lower RMSE = better</span>
    </div>
    <div class="card-body no-pad">
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R&#178;</th><th>RMSE Bar</th><th>Status</th></tr>
          </thead>
          <tbody>
            <tr class="best-row">
              <td><strong>Linear Regression</strong></td><td><strong>1.8647</strong></td><td>1.2845</td><td>0.1191</td>
              <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:28%;background:#22c55e"></div></div><span style="font-size:11px;color:#16a34a;font-weight:700">1.86</span></div></td>
              <td><span class="tag tag-green">Best RMSE</span></td>
            </tr>
            <tr>
              <td>Random Forest Regressor</td><td>2.0921</td><td>1.5078</td><td>-0.1089</td>
              <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:35%;background:#94a3b8"></div></div><span style="font-size:11px;color:#64748b">2.09</span></div></td>
              <td></td>
            </tr>
            <tr>
              <td>Gradient Boosting</td><td>2.4845</td><td>1.6798</td><td>-0.5639</td>
              <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:45%;background:#94a3b8"></div></div><span style="font-size:11px;color:#64748b">2.48</span></div></td>
              <td></td>
            </tr>
            <tr>
              <td>XGBoost Regressor</td><td>2.5769</td><td>1.6782</td><td>-0.6825</td>
              <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:48%;background:#94a3b8"></div></div><span style="font-size:11px;color:#64748b">2.58</span></div></td>
              <td></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="chart-grid one">
    <div class="chart-wrap">
      <div class="chart-title">Actual vs Predicted CO — Linear Regression</div>
      <img src="data:image/png;base64,{{ charts.regression }}" alt="Regression Plot"/>
    </div>
  </div>
</div>

<!-- ════ PANEL: FEATURE IMPORTANCE ══════════════════════════════════════════ -->
<div id="panel-features" class="panel">
  <div class="page-hdr">
    <div class="breadcrumb"><span>Dashboard</span> / Feature Importance</div>
    <h2>Feature Importance</h2>
    <p>Top pollution-driving variables identified via Random Forest for CO prediction</p>
  </div>

  <div class="card">
    <div class="card-hdr"><h3><span class="hdr-icon">&#127959;</span> Top Pollution Drivers</h3></div>
    <div class="card-body">
      <div class="progress-row"><span class="progress-label">AH (Absolute Humidity)</span><div class="progress-track"><div class="progress-fill" style="width:100%;background:#3b82f6"></div></div><span class="progress-val">0.195</span></div>
      <div class="progress-row"><span class="progress-label">RH (Relative Humidity)</span><div class="progress-track"><div class="progress-fill" style="width:87%;background:#8b5cf6"></div></div><span class="progress-val">0.170</span></div>
      <div class="progress-row"><span class="progress-label">PT08.S3(NOx)</span><div class="progress-track"><div class="progress-fill" style="width:53%;background:#f59e0b"></div></div><span class="progress-val">0.103</span></div>
      <div class="progress-row"><span class="progress-label">CO_NO2_ratio</span><div class="progress-track"><div class="progress-fill" style="width:49%;background:#22c55e"></div></div><span class="progress-val">0.096</span></div>
      <div class="progress-row"><span class="progress-label">Hour</span><div class="progress-track"><div class="progress-fill" style="width:36%;background:#ef4444"></div></div><span class="progress-val">0.070</span></div>
      <div class="divider"></div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Rank</th><th>Feature</th><th>Importance</th><th>Description</th></tr></thead>
          <tbody>
            <tr class="best-row"><td>1</td><td>AH — Absolute Humidity</td><td>0.1947</td><td>Atmospheric moisture trapping effect</td></tr>
            <tr class="best-row"><td>2</td><td>RH — Relative Humidity</td><td>0.1699</td><td>Humidity-pollution interaction</td></tr>
            <tr><td>3</td><td>PT08.S3(NOx)</td><td>0.1033</td><td>Tungsten oxide NOx sensor reading</td></tr>
            <tr><td>4</td><td>CO_NO2_ratio</td><td>0.0956</td><td>Engineered combustion-source differentiator</td></tr>
            <tr><td>5</td><td>Hour</td><td>0.0696</td><td>Time-of-day traffic cycle signal</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="chart-grid one">
    <div class="chart-wrap">
      <div class="chart-title">Feature Importance — Random Forest Regressor for CO</div>
      <img src="data:image/png;base64,{{ charts.feat_imp }}" alt="Feature Importance"/>
    </div>
  </div>
</div>

<!-- ════ PANEL: LIVE PREDICT ════════════════════════════════════════════════ -->
<div id="panel-predict" class="panel">
  <div class="page-hdr">
    <div class="breadcrumb"><span>Dashboard</span> / Live Predict</div>
    <h2>Live AQI + CO Prediction</h2>
    <p>Enter sensor readings to get real-time predictions from the trained models</p>
  </div>

  <div class="card">
    <div class="card-hdr">
      <h3><span class="hdr-icon">&#9889;</span> Sensor Input</h3>
      <div style="display:flex;gap:8px">
        <button class="btn btn-outline-green" onclick="fillExample('good')">Good Example</button>
        <button class="btn btn-outline-orange" onclick="fillExample('moderate')">Moderate</button>
        <button class="btn btn-outline-red" onclick="fillExample('hazardous')">Hazardous</button>
      </div>
    </div>
    <div class="card-body">
      <div class="section-label">Pollutant Sensors</div>
      <div class="form-grid">
        <div class="field">
          <label>CO(GT)</label>
          <input type="number" id="f_co" step="0.1" value="2.6"/>
          <span class="field-hint">Carbon monoxide mg/m&#179;</span>
        </div>
        <div class="field">
          <label>C6H6(GT)</label>
          <input type="number" id="f_c6h6" step="0.1" value="11.9"/>
          <span class="field-hint">Benzene &#956;g/m&#179;</span>
        </div>
        <div class="field">
          <label>NOx(GT)</label>
          <input type="number" id="f_nox" step="1" value="166"/>
          <span class="field-hint">Nitrogen oxides ppb</span>
        </div>
        <div class="field">
          <label>NO2(GT)</label>
          <input type="number" id="f_no2" step="1" value="113"/>
          <span class="field-hint">Nitrogen dioxide &#956;g/m&#179;</span>
        </div>
      </div>
      <div class="section-label">Metal Oxide Sensors</div>
      <div class="form-grid">
        <div class="field">
          <label>PT08.S1(CO)</label>
          <input type="number" id="f_s1" step="1" value="1360"/>
          <span class="field-hint">Tin oxide — CO proxy</span>
        </div>
        <div class="field">
          <label>PT08.S2(NMHC)</label>
          <input type="number" id="f_s2" step="1" value="1046"/>
          <span class="field-hint">Titania — NMHC proxy</span>
        </div>
        <div class="field">
          <label>PT08.S3(NOx)</label>
          <input type="number" id="f_s3" step="1" value="1056"/>
          <span class="field-hint">Tungsten oxide — NOx proxy</span>
        </div>
      </div>
      <div class="section-label">Meteorological Readings</div>
      <div class="form-grid">
        <div class="field">
          <label>Temperature</label>
          <input type="number" id="f_t" step="0.1" value="13.6"/>
          <span class="field-hint">&#176;C</span>
        </div>
        <div class="field">
          <label>Relative Humidity</label>
          <input type="number" id="f_rh" step="0.1" value="48.9"/>
          <span class="field-hint">Percentage %</span>
        </div>
        <div class="field">
          <label>Absolute Humidity</label>
          <input type="number" id="f_ah" step="0.001" value="0.7578"/>
          <span class="field-hint">g/m&#179;</span>
        </div>
        <div class="field">
          <label>Hour of Day</label>
          <input type="number" id="f_hour" step="1" min="0" max="23" value="18"/>
          <span class="field-hint">0 = midnight, 23 = 11pm</span>
        </div>
      </div>

      <div class="btn-row">
        <button class="btn btn-primary" onclick="runPredict()">&#9889; Run Prediction</button>
        <button class="btn btn-ghost" onclick="clearResult()">Clear</button>
      </div>

      <div class="spinner" id="spinner"></div>

      <div class="pred-result" id="predResult">
        <div class="pred-banner" id="predBanner">
          <div>
            <div class="aqi-big" id="predAqi"></div>
            <div class="aqi-sub" id="predConf"></div>
          </div>
        </div>
        <div class="pred-body">
          <div class="pred-metric">
            <span class="label">AQI Category</span>
            <span class="value" id="predAqiVal"></span>
            <span class="unit">EPA classification</span>
          </div>
          <div class="pred-metric">
            <span class="label">Predicted CO</span>
            <span class="value" id="predCo"></span>
            <span class="unit">mg/m&#179; Carbon monoxide</span>
          </div>
          <div class="pred-advice" id="predAdvice"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-hdr"><h3><span class="hdr-icon">&#128279;</span> REST API Reference</h3></div>
    <div class="card-body">
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Endpoint</th><th>Method</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><code>/predict/classify</code></td><td><span class="method-badge method-post">POST</span></td><td>Returns AQI label + confidence score</td></tr>
            <tr><td><code>/predict/co</code></td><td><span class="method-badge method-post">POST</span></td><td>Returns predicted CO concentration (mg/m&#179;)</td></tr>
            <tr><td><code>/results</code></td><td><span class="method-badge method-get">GET</span></td><td>Returns all model metrics as JSON</td></tr>
          </tbody>
        </table>
      </div>
      <div style="margin-top:14px">
        <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px">Example Request Body</div>
        <div class="code-block"><span class="kw">{</span>
  <span class="str">"CO_GT"</span>: 2.6,  <span class="str">"C6H6_GT"</span>: 11.9,  <span class="str">"NOx_GT"</span>: 166,  <span class="str">"NO2_GT"</span>: 113,
  <span class="str">"S1_CO"</span>: 1360, <span class="str">"S2_NMHC"</span>: 1046, <span class="str">"S3_NOx"</span>: 1056,
  <span class="str">"T"</span>: 13.6, <span class="str">"RH"</span>: 48.9, <span class="str">"AH"</span>: 0.7578, <span class="str">"Hour"</span>: 18
<span class="kw">}</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ════ PANEL: PIPELINE ════════════════════════════════════════════════════ -->
<div id="panel-pipeline" class="panel">
  <div class="page-hdr">
    <div class="breadcrumb"><span>Dashboard</span> / Pipeline</div>
    <h2>ML Pipeline Architecture</h2>
    <p>Complete end-to-end data flow from raw CSV to trained models and live predictions</p>
  </div>

  <div class="card">
    <div class="card-hdr"><h3><span class="hdr-icon">&#9881;</span> Pipeline Flow</h3></div>
    <div class="card-body no-pad">
      <div class="pipeline-box"><span class="step">Raw CSV</span>  data/air_quality_dataset.csv  (94 rows, 16 cols)
<span class="arrow">        |
        v</span>
<span class="step">[Step 1]</span> Data Loading &amp; Cleaning
         Replace -200 sentinels with NaN
         Linear interpolation  |  dropna
<span class="arrow">        |
        v</span>
<span class="step">[Step 2]</span> Exploratory Data Analysis
         AQI distribution  |  CO by hour  |  Correlation heatmap
         <span class="out">--&gt; outputs/01...05_*.png</span>
<span class="arrow">        |
        v</span>
<span class="step">[Step 3]</span> Feature Engineering
         CO_NO2_ratio  |  Pollution_Index  |  Peak_Hour  |  Weekend
<span class="arrow">        |
        +------------------+------------------+</span>
        |                                     |
<span class="step">  [Step 4] Classification          [Step 5] Regression</span>
  Logistic Regression               Linear Regression
  Decision Tree                     Random Forest
  Random Forest                     Gradient Boosting
  XGBoost                           XGBoost
  <span class="out">--&gt; outputs/06,07_*.png          --&gt; outputs/08_*.png</span>
<span class="arrow">        |                                     |
        +------------------+------------------+
        |
        v</span>
<span class="step">[Step 6]</span> Feature Importance  <span class="out">--&gt; outputs/09_feature_importance.png</span>
<span class="arrow">        |
        v</span>
<span class="step">[Step 7]</span> Save Models
         <span class="out">models/best_classifier.pkl  |  models/best_regressor.pkl</span>
         <span class="out">models/scaler.pkl           |  models/label_encoder.pkl</span>
<span class="arrow">        |
        v</span>
<span class="step">[Flask]</span>  Web Dashboard  +  REST API
         GET  /                  <span class="cm"># this dashboard</span>
         POST /predict/classify  <span class="cm"># AQI label + confidence</span>
         POST /predict/co        <span class="cm"># CO concentration (mg/m3)</span>
         GET  /results           <span class="cm"># all model metrics as JSON</span></div>
    </div>
  </div>

  <div class="card">
    <div class="card-hdr"><h3><span class="hdr-icon">&#9881;</span> Engineered Features</h3></div>
    <div class="card-body no-pad">
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Feature</th><th>Formula</th><th>Rationale</th></tr></thead>
          <tbody>
            <tr><td><code>CO_NO2_ratio</code></td><td>CO / (NO&#8322; + &#949;)</td><td>Differentiates combustion sources</td></tr>
            <tr><td><code>Pollution_Index</code></td><td>0.4&#215;CO + 0.35&#215;NOx + 0.25&#215;NO&#8322;</td><td>Weighted composite pollution score</td></tr>
            <tr><td><code>Peak_Hour</code></td><td>1 if 07&#8211;09h or 17&#8211;19h, else 0</td><td>Rush-hour traffic binary flag</td></tr>
            <tr><td><code>Weekend</code></td><td>1 if Sat/Sun, else 0</td><td>Traffic pattern modulation</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

</main>
</div><!-- end shell -->

<script>
  const AQI_COLORS = {
    'Good':'#22c55e','Moderate':'#f59e0b',
    'Unhealthy for Sensitive Groups':'#f97316',
    'Unhealthy':'#ef4444','Very Unhealthy':'#a855f7','Hazardous':'#7f0000'
  };
  const AQI_ADVICE = {
    'Good':'Air quality is satisfactory — no health risk.',
    'Moderate':'Acceptable for most people; sensitive individuals may be mildly affected.',
    'Unhealthy for Sensitive Groups':'Sensitive individuals (elderly, children, respiratory conditions) at risk.',
    'Unhealthy':'Everyone may begin to experience health effects.',
    'Very Unhealthy':'Serious health effects for everyone — limit outdoor exposure.',
    'Hazardous':'Emergency conditions — avoid all outdoor activity.'
  };

  function showPanel(name, btn) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.sidebar-item').forEach(b => b.classList.remove('active'));
    document.getElementById('panel-' + name).classList.add('active');
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
    const sp  = document.getElementById('spinner');
    const res = document.getElementById('predResult');
    sp.style.display = 'block';
    res.classList.remove('show');
    const opts = {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
    try {
      const [clfR, regR] = await Promise.all([
        fetch('/predict/classify', opts).then(r=>r.json()),
        fetch('/predict/co',       opts).then(r=>r.json()),
      ]);
      sp.style.display = 'none';
      if (clfR.error || regR.error) {
        document.getElementById('predAqi').textContent = 'Error';
        document.getElementById('predAdvice').textContent = clfR.error || regR.error;
        document.getElementById('predBanner').style.background = '#64748b';
        res.classList.add('show'); return;
      }
      const aqi   = clfR.aqi_label;
      const conf  = (clfR.confidence * 100).toFixed(1);
      const co    = regR.predicted_co.toFixed(3);
      const color = AQI_COLORS[aqi] || '#3b82f6';
      document.getElementById('predBanner').style.background = color;
      document.getElementById('predAqi').textContent = aqi;
      document.getElementById('predConf').textContent = 'Model confidence: ' + conf + '%';
      document.getElementById('predAqiVal').textContent = aqi;
      document.getElementById('predAqiVal').style.color = color;
      document.getElementById('predCo').textContent = co;
      document.getElementById('predAdvice').textContent = AQI_ADVICE[aqi] || '';
      document.getElementById('predAdvice').style.borderLeftColor = color;
      res.classList.add('show');
    } catch(e) {
      sp.style.display = 'none';
      alert('Prediction failed: ' + e.message);
    }
  }

  function clearResult() {
    document.getElementById('predResult').classList.remove('show');
  }

  function fillExample(type) {
    const ex = {
      good:      {co:0.8,  c6h6:3.0,  nox:50,  no2:45,  s1:900,  s2:750,  s3:1400, t:18.0, rh:55.0, ah:0.710, h:14},
      moderate:  {co:2.6,  c6h6:11.9, nox:166, no2:113, s1:1360, s2:1046, s3:1056, t:13.6, rh:48.9, ah:0.758, h:18},
      hazardous: {co:11.0, c6h6:28.0, nox:450, no2:260, s1:2100, s2:1900, s3:650,  t:10.0, rh:75.0, ah:0.900, h:8 },
    }[type];
    document.getElementById('f_co').value=ex.co; document.getElementById('f_c6h6').value=ex.c6h6;
    document.getElementById('f_nox').value=ex.nox; document.getElementById('f_no2').value=ex.no2;
    document.getElementById('f_s1').value=ex.s1;  document.getElementById('f_s2').value=ex.s2;
    document.getElementById('f_s3').value=ex.s3;  document.getElementById('f_t').value=ex.t;
    document.getElementById('f_rh').value=ex.rh;  document.getElementById('f_ah').value=ex.ah;
    document.getElementById('f_hour').value=ex.h;
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
