"""Generate SHAP bar and beeswarm plots for the best tuned model(s).
Saves images to output/graficos/shap/. Expects tuned pipeline files in
output/models/*_optuna_pipeline.pkl.
"""
import os
import sys
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from titanic_pipeline.preprocessing import AdvancedFeatureEngineer

try:
    import shap
except Exception as e:
    raise RuntimeError("shap not available") from e

MODEL_DIR = "output/models"
OUT_DIR = "output/graficos/shap"
os.makedirs(OUT_DIR, exist_ok=True)

# Find tuned pipelines
pipelines = [
    f for f in os.listdir(MODEL_DIR) if f.endswith("_optuna_pipeline.pkl")
]
if not pipelines:
    print("No tuned pipelines found in output/models/; aborting")
    sys.exit(0)

# Load data and run feature engineering (same as training pipeline)
train = pd.read_csv("data/raw/train.csv")
# Use full train for SHAP computation (or subset if very large)
fe = AdvancedFeatureEngineer()
train_fe, _ = fe.fit_transform(train.copy(), train.copy())
feature_cols = [
    c
    for c in train_fe.columns
    if c not in ["PassengerId", "Survived", "Name", "Ticket", "Cabin", "Title", "AgeGroup"]
]

# We'll compute SHAP for the best tuned pipeline (by metrics file if present)
metrics_aug_path = "output/relatorios/metrics_optuna_augmented.json"
best_name = None
if os.path.exists(metrics_aug_path):
    import json

    m = json.load(open(metrics_aug_path))
    retr = m.get("retrained_models", {})
    if retr:
        best_name = max(retr.keys(), key=lambda k: retr[k]["accuracy"])

# Fallback: pick first pipeline
if not best_name:
    p = pipelines[0]
    best_name = p.replace("_optuna_pipeline.pkl", "")

pipeline_path = os.path.join(MODEL_DIR, f"{best_name}_optuna_pipeline.pkl")
if not os.path.exists(pipeline_path):
    print("Pipeline not found:", pipeline_path)
    sys.exit(1)

with open(pipeline_path, "rb") as f:
    pipeline = pickle.load(f)

preprocessor = pipeline.named_steps.get("preprocessor")
model = pipeline.named_steps.get("model")

if preprocessor is None or model is None:
    print("Pipeline invalid: missing steps")
    sys.exit(1)

# Prepare X
X_df = train_fe[feature_cols]
X_trans = preprocessor.transform(X_df)
try:
    X_np = X_trans.toarray()
except Exception:
    X_np = np.asarray(X_trans)

# Limit to subset if too large
if X_np.shape[0] > 2000:
    sample_idx = np.random.RandomState(42).choice(X_np.shape[0], 2000, replace=False)
    X_sample = X_np[sample_idx]
    X_df_sample = X_df.iloc[sample_idx]
else:
    X_sample = X_np
    X_df_sample = X_df

print(f"Generating SHAP for {best_name} on {X_sample.shape[0]} samples...")

# Create explainer
try:
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
except Exception:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        raise RuntimeError("Failed to create SHAP explainer") from e

# Bar plot (summary by mean(|shap|))
try:
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    out_bar = os.path.join(OUT_DIR, f"{best_name}_shap_bar.png")
    plt.savefig(out_bar, bbox_inches="tight", dpi=200)
    plt.close()
    print("Saved SHAP bar:", out_bar)
except Exception as e:
    print("Failed to save SHAP bar:", e)

# Beeswarm (may be heavy)
try:
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    out_bee = os.path.join(OUT_DIR, f"{best_name}_shap_beeswarm.png")
    plt.savefig(out_bee, bbox_inches="tight", dpi=200)
    plt.close()
    print("Saved SHAP beeswarm:", out_bee)
except Exception as e:
    print("Failed to save SHAP beeswarm:", e)

print("SHAP generation complete.")