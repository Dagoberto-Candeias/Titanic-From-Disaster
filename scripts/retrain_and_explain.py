"""Retrain top models with best_params_*.json, generate SHAP and calibration plots,
and update metrics JSON. Intended to be run from repo root.
"""
import json
import os
from datetime import datetime
import sys

# Ensure repository root is on sys.path so local package imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from titanic_pipeline.preprocessing import AdvancedFeatureEngineer, create_feature_pipeline
from titanic_pipeline.core.modeling import save_model_pipeline
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Optional imports
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False
    LGBMClassifier = None

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "graficos", "shap"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "graficos", "calibration"), exist_ok=True)

# Load data
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

ef = AdvancedFeatureEngineer()
train_fe, test_fe = ef.fit_transform(train.copy(), test.copy())

# Determine feature cols (same logic as train.py)
feature_cols = [
    col
    for col in train_fe.columns
    if col not in ["PassengerId", "Survived", "Name", "Ticket", "Cabin", "Title", "AgeGroup"]
]

# Create preprocessor
preprocessor = create_feature_pipeline(df=train_fe, feature_cols=feature_cols)
X = preprocessor.fit_transform(train_fe[feature_cols])
y = train_fe["Survived"]

# If X is sparse or has toarray
try:
    X_np = X.toarray()
except Exception:
    X_np = np.asarray(X)

# Load best params files
best_params_files = [f for f in os.listdir("output") if f.startswith("best_params_") and f.endswith('.json')]
best_models = {}
for fname in best_params_files:
    model_key = fname.replace("best_params_", "").replace('.json','')
    params = json.load(open(os.path.join("output", fname)))
    best_models[model_key] = params

# Retrain and evaluate
results = {}
for model_name, params in best_models.items():
    print(f"Retraining {model_name} with params: {params}")
    if model_name.lower().startswith("random") or "Random" in model_name:
        model = RandomForestClassifier(**params, random_state=42)
    elif model_name.lower().startswith("xg") and XGB_AVAILABLE:
        model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    elif model_name.lower().startswith("light") and LGBM_AVAILABLE:
        model = LGBMClassifier(**params, random_state=42)
    else:
        print(f"Skipping {model_name}: Unsupported or dependency missing")
        continue

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X_np, y, cv=rskf, scoring='accuracy')
    model.fit(X_np, y)
    results[model_name] = {
        'mean_score': float(scores.mean()),
        'std_score': float(scores.std()),
        'trained_model': model,
        'best_params': params,
    }

    # Save pipeline for the tuned model
    save_path = os.path.join(OUTPUT_DIR, 'models', f'{model_name}_optuna_pipeline.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_model_pipeline(preprocessor, model, save_path)
    print(f"Saved pipeline to {save_path}")

# Choose best model for SHAP and calibration
best_model_name = max(results.keys(), key=lambda k: results[k]['mean_score']) if results else None
if best_model_name:
    print('Best tuned model:', best_model_name, results[best_model_name]['mean_score'])
    model = results[best_model_name]['trained_model']

    # SHAP
    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(model.predict_proba, X_np)
            shap_values = explainer(X_np)
            # Summary plot (bar)
            plt.figure()
            shap.plots.bar(shap_values, show=False)
            plt.savefig(os.path.join(OUTPUT_DIR, 'graficos', 'shap', f'{best_model_name}_shap_bar.png'), bbox_inches='tight')
            plt.close()
            # Beeswarm
            plt.figure()
            shap.plots.beeswarm(shap_values, show=False)
            plt.savefig(os.path.join(OUTPUT_DIR, 'graficos', 'shap', f'{best_model_name}_shap_beeswarm.png'), bbox_inches='tight')
            plt.close()
            print('SHAP plots saved')
        except Exception as e:
            print('SHAP generation failed:', e)
    else:
        print('SHAP not available, skipping SHAP generation')

    # Calibration
    try:
        from sklearn.calibration import calibration_curve
        probs = model.predict_proba(X_np)[:,1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y, probs, n_bins=10)
        plt.figure()
        plt.plot(mean_predicted_value, fraction_of_positives, 's-')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('Mean predicted value')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration curve - {best_model_name}')
        plt.savefig(os.path.join(OUTPUT_DIR, 'graficos', 'calibration', f'{best_model_name}_calibration.png'))
        plt.close()
        print('Calibration plot saved')
    except Exception as e:
        print('Calibration failed:', e)

# Update metrics file
metrics_path = os.path.join(OUTPUT_DIR, 'relatorios', 'metrics_optuna_augmented.json')
metrics_out = {
    'timestamp': datetime.now().isoformat(),
    'validation_type': 'RepeatedStratifiedKFold(n_splits=5, n_repeats=3)',
    'retrained_models': {k: {'accuracy': v['mean_score'], 'std': v['std_score'], 'best_params': v['best_params'], 'scores': scores.tolist()} for k,v in results.items()}
}
json.dump(metrics_out, open(metrics_path,'w'), indent=2)
print('Metrics saved to', metrics_path)
print('Done')