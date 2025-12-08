"""Regenerate failed plots: feature correlation heatmap and ROC curves."""
import json
from pathlib import Path
import pandas as pd
import pickle

from titanic_pipeline.core.reporting import generate_feature_correlation_heatmap, generate_roc_curves

def main():
    repo_root = Path('.')
    train_path = repo_root / 'train.csv'
    models_dir = repo_root / 'output' / 'models'
    best_model_path = models_dir / 'best_model_pipeline.pkl'

    # Load train data
    train = pd.read_csv(train_path)

    # Determine feature columns: use numeric columns excluding target
    feature_cols = [c for c in train.columns.tolist() if c not in ('PassengerId', 'Name', 'Survived')]

    # Load best pipeline as a single trained model for ROC generation
    model_results = {}
    if best_model_path.exists():
        with open(best_model_path, 'rb') as f:
            pipeline = pickle.load(f)
        model_results['Ensemble_Stacking'] = {'trained_model': pipeline}

    # regenerate heatmap
    print('Regenerating feature correlation heatmap...')
    generate_feature_correlation_heatmap(train, feature_cols)
    print('Heatmap regeneration done.')

    # regenerate ROC curves (if we have a model and training labels)
    if 'Ensemble_Stacking' in model_results:
        print('Regenerating ROC curves...')
        try:
            y_train = train['Survived']
            generate_roc_curves(model_results, train[feature_cols], y_train, feature_cols=feature_cols)
            print('ROC regeneration done.')
        except Exception as e:
            print('ROC regeneration failed:', e)
    else:
        print('No trained models found to regenerate ROC curves.')

if __name__ == '__main__':
    main()
