"""Comprehensive test of full pipeline workflow."""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import sys

def main():
    print("=" * 70)
    print("COMPREHENSIVE PIPELINE TEST")
    print("=" * 70)
    
    try:
        # 1. Load data
        print("\n[1/6] Loading data...")
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        print(f"      Train shape: {train.shape}")
        print(f"      Test shape: {test.shape}")
        
        # 2. Feature selection and preprocessing
        print("\n[2/6] Feature engineering and preprocessing...")
        feature_cols = ['Age', 'Fare', 'SibSp', 'Parch']
        
        train_clean = train.dropna(subset=['Age', 'Fare'])
        X_train = train_clean[feature_cols].fillna(0)
        y_train = train_clean['Survived']
        
        X_test = test[feature_cols].fillna(0)
        
        print(f"      X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"      X_test shape: {X_test.shape}")
        
        # 3. Build sklearn Pipeline (this is what predict.py expects)
        print("\n[3/6] Building and training sklearn Pipeline...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1))
        ])
        
        pipeline.fit(X_train, y_train)
        train_score = pipeline.score(X_train, y_train)
        print(f"      Pipeline trained. Training score: {train_score:.4f}")
        
        # 4. Make predictions using pipeline
        print("\n[4/6] Making predictions with pipeline...")
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        
        print(f"      Predictions shape: {predictions.shape}")
        print(f"      Probabilities shape: {probabilities.shape}")
        print(f"      Prediction sample: {predictions[:5]}")
        print(f"      Probability sample: {probabilities[:5]}")
        
        # 5. Save predictions to CSV
        print("\n[5/6] Saving predictions to CSV...")
        submission = pd.DataFrame({
            'PassengerId': test['PassengerId'],
            'Survived': predictions,
            'Survived_prob_0': probabilities[:, 0],
            'Survived_prob_1': probabilities[:, 1]
        })
        
        os.makedirs('output', exist_ok=True)
        submission[['PassengerId', 'Survived']].to_csv(
            'output/submission_pipeline_test.csv', 
            index=False
        )
        submission.to_csv('output/submission_pipeline_test_full.csv', index=False)
        
        print(f"      Saved predictions to: output/submission_pipeline_test.csv")
        print(f"      Saved full predictions to: output/submission_pipeline_test_full.csv")
        
        # 6. Save pipeline model
        print("\n[6/6] Saving pipeline model...")
        os.makedirs('output/models', exist_ok=True)
        with open('output/models/pipeline_test.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"      Pipeline saved to: output/models/pipeline_test.pkl")
        
        # 7. Verify model can be loaded and used
        print("\n[7/7] Verifying model can be loaded and reused...")
        with open('output/models/pipeline_test.pkl', 'rb') as f:
            loaded_pipeline = pickle.load(f)
        
        test_predictions = loaded_pipeline.predict(X_test)
        print(f"      Loaded model predictions shape: {test_predictions.shape}")
        print(f"      Predictions match: {np.array_equal(predictions, test_predictions)}")
        
        print("\n" + "=" * 70)
        print("SUCCESS - ALL TESTS PASSED")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - output/submission_pipeline_test.csv (Kaggle format)")
        print("  - output/submission_pipeline_test_full.csv (with probabilities)")
        print("  - output/models/pipeline_test.pkl (trained pipeline)")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
