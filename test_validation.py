"""Quick validation test script for Titanic ML Pipeline."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import sys

def main():
    print("=" * 70)
    print("QUICK VALIDATION TEST - TITANIC ML PIPELINE")
    print("=" * 70)
    
    try:
        # 1. Load data
        print("\n[1/6] Loading data...")
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        print(f"      Train shape: {train.shape}")
        print(f"      Test shape: {test.shape}")
        
        # 2. Basic preprocessing
        print("\n[2/6] Basic preprocessing...")
        train_clean = train.dropna(subset=['Age', 'Fare'])
        X = train_clean[['Age', 'Fare', 'SibSp', 'Parch']].fillna(0)
        y = train_clean['Survived']
        print(f"      X shape: {X.shape}, y shape: {y.shape}")
        
        # 3. Train quick model
        print("\n[3/6] Training simple RandomForest (10 trees)...")
        model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        model.fit(X, y)
        train_score = model.score(X, y)
        print(f"      Model trained. Training score: {train_score:.4f}")
        
        # 4. Make predictions
        print("\n[4/6] Making predictions on test set...")
        test_features = test[['Age', 'Fare', 'SibSp', 'Parch']].fillna(0)
        predictions = model.predict(test_features)
        print(f"      Predictions shape: {predictions.shape}")
        print(f"      Prediction sample: {predictions[:5]}")
        
        # 5. Save predictions
        print("\n[5/6] Saving predictions...")
        submission = pd.DataFrame({
            'PassengerId': test['PassengerId'],
            'Survived': predictions
        })
        os.makedirs('output', exist_ok=True)
        submission.to_csv('output/submission_quick_test.csv', index=False)
        print(f"      Saved to: output/submission_quick_test.csv")
        print(f"      Submission shape: {submission.shape}")
        
        # 6. Save model
        print("\n[6/6] Saving model...")
        os.makedirs('output/models', exist_ok=True)
        with open('output/models/quick_test_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"      Model saved to: output/models/quick_test_model.pkl")
        
        print("\n" + "=" * 70)
        print("SUCCESS - QUICK VALIDATION TEST PASSED")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
