import pandas as pd
from features import AdvancedFeatureEngineer

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Initialize feature engineer
fe = AdvancedFeatureEngineer()

# Create features
train_featured = fe.create_advanced_features(train, is_training=True)
test_featured = fe.create_advanced_features(test, is_training=False)

print(f"Train original columns: {len(train.columns)}")
print(f"Train featured columns: {len(train_featured.columns)}")
print(f"Test featured columns: {len(test_featured.columns)}")

# Simulate preprocessing (basic one-hot)
train_dummies = pd.get_dummies(
    train_featured.drop(
        ["PassengerId", "Name", "Ticket", "Cabin", "Survived"], axis=1, errors="ignore"
    ),
    drop_first=True,
)
test_dummies = pd.get_dummies(
    test_featured.drop(
        ["PassengerId", "Name", "Ticket", "Cabin"], axis=1, errors="ignore"
    ),
    drop_first=True,
)

print(f"Train after dummies: {len(train_dummies.columns)}")
print(f"Test after dummies: {len(test_dummies.columns)}")

# Feature selection
selected_features, selector = fe.select_features_via_model(
    train_dummies.values,
    train["Survived"].values,
    train_dummies.columns.tolist(),
    method="rf_importance",
    threshold=0.01,
)

print(f"Selected features: {len(selected_features)}")
print("Selected features list:", selected_features[:10])  # First 10
