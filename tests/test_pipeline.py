import pandas as pd
from sklearn.linear_model import LogisticRegression

from ELT579_118550_Titanic_DOCUMENTADO_ComRelatorio import (
    preprocess_data,
    train_single_model,
    validate_data_schema,
)


def test_validate_data_schema_basic():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    assert validate_data_schema(df, ["A", "B"], "unit_test_basic") is True


def test_preprocess_data_shapes():
    train = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "Survived": [0, 1, 0]})
    test = pd.DataFrame({"A": [1, 2], "B": [4, 5]})
    X_train, X_test, y_train, preprocessor = preprocess_data(train, test, ["A", "B"])
    # X_train/X_test are arrays (sparsed/ndarray). Check row counts
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert len(y_train) == 3


def test_train_single_model_returns_model():
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 0]
    res = train_single_model("test_model", LogisticRegression(), X, y, cv_folds=2)
    assert res.get("trained_model") is not None
    assert res.get("mean_score") is not None
