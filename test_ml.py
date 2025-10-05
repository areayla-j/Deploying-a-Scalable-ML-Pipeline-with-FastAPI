# test_ml.py
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# ----- Fixtures -----
@pytest.fixture
def sample_data():
    # minimal, but includes both numeric + categorical + label
    return pd.DataFrame(
        {
            "age": [25, 32, 47, 38, 52],
            "workclass": ["Private", "Self-emp", "Government", "Private", "Private"],
            "education": ["Bachelors", "Masters", "PhD", "Bachelors", "HS-grad"],
            "salary": ["<=50K", ">50K", ">50K", "<=50K", ">50K"],
        }
    )

# columns used by process_data in this test
CAT_FEATURES = ["workclass", "education"]
LABEL = "salary"


# ----- Tests -----
def test_train_model_returns_randomforest(sample_data):
    """The trained model should be a RandomForestClassifier."""
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_model_inference_shape(sample_data):
    """Prediction count should match number of rows."""
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape[0] == y.shape[0]


def test_compute_model_metrics_range(sample_data):
    """Precision/Recall/F1 should be within [0, 1]."""
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, f1 = compute_model_metrics(y, preds)
    for m in (precision, recall, f1):
        assert 0.0 <= m <= 1.0

