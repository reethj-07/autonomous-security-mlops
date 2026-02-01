import pytest
import pandas as pd
import numpy as np
from src.models.anomaly import SecurityIsolationForest


class TestSecurityIsolationForest:
    """Unit tests for anomaly detection model."""

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = SecurityIsolationForest()
        assert model.model is not None
        assert model.model.n_estimators == 200

    def test_model_fit(self, sample_features):
        """Test model fitting on sample data."""
        model = SecurityIsolationForest()
        model.fit(sample_features.drop("label", axis=1))
        # Model should fit without errors
        assert model.model is not None

    def test_score_output_shape(self, sample_features):
        """Test that score output has correct shape."""
        model = SecurityIsolationForest()
        X = sample_features.drop("label", axis=1)
        model.fit(X)
        scores = model.score(X)
        
        assert scores.shape == (len(X),)
        assert np.all(scores >= 0)

    def test_predict_binary_output(self, sample_features):
        """Test that predictions are binary (0 or 1)."""
        model = SecurityIsolationForest()
        X = sample_features.drop("label", axis=1)
        model.fit(X)
        predictions = model.predict(X, threshold=0.5)
        
        assert np.all((predictions == 0) | (predictions == 1))
        assert len(predictions) == len(X)

    def test_predict_with_custom_threshold(self, sample_features):
        """Test predictions with custom threshold."""
        model = SecurityIsolationForest()
        X = sample_features.drop("label", axis=1)
        model.fit(X)
        
        preds_low = model.predict(X, threshold=0.2)
        preds_high = model.predict(X, threshold=0.8)
        
        # Higher threshold should give fewer anomalies
        assert preds_low.sum() >= preds_high.sum()

    def test_custom_contamination(self):
        """Test model with custom contamination parameter."""
        model = SecurityIsolationForest(contamination=0.05)
        assert model.model.contamination == 0.05

    def test_score_higher_for_anomalies(self):
        """Test that scores are higher for injected anomalies."""
        np.random.seed(42)
        # Normal data
        X_normal = np.random.randn(100, 5)
        # Anomalies
        X_anomaly = np.random.uniform(10, 20, (10, 5))
        
        model = SecurityIsolationForest()
        model.fit(X_normal)
        
        scores_normal = model.score(pd.DataFrame(X_normal))
        scores_anomaly = model.score(pd.DataFrame(X_anomaly))
        
        # On average, anomalies should have higher scores
        assert scores_anomaly.mean() > scores_normal.mean()
