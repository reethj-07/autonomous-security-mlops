import pytest
import pandas as pd
import numpy as np
from src.monitoring.psi import calculate_psi
from src.monitoring.drift import should_retrain


class TestDriftDetection:
    """Unit tests for drift detection and monitoring."""

    def test_psi_calculation_identical_distribution(self):
        """Test PSI is near 0 for identical distributions."""
        data = np.random.normal(0, 1, 1000)
        series = pd.Series(data)
        
        psi = calculate_psi(series, series)
        assert psi < 0.01, "PSI should be near 0 for identical distributions"

    def test_psi_detects_shifted_distribution(self):
        """Test PSI detects shifted distribution."""
        expected = pd.Series(np.random.normal(0, 1, 1000))
        actual = pd.Series(np.random.normal(2, 1, 1000))  # Shifted mean
        
        psi = calculate_psi(expected, actual)
        assert psi > 0.5, "PSI should detect significant distribution shift"

    def test_psi_buckets_parameter(self):
        """Test PSI with different bucket sizes."""
        expected = pd.Series(np.random.uniform(0, 100, 1000))
        actual = pd.Series(np.random.uniform(0, 100, 1000))
        
        psi_5 = calculate_psi(expected, actual, buckets=5)
        psi_20 = calculate_psi(expected, actual, buckets=20)
        
        # Both should be computed without errors
        assert isinstance(psi_5, float)
        assert isinstance(psi_20, float)

    def test_should_retrain_feature_drift_flag(self):
        """Test retraining decision with feature drift."""
        feature_drift = {
            "feature_1": 0.3,
            "feature_2": 0.15,
            "feature_3": 0.1,
        }
        
        result = should_retrain(
            feature_drift=feature_drift,
            psi_threshold=0.25
        )
        
        assert result["retrain"] == True
        assert result["feature_drift"] == True
        assert "psi_threshold" in result

    def test_should_retrain_prediction_drift_flag(self):
        """Test retraining decision with prediction drift."""
        feature_drift = {"f1": 0.1}
        result = should_retrain(
            feature_drift=feature_drift,
            prediction_drift_score=0.2,
            prediction_drift_threshold=0.15
        )
        
        assert result["retrain"] == True
        assert result["prediction_drift"] == True

    def test_should_retrain_no_drift(self):
        """Test retraining decision when no drift detected."""
        feature_drift = {"f1": 0.05, "f2": 0.08}
        result = should_retrain(
            feature_drift=feature_drift,
            prediction_drift_score=0.05,
            psi_threshold=0.25,
            prediction_drift_threshold=0.15
        )
        
        assert result["retrain"] == False
        assert result["feature_drift"] == False
        assert result["prediction_drift"] == False

    def test_psi_handles_nan_values(self):
        """Test PSI handles NaN values gracefully."""
        expected = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        actual = pd.Series([1.5, 2.5, 3.5, np.nan, 5.5])
        
        psi = calculate_psi(expected, actual)
        assert isinstance(psi, float)
        assert not np.isnan(psi)
