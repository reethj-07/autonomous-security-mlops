import pytest
import pandas as pd
import numpy as np
from src.scoring.hybrid_risk import compute_hybrid_risk, compute_context_risk, normalize_scores


class TestHybridRiskScoring:
    """Unit tests for hybrid risk scoring engine."""

    def test_normalize_scores_range(self):
        """Test scores are normalized to [0, 1]."""
        scores = np.array([10, 20, 30, 40, 50])
        normalized = normalize_scores(scores)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert len(normalized) == len(scores)

    def test_normalize_scores_identical(self):
        """Test normalization of identical scores returns zeros."""
        scores = np.array([5.0, 5.0, 5.0, 5.0])
        normalized = normalize_scores(scores)
        
        assert np.allclose(normalized, np.zeros_like(normalized))

    def test_compute_context_risk_admin_path(self):
        """Test context risk increases for admin paths."""
        features_normal = {"is_admin_path": 0}
        features_admin = {"is_admin_path": 1}
        
        risk_normal = compute_context_risk(features_normal)
        risk_admin = compute_context_risk(features_admin)
        
        assert risk_admin > risk_normal

    def test_compute_context_risk_sql_injection(self):
        """Test context risk increases for SQL keywords."""
        features_clean = {"has_sql_keywords": 0}
        features_sql = {"has_sql_keywords": 1}
        
        risk_clean = compute_context_risk(features_clean)
        risk_sql = compute_context_risk(features_sql)
        
        assert risk_sql > risk_clean

    def test_compute_context_risk_bounds(self):
        """Test context risk is bounded to [0, 1]."""
        features = {
            "is_admin_path": 1,
            "has_sql_keywords": 1,
            "request_length": 500
        }
        risk = compute_context_risk(features)
        
        assert 0 <= risk <= 1.0

    def test_compute_hybrid_risk_output_structure(self):
        """Test hybrid risk output has required keys."""
        result = compute_hybrid_risk(
            classifier_prob=0.7,
            anomaly_score=0.5,
            anomaly_score_norm=0.6,
            context_features={"is_admin_path": 1}
        )
        
        required_keys = ["risk_score", "classifier_prob", "anomaly_score", "context_risk"]
        for key in required_keys:
            assert key in result

    def test_compute_hybrid_risk_custom_weights(self):
        """Test hybrid risk with custom weights."""
        weights = {
            "classifier": 0.8,
            "anomaly": 0.15,
            "context": 0.05,
        }
        
        result = compute_hybrid_risk(
            classifier_prob=0.9,
            anomaly_score=0.3,
            anomaly_score_norm=0.4,
            context_features={},
            weights=weights
        )
        
        # Risk should be dominated by classifier (0.8 * 0.9 = 0.72)
        assert result["risk_score"] > 0.7

    def test_compute_hybrid_risk_default_weights(self):
        """Test hybrid risk uses sensible defaults."""
        result = compute_hybrid_risk(
            classifier_prob=0.5,
            anomaly_score=0.5,
            anomaly_score_norm=0.5,
            context_features={"is_admin_path": 0}
        )
        
        # With equal scores and equal weights (0.5, 0.4, 0.1), should be ~0.48
        assert 0.4 < result["risk_score"] < 0.6

    def test_hybrid_risk_all_zeros(self):
        """Test hybrid risk with all zero inputs."""
        result = compute_hybrid_risk(
            classifier_prob=0.0,
            anomaly_score=0.0,
            anomaly_score_norm=0.0,
            context_features={}
        )
        
        assert result["risk_score"] == 0.0

    def test_hybrid_risk_all_ones(self):
        """Test hybrid risk with maximum inputs."""
        result = compute_hybrid_risk(
            classifier_prob=1.0,
            anomaly_score=1.0,
            anomaly_score_norm=1.0,
            context_features={"is_admin_path": 1, "has_sql_keywords": 1}
        )
        
        # Should be close to 1.0
        assert result["risk_score"] > 0.9
