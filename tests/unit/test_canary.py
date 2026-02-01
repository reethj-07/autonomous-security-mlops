import pytest
from src.deployment.canary_evaluator import evaluate_canary


class TestCanaryEvaluator:
    """Unit tests for canary deployment evaluation."""

    def test_canary_promote_decision(self):
        """Test PROMOTE decision when metrics are good."""
        canary_metrics = {
            "alerts_per_min": 1.0,
            "prediction_entropy": 0.5,
            "p95_latency_ms": 100.0,
        }
        baseline_metrics = {
            "alerts_per_min": 1.0,
            "prediction_entropy": 0.5,
            "p95_latency_ms": 100.0,
        }
        
        result = evaluate_canary(canary_metrics, baseline_metrics)
        assert result["decision"] == "PROMOTE"

    def test_canary_rollback_alert_spike(self):
        """Test ROLLBACK when alert rate spikes."""
        canary_metrics = {
            "alerts_per_min": 3.0,  # 3x baseline
            "prediction_entropy": 0.5,
            "p95_latency_ms": 100.0,
        }
        baseline_metrics = {
            "alerts_per_min": 1.0,
            "prediction_entropy": 0.5,
            "p95_latency_ms": 100.0,
        }
        
        result = evaluate_canary(canary_metrics, baseline_metrics)
        assert result["decision"] == "ROLLBACK"
        assert "Alert rate spike detected" in result["reasons"]

    def test_canary_rollback_entropy_increase(self):
        """Test ROLLBACK when prediction entropy increases."""
        canary_metrics = {
            "alerts_per_min": 1.0,
            "prediction_entropy": 0.7,  # +0.2 above baseline
            "p95_latency_ms": 100.0,
        }
        baseline_metrics = {
            "alerts_per_min": 1.0,
            "prediction_entropy": 0.5,
            "p95_latency_ms": 100.0,
        }
        
        result = evaluate_canary(canary_metrics, baseline_metrics)
        assert result["decision"] == "ROLLBACK"
        assert "Prediction entropy increased" in result["reasons"]

    def test_canary_rollback_latency_regression(self):
        """Test ROLLBACK when latency regresses."""
        canary_metrics = {
            "alerts_per_min": 1.0,
            "prediction_entropy": 0.5,
            "p95_latency_ms": 200.0,  # 2x baseline
        }
        baseline_metrics = {
            "alerts_per_min": 1.0,
            "prediction_entropy": 0.5,
            "p95_latency_ms": 100.0,
        }
        
        result = evaluate_canary(canary_metrics, baseline_metrics)
        assert result["decision"] == "ROLLBACK"
        assert "Latency regression detected" in result["reasons"]

    def test_canary_extend_missing_metrics(self):
        """Test EXTEND when metrics are missing."""
        canary_metrics = {
            "alerts_per_min": 1.0,
            # Missing prediction_entropy and p95_latency_ms
        }
        baseline_metrics = {
            "alerts_per_min": 1.0,
            "prediction_entropy": 0.5,
            "p95_latency_ms": 100.0,
        }
        
        result = evaluate_canary(canary_metrics, baseline_metrics)
        assert result["decision"] == "EXTEND"
        assert any("Missing metric" in r for r in result["reasons"])

    def test_canary_multiple_violations(self):
        """Test ROLLBACK with multiple guardrail violations."""
        canary_metrics = {
            "alerts_per_min": 3.0,  # Violation
            "prediction_entropy": 0.7,  # Violation
            "p95_latency_ms": 100.0,
        }
        baseline_metrics = {
            "alerts_per_min": 1.0,
            "prediction_entropy": 0.5,
            "p95_latency_ms": 100.0,
        }
        
        result = evaluate_canary(canary_metrics, baseline_metrics)
        assert result["decision"] == "ROLLBACK"
        assert len(result["reasons"]) >= 2

    def test_canary_timestamp_in_result(self):
        """Test that result includes timestamp."""
        result = evaluate_canary(
            {"alerts_per_min": 1.0, "prediction_entropy": 0.5, "p95_latency_ms": 100.0},
            {"alerts_per_min": 1.0, "prediction_entropy": 0.5, "p95_latency_ms": 100.0}
        )
        
        assert "timestamp" in result
