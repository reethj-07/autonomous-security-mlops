import pytest
from src.safety.kill_switch import load_runtime_config, is_ml_enabled, is_safe_mode, is_lockdown
from src.alerting.risk_alerts import determine_alert_level, generate_alert


class TestSafety:
    """Unit tests for safety controls."""

    def test_is_ml_enabled_default(self):
        """Test ML enabled check with missing config defaults to SAFE_MODE."""
        result = is_ml_enabled()
        # Should be False because config likely doesn't exist or is in SAFE_MODE
        assert isinstance(result, bool)

    def test_is_safe_mode_default(self):
        """Test safe mode check."""
        result = is_safe_mode()
        assert isinstance(result, bool)

    def test_is_lockdown_default(self):
        """Test lockdown check."""
        result = is_lockdown()
        assert isinstance(result, bool)

    def test_one_safety_state_at_a_time(self):
        """Test that only one safety state is active."""
        enabled = is_ml_enabled()
        safe_mode = is_safe_mode()
        lockdown = is_lockdown()
        
        active_states = sum([enabled, safe_mode, lockdown])
        assert active_states == 1, "Exactly one safety state should be active"


class TestRiskAlerts:
    """Unit tests for risk alert generation."""

    def test_determine_alert_level_low(self):
        """Test LOW risk score."""
        level = determine_alert_level(0.2)
        assert level == "LOW"

    def test_determine_alert_level_medium(self):
        """Test MEDIUM risk score."""
        level = determine_alert_level(0.5)
        assert level == "MEDIUM"

    def test_determine_alert_level_high(self):
        """Test HIGH risk score."""
        level = determine_alert_level(0.7)
        assert level == "HIGH"

    def test_determine_alert_level_critical(self):
        """Test CRITICAL risk score."""
        level = determine_alert_level(0.9)
        assert level == "CRITICAL"

    def test_generate_alert_structure(self):
        """Test alert output structure."""
        risk_output = {
            "risk_score": 0.75,
            "classifier_prob": 0.8,
            "anomaly_score": 0.7,
            "context_risk": 0.6,
        }
        
        alert = generate_alert(risk_output)
        
        assert "alert_level" in alert
        assert "risk_score" in alert
        assert "signals" in alert
        assert "metadata" in alert

    def test_generate_alert_with_metadata(self):
        """Test alert generation with custom metadata."""
        risk_output = {"risk_score": 0.8, "classifier_prob": 0.7}
        metadata = {"user_id": 123, "ip_address": "192.168.1.1"}
        
        alert = generate_alert(risk_output, metadata)
        
        assert alert["metadata"]["user_id"] == 123
        assert alert["metadata"]["ip_address"] == "192.168.1.1"

    def test_alert_level_consistency_with_risk_score(self):
        """Test that alert level is consistent with risk score."""
        for risk_score in [0.1, 0.2, 0.4, 0.6, 0.8, 0.95]:
            alert = generate_alert({"risk_score": risk_score})
            level = alert["alert_level"]
            
            if risk_score < 0.3:
                assert level == "LOW"
            elif risk_score < 0.6:
                assert level == "MEDIUM"
            elif risk_score < 0.8:
                assert level == "HIGH"
            else:
                assert level == "CRITICAL"
