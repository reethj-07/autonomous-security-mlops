"""
Pydantic schemas for data validation across the pipeline.
Ensures type safety and data quality at runtime.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np


# ============================================================
# INPUT VALIDATION SCHEMAS
# ============================================================

class SecurityLogInput(BaseModel):
    """Validates raw security log entries"""
    
    timestamp: datetime
    user_id: str
    ip_address: str
    path: str
    method: str = Field(..., regex="^(GET|POST|PUT|DELETE|PATCH)$")
    request: str
    status_code: int = Field(..., ge=100, le=599)
    latency_ms: int = Field(..., ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-01-01T12:00:00",
                "user_id": "user_123",
                "ip_address": "192.168.1.1",
                "path": "/api/users",
                "method": "GET",
                "request": "GET /api/users",
                "status_code": 200,
                "latency_ms": 45
            }
        }


class FeatureRow(BaseModel):
    """Validates engineered feature rows"""
    
    request_length: int = Field(..., ge=0)
    has_sql_keywords: int = Field(..., ge=0, le=1)
    is_admin_path: int = Field(..., ge=0, le=1)
    path_rarity: float = Field(..., ge=0, le=1)
    method_entropy: float = Field(..., ge=0, le=10)
    sql_keyword_rarity: float = Field(..., ge=0, le=1)
    path_transition_risk: float = Field(..., ge=0, le=1)
    repeated_request_count: int = Field(..., ge=0)
    method_transition_flag: int = Field(..., ge=0, le=1)
    label: Optional[int] = Field(None, ge=0, le=1)
    
    @validator("request_length", "path_rarity", "method_entropy", "sql_keyword_rarity", 
               "path_transition_risk", pre=True)
    def validate_no_inf_nan(cls, v):
        if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
            raise ValueError("Values cannot be NaN or Inf")
        return v


# ============================================================
# PREDICTION OUTPUT SCHEMAS
# ============================================================

class AnomalyPrediction(BaseModel):
    """Validates anomaly detection output"""
    
    anomaly_score: float = Field(..., ge=0, le=1)
    is_anomaly: int = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "anomaly_score": 0.85,
                "is_anomaly": 1,
                "confidence": 0.92
            }
        }


class ClassifierPrediction(BaseModel):
    """Validates classifier output"""
    
    probability: float = Field(..., ge=0, le=1)
    predicted_label: int = Field(..., ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "probability": 0.78,
                "predicted_label": 1
            }
        }


class HybridRiskScore(BaseModel):
    """Validates hybrid risk computation output"""
    
    risk_score: float = Field(..., ge=0, le=1)
    classifier_prob: float = Field(..., ge=0, le=1)
    anomaly_score: float = Field(..., ge=0, le=1)
    context_risk: float = Field(..., ge=0, le=1)
    
    @validator("risk_score", "classifier_prob", "anomaly_score", "context_risk")
    def validate_probabilities(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Score must be between 0 and 1")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 0.72,
                "classifier_prob": 0.65,
                "anomaly_score": 0.81,
                "context_risk": 0.45
            }
        }


class PredictionResponse(BaseModel):
    """Complete prediction response schema"""
    
    request_id: str
    timestamp: datetime
    features: Dict[str, Any]
    anomaly_pred: AnomalyPrediction
    classifier_pred: ClassifierPrediction
    hybrid_risk: HybridRiskScore
    alert_level: str = Field(..., regex="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    model_version: str
    model_stage: str = Field(..., regex="^(Production|Staging|None)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_12345",
                "timestamp": "2025-01-01T12:00:00",
                "features": {"request_length": 150},
                "anomaly_pred": {"anomaly_score": 0.85, "is_anomaly": 1, "confidence": 0.92},
                "classifier_pred": {"probability": 0.78, "predicted_label": 1},
                "hybrid_risk": {"risk_score": 0.72, "classifier_prob": 0.65, "anomaly_score": 0.81, "context_risk": 0.45},
                "alert_level": "HIGH",
                "model_version": "v1.2.3",
                "model_stage": "Production"
            }
        }


# ============================================================
# MONITORING & DRIFT SCHEMAS
# ============================================================

class DriftMetrics(BaseModel):
    """Validates drift calculation outputs"""
    
    feature_name: str
    psi_score: float = Field(..., ge=0, le=10)
    is_drift_detected: bool
    threshold: float = Field(..., ge=0, le=1)
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature_name": "request_length",
                "psi_score": 0.35,
                "is_drift_detected": True,
                "threshold": 0.25,
                "timestamp": "2025-01-01T12:00:00"
            }
        }


class CanaryMetrics(BaseModel):
    """Validates canary deployment metrics"""
    
    alerts_per_min: float = Field(..., ge=0)
    prediction_entropy: float = Field(..., ge=0, le=10)
    p95_latency_ms: int = Field(..., ge=0)
    model_stage: str
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "alerts_per_min": 5.2,
                "prediction_entropy": 0.8,
                "p95_latency_ms": 120,
                "model_stage": "Staging",
                "timestamp": "2025-01-01T12:00:00"
            }
        }


class RetrainingDecision(BaseModel):
    """Validates retraining decision logic"""
    
    retrain: bool
    feature_drift: bool
    prediction_drift: bool
    psi_threshold: float
    prediction_drift_threshold: float
    timestamp: datetime
    reason: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "retrain": True,
                "feature_drift": True,
                "prediction_drift": False,
                "psi_threshold": 0.25,
                "prediction_drift_threshold": 0.15,
                "timestamp": "2025-01-01T12:00:00",
                "reason": "Feature drift detected in request_length"
            }
        }


# ============================================================
# FEEDBACK & EXPLAINABILITY SCHEMAS
# ============================================================

class AnalystFeedback(BaseModel):
    """Validates analyst feedback on predictions"""
    
    prediction_id: str
    analyst_id: str
    feedback_type: str = Field(..., regex="^(correct|incorrect|uncertain)$")
    ground_truth_label: Optional[int] = Field(None, ge=0, le=1)
    notes: Optional[str] = Field(None, max_length=500)
    confidence_in_feedback: int = Field(..., ge=0, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "pred_12345",
                "analyst_id": "analyst_007",
                "feedback_type": "incorrect",
                "ground_truth_label": 0,
                "notes": "False positive, legitimate admin access",
                "confidence_in_feedback": 95
            }
        }


class ExplanationOutput(BaseModel):
    """Validates LLM-generated explanations"""
    
    incident_id: str
    timestamp: datetime
    model_version: str
    environment: str
    explanation_text: str = Field(..., min_length=50, max_length=5000)
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    contributing_features: Optional[List[Dict[str, float]]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "incident_id": "inc_12345",
                "timestamp": "2025-01-01T12:00:00",
                "model_version": "v1.2.3",
                "environment": "production",
                "explanation_text": "High risk detected due to SQL injection keywords and admin path access...",
                "confidence_score": 0.87,
                "contributing_features": [
                    {"has_sql_keywords": 0.45},
                    {"is_admin_path": 0.35}
                ]
            }
        }


# ============================================================
# TRAINING & EVALUATION SCHEMAS
# ============================================================

class ModelMetrics(BaseModel):
    """Validates model training metrics"""
    
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    roc_auc: Optional[float] = Field(None, ge=0, le=1)
    confusion_matrix: List[List[int]]
    threshold: float = Field(..., ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "precision": 0.89,
                "recall": 0.84,
                "f1_score": 0.86,
                "roc_auc": 0.92,
                "confusion_matrix": [[950, 50], [40, 60]],
                "threshold": 0.5
            }
        }


class TrainingRun(BaseModel):
    """Validates training run metadata"""
    
    experiment_name: str
    run_id: str
    model_type: str
    dataset_size: int = Field(..., ge=1)
    features_used: List[str]
    metrics: ModelMetrics
    git_commit: Optional[str] = None
    artifacts_path: str
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "experiment_name": "security-log-detection",
                "run_id": "run_abc123",
                "model_type": "IsolationForest",
                "dataset_size": 10000,
                "features_used": ["request_length", "has_sql_keywords"],
                "metrics": {
                    "precision": 0.89,
                    "recall": 0.84,
                    "f1_score": 0.86,
                    "roc_auc": 0.92,
                    "confusion_matrix": [[950, 50], [40, 60]],
                    "threshold": 0.5
                },
                "git_commit": "abc123def456",
                "artifacts_path": "s3://bucket/artifacts/run_abc123",
                "timestamp": "2025-01-01T12:00:00"
            }
        }


# ============================================================
# HEALTH CHECK SCHEMAS
# ============================================================

class HealthCheckResponse(BaseModel):
    """Validates service health status"""
    
    status: str = Field(..., regex="^(healthy|degraded|unhealthy)$")
    timestamp: datetime
    uptime_seconds: int = Field(..., ge=0)
    model_stage: str
    model_loaded: bool
    database_connected: bool
    error_rate_percent: float = Field(..., ge=0, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-01T12:00:00",
                "uptime_seconds": 3600,
                "model_stage": "Production",
                "model_loaded": True,
                "database_connected": True,
                "error_rate_percent": 0.5
            }
        }
