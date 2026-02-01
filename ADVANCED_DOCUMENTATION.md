# Security MLOps Platform - Complete Documentation

## 🚀 Quick Start

### Local Development (Docker Compose)

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# Access services:
# - Inference API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Grafana: http://localhost:3000 (admin/admin)
# - Airflow: http://localhost:8080
```

### Production Deployment (Kubernetes)

```bash
# Create namespace and deploy
kubectl apply -f kubernetes/deployment.yaml

# Check deployment status
kubectl get pods -n security-mlops

# Access inference service
kubectl port-forward -n security-mlops svc/inference-service 8000:80
```

---

## 📋 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              Data Ingestion Layer                        │
│  - Security Logs  - API Streams  - DVC Data Versioning  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│         Feature Engineering & Validation                │
│  - Advanced temporal features  - SHAP features          │
│  - Attack pattern detection    - Pydantic validation    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│           Training & Model Ensembles                    │
│  - XGBoost/LightGBM  - Stacking  - Voting ensemble      │
│  - MLflow tracking   - Model Registry                   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│    Canary Deployment & Safety Evaluation                │
│  - Prod/Staging fallback  - Alert rate monitoring       │
│  - Entropy guards         - Auto-rollback               │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│         Inference & Risk Scoring Service                │
│  - FastAPI endpoint   - Hybrid risk scoring             │
│  - Rate limiting      - Health checks                   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│      Monitoring & Explainability                        │
│  - Drift detection (PSI)  - SHAP explanations           │
│  - Evidently AI dashboards  - Feedback loops            │
│  - Real-time feature monitoring                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Inference Service (.env)
MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_STAGE=Production              # Production/Staging
DATABASE_URL=postgresql://user:pass@postgres:5432/security_mlops
API_KEY=your-secret-key
RATE_LIMIT=100/minute
LOG_LEVEL=INFO
```

### Runtime Safety Config

```yaml
# configs/runtime_config.yaml
system_state: ENABLED              # ENABLED/SAFE_MODE/LOCKDOWN
kill_switch_enabled: true
alert_threshold: 0.7
max_model_latency_ms: 500
```

---

## 📊 Data Pipeline

### 1. Feature Engineering

```python
from src.features.advanced_engineering import AdvancedFeatureEngineer

engineer = AdvancedFeatureEngineer()
df = engineer.generate_comprehensive_features(df)
# Features: temporal, behavioral, attack patterns, statistical
```

### 2. Data Validation

```python
from src.validation.data_schemas import SecurityLogInput, FeatureRow
from src.validation.data_quality import SecurityLogDataValidator

# Schema validation
input_data = SecurityLogInput(**log_entry)

# Great Expectations validation
validator = SecurityLogDataValidator()
is_valid, report = validator.validate_features(df)
```

### 3. Model Training

```python
from src.models.ensemble import HybridEnsembleSelector

# Train all models
selector = HybridEnsembleSelector()
results = selector.train_all_models(X_train, y_train, X_val, y_val)

# Get best model
best_name, best_model = selector.get_best_model(metric='f1')
```

---

## 🤖 Advanced Models

### Ensemble Architectures

#### XGBoost
- Fast gradient boosting with early stopping
- Feature importance tracking
- Best for: Initial model baseline

```python
from src.models.ensemble import XGBoostSecurityModel

xgb = XGBoostSecurityModel()
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
importance = xgb.get_feature_importance()
```

#### LightGBM
- Memory-efficient, handles categorical features
- Faster training than XGBoost
- Best for: Large-scale deployments

```python
from src.models.ensemble import LightGBMSecurityModel

lgb = LightGBMSecurityModel()
lgb.fit(X_train, y_train)
```

#### Stacking Ensemble
- Combines 3 base learners (XGB, LGB, CatBoost)
- Meta-learner: Logistic Regression
- Best for: Maximum accuracy

```python
from src.models.ensemble import StackingEnsembleModel

stacking = StackingEnsembleModel()
stacking.fit(X_train, y_train)
weights = stacking.get_base_learner_weights()
```

---

## 🔍 Monitoring & Drift Detection

### Evidently AI Reports

```python
from src.monitoring.advanced_monitoring import AdvancedDriftDetector

detector = AdvancedDriftDetector()
drift_report = detector.generate_data_drift_report(current_data, reference_data)
quality_report = detector.generate_data_quality_report(current_data)
```

### SHAP Explainability

```python
from src.monitoring.advanced_monitoring import SHAPExplainer

explainer = SHAPExplainer(model, feature_names)
explainer.initialize_explainer(X_background)

# Global importance
importance_df = explainer.get_global_feature_importance(X_test)

# Instance explanation
explanation = explainer.explain_instance(X_test, instance_idx=0)
```

### Concept Drift Detection

```python
drift_results = detector.detect_concept_drift(y_true, y_pred, window_size=100)
if drift_results["concept_drift_detected"]:
    logger.warning(f"Retraining triggered: {drift_results}")
```

---

## ⚗️ A/B Testing

### Setup Experiment

```python
from src.experimentation.ab_testing import ABTestExperiment, ExperimentConfig

config = ExperimentConfig(
    experiment_id="exp_001",
    variant_a_name="Production",
    variant_b_name="Staging",
    metric="f1",
    alpha=0.05,
    expected_effect_size=0.05
)

experiment = ABTestExperiment(config)
results = experiment.run_experiment(X_a, model_a, X_b, model_b, y_true)
```

### Statistical Tests

- **Two-Proportion Z-Test**: For precision/recall metrics
- **Welch's T-Test**: For F1, AUC metrics
- **Mann-Whitney U Test**: Non-parametric alternative

### Results Interpretation

```
p-value < 0.05: Statistically significant difference
Effect size: Magnitude of improvement
Confidence Interval: Range of expected effect
```

---

## 📈 Inference API

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "model_stage": "Production",
  "model_loaded": true,
  "error_rate_percent": 0.5
}
```

### Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "request": "GET /admin",
    "user_id": "user_123",
    "path": "/admin",
    "method": "GET",
    "status_code": 200,
    "latency_ms": 45
  }'
```

Response:
```json
{
  "request_id": "req_abc123",
  "alert_level": "HIGH",
  "risk_score": 0.82,
  "model_version": "v1.2.3",
  "model_stage": "Production"
}
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

Prometheus-compatible metrics for monitoring

---

## 🚀 Deployment Pipeline (GitHub Actions)

### 1. Training CI

```
- Trigger: Push to main
- Steps:
  1. Load data (DVC)
  2. Run feature engineering
  3. Train ensemble models
  4. Log to MLflow
  5. Promote best model
  6. Update Model Registry
```

### 2. Inference CI

```
- Trigger: Model promotion
- Steps:
  1. Validate inference imports
  2. Load promoted model
  3. Run smoke tests
  4. Docker build
  5. Push to GHCR
  6. Deploy to Kubernetes
```

### 3. Monitoring DAG (Airflow)

```
- Schedule: Every 6 hours
- Tasks:
  1. Compute drift metrics (PSI)
  2. Evaluate canary model
  3. Decide retraining
  4. Trigger retraining if needed
```

---

## 🔐 Security Best Practices

### API Security
- ✅ API key authentication
- ✅ Rate limiting (100 req/min)
- ✅ Input validation (Pydantic schemas)
- ✅ CORS protection

### Docker Security
- ✅ Non-root user execution
- ✅ Read-only filesystem
- ✅ No privileged containers
- ✅ Health checks

### Kubernetes Security
- ✅ Network policies
- ✅ Resource quotas
- ✅ Pod security policies
- ✅ Secret management

---

## 📊 Observability

### Grafana Dashboards

Pre-configured dashboards:
1. **Model Performance**: Precision, Recall, F1, AUC
2. **Inference Metrics**: Latency, Throughput, Errors
3. **Drift Monitoring**: PSI scores, Feature statistics
4. **System Health**: CPU, Memory, Pod count

### Prometheus Metrics

```
# Model metrics
security_model_predictions_total
security_model_accuracy
security_model_prediction_latency_ms

# Inference service
inference_requests_total
inference_errors_total
inference_latency_ms

# Drift detection
drift_psi_score
drift_detected_count
```

### Alerting Rules

```yaml
# Prometheus rules
- alert: ModelAccuracyDegradation
  expr: rate(security_model_accuracy[1h]) < 0.8
  
- alert: HighErrorRate
  expr: rate(inference_errors_total[5m]) > 0.05

- alert: DriftDetected
  expr: drift_psi_score > 0.25
```

---

## 🔄 CI/CD Workflows

### Training Pipeline
```
Data → Feature Engineering → Model Training → 
MLflow Logging → Model Promotion → Registry Update
```

### Deployment Pipeline
```
Model Promotion → Build Docker Image → 
Push to GHCR → Deploy to K8s → Health Checks
```

### Monitoring Pipeline
```
Collect Metrics → Compute Drift → 
Evaluate Canary → Decision → Retrain (optional)
```

---

## 🐛 Troubleshooting

### Model Not Loading
```bash
# Check MLflow connectivity
curl http://mlflow:5000/api/2.0/health

# Check model in registry
mlflow models ls

# View model stage
mlflow models get-latest-versions --name security-anomaly
```

### High Latency
```bash
# Check model complexity
# Reduce input features or batch size
# Consider model quantization
```

### Drift Detected
```bash
# Check PSI scores
# Review feature distributions
# Trigger retraining if PSI > 0.25
```

---

## 📚 References

- [MLflow Documentation](https://mlflow.org/docs)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Evidently AI](https://evidentlyai.com/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/architecture/nodes/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

---

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/improvement`
2. Add tests: `pytest tests/`
3. Update documentation
4. Submit PR with description

---

## 📄 License

MIT License - See LICENSE file
