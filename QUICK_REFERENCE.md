# Quick Reference Card - Security MLOps Platform

## 🚀 Start Here

### Local Development (3 commands)
```bash
git clone <repo>
cd autonomous-security-mlops
docker-compose up -d
```

### Access Services
- **API**: http://localhost:8000
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/admin)
- **Airflow**: http://localhost:8080

---

## 📚 Core Modules

### 1. Data Validation
```python
from src.validation.data_schemas import SecurityLogInput, FeatureRow
from src.validation.data_quality import SecurityLogDataValidator

# Validate input
input_data = SecurityLogInput(**log_entry)

# Validate features
validator = SecurityLogDataValidator()
is_valid, report = validator.validate_features(df)
```

### 2. Feature Engineering
```python
from src.features.advanced_engineering import AdvancedFeatureEngineer

engineer = AdvancedFeatureEngineer()
df = engineer.generate_comprehensive_features(df)

# Get best features
X_selected = engineer.select_best_features(X, y, n_features=20)
```

### 3. Model Training
```python
from src.models.ensemble import HybridEnsembleSelector

selector = HybridEnsembleSelector()
results = selector.train_all_models(X_train, y_train, X_val, y_val)
best_name, best_model = selector.get_best_model(metric='f1')
```

### 4. Monitoring & Drift
```python
from src.monitoring.advanced_monitoring import (
    AdvancedDriftDetector, SHAPExplainer, RealTimeFeatureMonitor
)

# Drift detection
detector = AdvancedDriftDetector()
drift_report = detector.generate_data_drift_report(current_data, reference_data)

# SHAP explainability
explainer = SHAPExplainer(model, feature_names)
importance = explainer.get_global_feature_importance(X_test)
```

### 5. A/B Testing
```python
from src.experimentation.ab_testing import ABTestExperiment, ExperimentConfig

config = ExperimentConfig(
    experiment_id="exp_001",
    variant_a_name="Production",
    variant_b_name="Staging",
    metric="f1"
)

exp = ABTestExperiment(config)
results = exp.run_experiment(X_a, model_a, X_b, model_b, y_true)
```

---

## 🔧 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "request": "GET /admin",
    "user_id": "user_123",
    "path": "/admin",
    "method": "GET",
    "status_code": 200,
    "latency_ms": 45
  }'
```

### Metrics
```bash
curl http://localhost:8000/metrics
```

---

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f inference_service

# Stop services
docker-compose down

# Restart service
docker-compose restart inference_service

# Build image
docker build -t security-inference:latest ./inference_service
```

---

## ☸️ Kubernetes Commands

```bash
# Deploy
kubectl apply -f kubernetes/deployment.yaml

# Check status
kubectl get pods -n security-mlops
kubectl describe pod <pod-name> -n security-mlops

# Port forward
kubectl port-forward svc/inference-service 8000:80 -n security-mlops

# View logs
kubectl logs deployment/inference-service -n security-mlops -f

# Rollback
kubectl rollout undo deployment/inference-service -n security-mlops
```

---

## 📊 Metrics & Monitoring

### Prometheus Queries
```promql
# Prediction latency
rate(inference_latency_ms[5m])

# Error rate
rate(inference_errors_total[5m])

# Model accuracy
security_model_accuracy

# Drift detection
drift_psi_score
```

### Grafana Dashboards
- Model Performance
- Inference Metrics
- Drift Monitoring
- System Health

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_inference.py::test_predict -v
```

---

## 📁 File Structure

```
src/
├── validation/
│   ├── data_schemas.py          # Pydantic validators
│   └── data_quality.py          # Great Expectations
├── models/
│   └── ensemble.py              # XGB, LGB, Stacking
├── features/
│   └── advanced_engineering.py  # 50+ features
├── monitoring/
│   └── advanced_monitoring.py   # Drift, SHAP, monitoring
└── experimentation/
    └── ab_testing.py            # A/B testing framework

kubernetes/
└── deployment.yaml              # K8s manifests

docker-compose.yml               # Local stack

ADVANCED_DOCUMENTATION.md        # Full guide
DEPLOYMENT_RUNBOOK.md            # Operations guide
IMPLEMENTATION_SUMMARY.md        # What was built
```

---

## 🎯 Common Tasks

### Deploy New Model
```bash
# Register in MLflow
mlflow models register --model-uri runs:/<run-id>/model --name security-anomaly

# Promote to Production
mlflow models update-model-version \
  --name security-anomaly \
  --version 1 \
  --model-stage Production

# Deploy to K8s
kubectl set image deployment/inference-service \
  inference=ghcr.io/reethj-07/security-inference:v1.2.3
```

### Check Model Performance
```bash
mlflow runs list --experiment-name security-log-detection
mlflow models list
mlflow models get-latest-versions --name security-anomaly
```

### View Drift Metrics
```bash
# Via Prometheus
curl 'http://localhost:9090/api/v1/query?query=drift_psi_score'

# Via logs
kubectl logs deployment/inference-service -n security-mlops | grep drift
```

### Run Feature Selection
```python
from src.features.advanced_engineering import AdvancedFeatureEngineer

engineer = AdvancedFeatureEngineer()
X_best = engineer.select_best_features(X, y, n_features=20, method='mutual_info')
```

---

## 🚨 Troubleshooting

### Pod not starting
```bash
kubectl describe pod <pod-name> -n security-mlops
kubectl logs <pod-name> -n security-mlops
```

### Model not loading
```bash
# Check MLflow
curl http://mlflow:5000/api/2.0/health

# Check model registry
mlflow models list
```

### High latency
```bash
kubectl top pod -n security-mlops
kubectl logs deployment/inference-service -n security-mlops | grep latency
```

---

## 📞 Key Files

| File | Purpose |
|------|---------|
| `ADVANCED_DOCUMENTATION.md` | Complete system guide |
| `DEPLOYMENT_RUNBOOK.md` | Operations procedures |
| `IMPLEMENTATION_SUMMARY.md` | What was implemented |
| `src/validation/data_schemas.py` | Input validation |
| `src/models/ensemble.py` | ML models |
| `src/monitoring/advanced_monitoring.py` | Monitoring |
| `docker-compose.yml` | Local development |
| `kubernetes/deployment.yaml` | Production deployment |

---

## ⚡ Performance Targets

- **API Latency**: <100ms p95
- **Model Accuracy**: >90% F1 score
- **Uptime**: 99.9%
- **Drift Detection**: <1 hour detection lag
- **Data Validation**: 100% coverage

---

## 🔐 Security

- ✅ API key authentication
- ✅ Rate limiting (100 req/min)
- ✅ Non-root containers
- ✅ Secrets management
- ✅ Network policies
- ✅ Resource limits

---

**Last Updated**: 2026-02-01
**Status**: ✅ Production Ready
