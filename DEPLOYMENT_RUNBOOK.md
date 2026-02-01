# Deployment Runbook

## 🚀 Pre-Deployment Checklist

- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Code reviewed and merged to main
- [ ] Documentation updated
- [ ] No breaking changes to API
- [ ] Model performance validated
- [ ] Infrastructure capacity verified
- [ ] Backup of current production model taken

---

## 📋 Local Development Setup

### Prerequisites
```bash
# Install Docker & Docker Compose
docker --version  # >= 24.0
docker-compose --version  # >= 2.0

# Install Python dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
```

### Start Development Environment

```bash
# Start all services
docker-compose up -d

# Initialize MLflow DB
docker-compose exec mlflow mlflow db upgrade sqlite:///mlflow.db

# Check services
docker-compose ps

# View logs
docker-compose logs -f inference_service
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_inference.py -v

# Run integration tests
pytest tests/integration/ -v
```

---

## 🐳 Docker Build & Push

### Build Image

```bash
# Build inference service image
cd inference_service
docker build -t security-inference:latest .

# Tag for registry
docker tag security-inference:latest ghcr.io/reethj-07/security-inference:latest

# Build with build args
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg MODEL_PATH=s3://bucket/model.pkl \
  -t security-inference:v1.2.3 .
```

### Push to Registry

```bash
# Authenticate with GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u reethj-07 --password-stdin

# Push image
docker push ghcr.io/reethj-07/security-inference:latest

# Push with version tag
docker push ghcr.io/reethj-07/security-inference:v1.2.3

# Verify image
docker run --rm ghcr.io/reethj-07/security-inference:latest --version
```

---

## ☸️ Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
kubectl version --client

# Create namespace
kubectl create namespace security-mlops

# Verify cluster access
kubectl cluster-info
```

### Deploy to Kubernetes

```bash
# Apply configuration
kubectl apply -f kubernetes/deployment.yaml

# Check rollout status
kubectl rollout status deployment/inference-service -n security-mlops

# Verify all pods running
kubectl get pods -n security-mlops

# Check service
kubectl get svc -n security-mlops
```

### Port Forwarding

```bash
# Forward inference service
kubectl port-forward -n security-mlops svc/inference-service 8000:80

# Forward MLflow
kubectl port-forward -n security-mlops svc/mlflow 5000:5000

# Forward Prometheus
kubectl port-forward -n security-mlops svc/prometheus 9090:9090
```

### Update Deployment

```bash
# Update image
kubectl set image deployment/inference-service \
  inference=ghcr.io/reethj-07/security-inference:v1.2.3 \
  -n security-mlops

# Watch rollout
kubectl rollout status deployment/inference-service -n security-mlops

# Rollback if needed
kubectl rollout undo deployment/inference-service -n security-mlops
```

---

## 🔄 Rolling Update Strategy

### Blue-Green Deployment

```bash
# Deploy new version alongside old
kubectl set image deployment/inference-service-blue \
  inference=ghcr.io/reethj-07/security-inference:v1.2.3 \
  -n security-mlops

# Route traffic gradually
kubectl patch service inference-service -p \
  '{"spec":{"selector":{"version":"v1.2.3"}}}'

# Verify new version
curl http://inference-service/health

# Switch all traffic
kubectl patch service inference-service -p \
  '{"spec":{"selector":{"version":"v1.2.3"}}}'

# Delete old version
kubectl delete deployment inference-service-green -n security-mlops
```

### Canary Deployment

```bash
# Deploy canary version (10% traffic)
kubectl apply -f kubernetes/canary-deployment.yaml

# Monitor canary metrics
kubectl logs -f deployment/inference-service-canary -n security-mlops

# Gradually increase traffic
kubectl patch deployment inference-service-canary \
  -p '{"spec":{"replicas":2}}'

# Promote to production
kubectl set image deployment/inference-service \
  inference=ghcr.io/reethj-07/security-inference:v1.2.3 \
  -n security-mlops

# Remove canary
kubectl delete deployment inference-service-canary -n security-mlops
```

---

## 📊 MLflow Model Promotion

### Register Model

```bash
# Via MLflow UI
# 1. Go to http://localhost:5000
# 2. Click on model
# 3. Click "Register Model"
# 4. Create new model: "security-anomaly"

# Via CLI
mlflow models list

mlflow models get-latest-versions --name security-anomaly
```

### Promote Model

```bash
# Promote to Staging
mlflow models update-model-version \
  --name security-anomaly \
  --version 1 \
  --description "v1.2.3 - Production candidate" \
  --model-stage Staging

# Promote to Production
mlflow models update-model-version \
  --name security-anomaly \
  --version 1 \
  --description "v1.2.3 - Live production model" \
  --model-stage Production

# Check stages
mlflow models list
```

---

## 🚨 Monitoring & Alerts

### Check Service Health

```bash
# Health endpoint
curl http://localhost:8000/health

# Metrics endpoint
curl http://localhost:8000/metrics

# Logs
kubectl logs deployment/inference-service -n security-mlops --tail=50

# Real-time logs
kubectl logs -f deployment/inference-service -n security-mlops
```

### Monitor Drift

```bash
# Check PSI scores
curl http://localhost:8000/metrics | grep drift_psi

# Monitor via Prometheus
# Query: drift_psi_score > 0.25

# Check Grafana dashboard
# http://localhost:3000/d/drift-monitoring
```

### Set Up Alerts

```bash
# Edit prometheus rules
vim monitoring/prometheus-rules.yml

# Add rule
- alert: HighErrorRate
  expr: rate(inference_errors_total[5m]) > 0.05
  for: 5m
  annotations:
    summary: "High error rate detected"
```

---

## 🔄 Rollback Procedure

### Manual Rollback

```bash
# Check previous deployment
kubectl rollout history deployment/inference-service -n security-mlops

# View specific revision
kubectl rollout history deployment/inference-service --revision=2 -n security-mlops

# Rollback to previous
kubectl rollout undo deployment/inference-service -n security-mlops

# Rollback to specific revision
kubectl rollout undo deployment/inference-service --to-revision=2 -n security-mlops

# Verify
kubectl rollout status deployment/inference-service -n security-mlops
```

### Emergency Fallback

```bash
# Activate SAFE_MODE
kubectl set env deployment/inference-service \
  SYSTEM_STATE=SAFE_MODE \
  -n security-mlops

# Switch to Staging model
kubectl set env deployment/inference-service \
  MODEL_STAGE=Staging \
  -n security-mlops

# Verify rollback
curl http://localhost:8000/health
```

---

## 🔐 Secrets Management

### Create Secrets

```bash
# Create DB credentials
kubectl create secret generic db-credentials \
  --from-literal=username=mlops_user \
  --from-literal=password=mlops_password \
  -n security-mlops

# Create API key
kubectl create secret generic api-keys \
  --from-literal=api-key=your-secret-key \
  -n security-mlops

# View secrets
kubectl get secrets -n security-mlops
```

### Rotate Secrets

```bash
# Update secret
kubectl patch secret db-credentials \
  -p '{"data":{"password":"'$(echo -n "new-password" | base64)'"}}' \
  -n security-mlops

# Rolling restart
kubectl rollout restart deployment/inference-service -n security-mlops
```

---

## 📊 Load Testing

### Setup Load Test

```bash
# Install locust
pip install locust

# Create locustfile.py
cat > locustfile.py << 'EOF'
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict(self):
        self.client.post("/predict", json={
            "request": "GET /api/users",
            "user_id": "user_123",
            "path": "/api/users",
            "method": "GET",
            "status_code": 200,
            "latency_ms": 45
        })
    
    @task
    def health(self):
        self.client.get("/health")
EOF

# Run load test
locust -f locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10
```

### Performance Analysis

```bash
# Check metrics during load test
kubectl top pod -n security-mlops

# View logs for errors
kubectl logs -f deployment/inference-service -n security-mlops

# Check latency percentiles
curl http://localhost:8000/metrics | grep latency
```

---

## 🔍 Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n security-mlops

# Check events
kubectl get events -n security-mlops --sort-by=.metadata.creationTimestamp

# Check logs
kubectl logs <pod-name> -n security-mlops

# Check resource requests
kubectl describe node <node-name>
```

### Connection Issues

```bash
# Test connectivity to MLflow
kubectl exec -it <pod-name> -n security-mlops -- \
  curl http://mlflow:5000/health

# Test DB connection
kubectl exec -it <pod-name> -n security-mlops -- \
  psql postgresql://user@postgres:5432/security_mlops

# Check DNS
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup postgres
```

### Model Loading Fails

```bash
# Check MLflow connectivity
kubectl logs <pod-name> -n security-mlops | grep mlflow

# Verify model exists
mlflow models list

# Check model artifacts
aws s3 ls s3://bucket/mlflow/artifacts/

# Manual model test
python -c "import mlflow; model = mlflow.pyfunc.load_model('models:/security-anomaly/Production')"
```

---

## 📝 Deployment Log Template

```markdown
# Deployment Log - Date: YYYY-MM-DD

## Version
- Model: v1.2.3
- Docker Image: ghcr.io/reethj-07/security-inference:v1.2.3
- Commit: abc123def456

## Pre-Deployment Checks
- [x] Tests passed
- [x] Code reviewed
- [x] Documentation updated
- [x] Infrastructure ready

## Deployment Steps
- [x] Build Docker image
- [x] Push to GHCR
- [x] Deploy to Kubernetes
- [x] Verify health checks
- [x] Monitor metrics

## Validation
- Health check: ✅ PASS
- Prediction latency: 45ms (threshold: 500ms)
- Error rate: 0.1% (threshold: 5%)
- Drift PSI: 0.18 (threshold: 0.25)

## Result: ✅ SUCCESSFUL

## Notes
- Smooth rollout completed
- No errors observed
- All metrics nominal

## Rollback Plan
- Previous version: v1.2.2
- Rollback command: kubectl rollout undo deployment/inference-service
```

---

## 🎯 Success Criteria

✅ Deployment successful when:
- All pods are Running
- Health check returns 200
- Prediction latency < 500ms
- Error rate < 5%
- No alerts firing
- Metrics match baseline ±10%

❌ Rollback if:
- Pod fails to start
- Health check fails consistently
- Error rate > 10%
- Latency spike > 2x baseline
- OOM or resource limit exceeded
- Any critical alert fires
