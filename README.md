# Multi-variate-Time-series-Anomaly-Detection

A production-ready anomaly detection system for multi-variate time-series data using unsupervised deep learning (Recurrent GRU-VAE).

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Git

### Installation (5 minutes)

```bash
# 1. Clone repository
cd c:\Users\maryj\Anomaly_Detection\mlops_project

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env with your settings

# 5. Prepare data
dvc pull

# 6. Start backend
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

**Server is now running at:** http://127.0.0.1:8000

Test it:

```bash
curl http://127.0.0.1:8000/health
```

---

## 📚 Documentation

### Getting Started

- **[SETUP.md](SETUP.md)** - Complete installation and configuration guide
- **[API.md](API.md)** - Full API reference with examples
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Solutions to common issues

### Project Structure

```
├── src/
│   ├── app.py                 # FastAPI backend service
│   ├── train_pipeline.py      # Model training pipeline
│   ├── evaluate.py            # Model evaluation
│   ├── monitor.py             # Drift monitoring engine
│   ├── drift_injector.py      # Test data injection
│   ├── dashboard.py           # Streamlit dashboard
│   └── split_data.py          # Data preprocessing
├── models/                     # Trained model artifacts
├── data/                       # Training and test data
├── logs/                       # Application logs
├── requirements.txt            # Python dependencies
├── .env.example               # Environment configuration template
├── API.md                      # API documentation
├── SETUP.md                    # Setup guide
├── TROUBLESHOOTING.md          # Troubleshooting guide
└── README.md                   # This file
```

---

## 🎯 Key Features

### Backend API

- **RESTful FastAPI** with automatic documentation
- **Health checks** and service monitoring
- **Real-time predictions** with sub-200ms latency
- **Metrics management** with historical queries
- **Structured logging** for all operations

### ML Model

- **Recurrent GRU-VAE** for sequence anomaly detection
- **Adaptive thresholds** based on rolling statistics
- **Unsupervised learning** - no labeled data required
- **38-dimensional input** for comprehensive system monitoring

### Operations

- **Data drift detection** using Evidently AI
- **Automated retraining** pipeline
- **MLflow experiment tracking**
- **SQLite persistence** for production telemetry
- **Streamlit dashboard** for visualization

---

## 🔌 API Endpoints

### Health & Status

```bash
GET /              # Service status
GET /health        # Detailed health check
```

### Predictions

```bash
POST /predict      # Submit telemetry, get anomaly alert
```

### Metrics

```bash
GET /metrics       # Historical metrics
GET /metrics/summary  # Summary statistics
DELETE /metrics    # Clear metrics (testing only)
```

Full documentation: [API.md](API.md)

---

## 🚀 Running Services

### Backend Only

```bash
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

### With Monitoring

```bash
# Terminal 1: Backend
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000

# Terminal 2: Monitoring engine
python src/monitor.py
```

### With Test Data Injection

```bash
# Terminal 1: Backend
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000

# Terminal 2: Test data injector
python src/drift_injector.py
```

### Full Stack

```bash
# Terminal 1: Backend
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000

# Terminal 2: Monitoring
python src/monitor.py

# Terminal 3: Test data
python src/drift_injector.py

# Terminal 4: Dashboard (requires InfluxDB)
streamlit run src/dashboard.py
```

---

## 📊 Model Architecture

### GRU-VAE (Generative Adversarial Variational Autoencoder)

**Input:** 30 timesteps × 38 metrics (time-series sequences)

**Encoder:**

- GRU layer (128 units) - captures temporal dependencies
- BatchNormalization - stabilizes learning
- Dense layers - feature compression
- Latent space (16 dimensions)

**Decoder:**

- Dense layer - feature expansion
- RepeatVector - sequence generation
- GRU layer (128 units) - recurrent reconstruction
- TimeDistributed Dense - feature-wise output

**Training:**

- Loss: MSE (reconstruction) + KL divergence (0.02 weight)
- Optimizer: Adam (learning rate 0.001)
- Batch size: 128
- Epochs: 15

**Inference:**

- Reconstruction error: MSE across all features and timesteps
- Threshold: 99.5th percentile of validation errors
- Alert confirmation: 4+ consecutive detections

---

## 📈 Performance

Rresults on test data:

- **Precision:** 0.99 (few false positives)
- **Recall:** 0.75 (catches most anomalies)
- **F1-Score:** 0.75-0.87
- **Point Adjusted F1-Score:** 0.99
- **Latency:** 50-200ms per prediction (standard CPU-dependent)

---

## 🔧 Configuration

All settings in `.env` file:

```bash
# API Configuration
FASTAPI_HOST=127.0.0.1
FASTAPI_PORT=8000

# Model Paths
ENCODER_MODEL_PATH=models/vae_encoder.keras
DECODER_MODEL_PATH=models/vae_decoder.keras
SCALER_MODEL_PATH=models/scaler.pkl

# Alert Tuning
ALERT_SIGMA_MULTIPLIER=4.0      # Sensitivity (higher = less sensitive)
ALERT_PERSISTENCE_WINDOW=4      # Consecutive alerts to confirm
DRIFT_THRESHOLD=0.45            # Feature drift trigger for retraining

# InfluxDB (for dashboard)
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=your_token
```

See [SETUP.md](SETUP.md) for all options.

---

## 🧪 Testing

### Test API Health

```bash
curl http://127.0.0.1:8000/health
```

### Submit Test Data

```bash
python src/drift_injector.py
```

### Run Evaluation

```bash
python src/evaluate.py
```

### Monitor Logs

```bash
tail -f logs/app.log
```

---

## 🚨 Troubleshooting

Common issues and solutions: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

Quick diagnostics:

```bash
# Check if server is running
curl http://127.0.0.1:8000/health

# View logs
cat logs/app.log

# Verify models exist
ls -la models/

# Test data format
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp": 1234567890, "metrics": [0.5]*38}'
```

---

## 📦 Dependencies

- **FastAPI** - Web framework
- **TensorFlow/Keras** - Deep learning architecture (GRU-VAE)
- **NumPy/Pandas** - Raw telemetry data processing
- **scikit-learn** - ML utilities
- **MLflow** - Experiment tracking and Model Registry
- **DVC** - Data versioning
- **Evidently** - Data drift detection
- **Streamlit** - Dashboard
- **InfluxDB client** - Time-series DB

Full list: [requirements.txt](requirements.txt)

---

## 📝 Model Training

### Train from scratch

```bash
# 1. Prepare data
python src/split_data.py

# 2. Train model
python src/train_pipeline.py

# 3. Evaluate model
python src/evaluate.py
```

This will:

- Train on processed data
- Save models to `./models/`
- Log experiments to MLflow
- Calculate optimal threshold

---

## 🔄 Data Flow

```
Raw Telemetry (38 metrics)
        ↓
MinMaxScaler (normalize to 0-1)
        ↓
30-timestep rolling windows
        ↓
GRU-VAE Encoder (compress to latent space)
        ↓
GRU-VAE Decoder (reconstruct)
        ↓
Reconstruction Error (MSE)
        ↓
Adaptive Threshold (mean + 4σ)
        ↓
Alert Decision (0 or 1)
        ↓
SQLite persistence
```

---

## 📊 Monitoring

### Application Logs

```bash
logs/app.log              # API server
logs/monitor.log          # Drift monitoring
logs/drift_injector.log   # Test data injection
logs/dashboard.log        # Streamlit dashboard
```

### Metrics Database

```bash
data/production_telemetry.db  # SQLite with live metrics
```

### MLflow Experiments

```bash
mlruns/                       # Experiment tracking
```

---

## 🌐 API Examples

### Python

```python
import requests
import time

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "timestamp": time.time(),
        "metrics": [0.5] * 38
    }
)

result = response.json()
if result.get("alert") == 1:
    print("🚨 Anomaly detected!")
```

### cURL

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": 1234567890.5,
    "metrics": [0.45, 0.32, ..., 0.42]
  }'
```

More examples: [API.md](API.md)

---

## 🔐 Security

For production deployment:

- [ ] Add API key authentication
- [ ] Use HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Use environment variables for secrets
- [ ] Implement request validation
- [ ] Monitor error logs
- [ ] Set up alerting

---

## 📚 References

- **OmniAnomaly Paper:** https://arxiv.org/abs/1802.06368
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **TensorFlow Docs:** https://www.tensorflow.org/
- **DVC Documentation:** https://dvc.org/
- **MLflow:** https://www.mlflow.org/

---

**Version:** 2.0  
**Status:** Production Ready  
**Last Updated:** 2026-08-30
