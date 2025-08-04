# 🎬 YouTube Comment Intelligence

A comprehensive MLOps pipeline for analyzing YouTube comment intelligence with secure API endpoints, interactive web interface, and enterprise-grade orchestration.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
# Copy environment template
cp env_template.txt .env

# Edit .env file with your API keys
# Replace placeholder values with your actual keys
```

### 3. Start Applications

**Option A: Simple Start (Development)**
```bash
# Start both Flask API and Streamlit UI
python run.py

# Or start individually:
# Terminal 1: Flask API
python app.py

# Terminal 2: Streamlit UI
streamlit run streamlit_app.py --server.port=8501
```

**Option B: Docker Compose (Recommended)**
```bash
# Start all services with orchestration
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

**Option C: Kubernetes (Production)**
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/base/

# Check deployment status
kubectl get pods -n youtube-comment-intelligence
```

## 📍 Access Points

### **Core Applications**
- **🌐 Flask API**: http://localhost:8080
- **📊 Streamlit UI**: http://localhost:8501
- **📚 API Documentation**: http://localhost:8080/docs
- **🏥 Health Check**: http://localhost:8080/health

### **Monitoring & Observability**
- **📈 Grafana**: http://localhost:3000 (admin/admin)
- **📊 Prometheus**: http://localhost:9090
- **📝 Kibana**: http://localhost:5601
- **🌼 Flower**: http://localhost:5555 (Celery monitoring)

### **Infrastructure**
- **🗄️ PostgreSQL**: localhost:5432
- **🔴 Redis**: localhost:6379
- **🌐 Nginx**: http://localhost:80

## 🏗️ Architecture & Orchestration

### **🐳 Docker Compose Stack**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  Grafana Dash   │    │  Kibana Logs    │
│   (Load Bal.)   │    │   (Monitoring)  │    │  (Visualization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Flask API     │    │   Prometheus    │    │  Elasticsearch  │
│   (Backend)     │    │   (Metrics)     │    │   (Log Store)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Streamlit UI   │    │   PostgreSQL    │    │     Redis       │
│   (Frontend)    │    │   (Database)    │    │   (Cache)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **☸️ Kubernetes Deployment**
- **Multi-environment support** (dev/staging/prod)
- **Auto-scaling** based on CPU/memory usage
- **Health checks** and **auto-restart**
- **Resource limits** and **requests**
- **Secrets management** for sensitive data
- **Network policies** for security

### **🔄 CI/CD Pipeline**
```mermaid
graph LR
    A[Code Push] --> B[Automated Tests]
    B --> C[Security Scan]
    C --> D[Build Images]
    D --> E[Deploy Staging]
    E --> F[Deploy Production]
    F --> G[Slack Notification]
```

## 🔐 Security Features

- ✅ **API Key Authentication**: Required for all endpoints
- ✅ **Rate Limiting**: 100 requests/hour per client
- ✅ **Input Validation**: XSS protection, length limits
- ✅ **Security Headers**: X-Frame-Options, X-XSS-Protection
- ✅ **Environment Variables**: Secure configuration management
- ✅ **HTTPS Support**: SSL/TLS encryption
- ✅ **Network Policies**: Kubernetes security

## 📁 Project Structure

```
End-to-end-Youtube-Sentiment/
├── 📄 Core Application Files
│   ├── app.py                          # Flask API server
│   ├── streamlit_app.py                # Streamlit web interface
│   ├── security_config.py              # Security configuration
│   ├── run.py                          # Simple startup script
│   ├── start_apps.py                   # Advanced startup script
│   └── generate_secrets.py             # Secret key generator
├── 🐳 Orchestration Files
│   ├── docker-compose.yml              # Docker Compose stack
│   ├── Dockerfile                      # API container
│   ├── Dockerfile.streamlit            # Web UI container
│   ├── k8s/                           # Kubernetes manifests
│   │   ├── base/                      # Base K8s configs
│   │   └── overlays/                  # Environment configs
│   └── .github/workflows/ci-cd.yml    # CI/CD pipeline
├── 📊 Monitoring & Observability
│   ├── monitoring/                     # Monitoring configs
│   │   ├── prometheus.yml             # Metrics collection
│   │   ├── filebeat.yml               # Log shipping
│   │   └── grafana/                   # Dashboard configs
│   └── nginx/                         # Reverse proxy config
├── 📄 Configuration Files
│   ├── requirements.txt                # Python dependencies
│   ├── .env                           # Environment variables
│   ├── env_template.txt               # Environment template
│   └── Makefile                       # Build automation
├── 🧪 Testing & Quality
│   ├── tests/                         # Test files
│   │   ├── test_api.py               # API tests
│   │   ├── test_security.py          # Security tests
│   │   ├── test_model.py             # Model tests
│   │   └── test_env_loading.py       # Environment tests
│   └── run_tests.py                   # Test runner
├── 📚 Documentation
│   ├── docs/                          # Documentation
│   │   ├── SECURITY.md               # Security guide
│   │   ├── TESTING.md                # Testing guide
│   │   └── PROJECT_OVERVIEW.md       # Project overview
│   └── README.md                      # This file
└── 📦 Data & Models
    ├── models/                        # Trained ML models
    ├── logs/                          # Application logs
    └── assets/                        # Static assets
```

## 🔑 API Configuration

### Required Environment Variables

Create a `.env` file with:

```env
# YouTube API Configuration
YOUTUBE_API_KEY=your_youtube_api_key_here

# Flask Configuration
FLASK_PORT=8080
FLASK_HOST=0.0.0.0
FLASK_DEBUG=True

# Security Configuration
SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here
REQUIRE_API_KEY=True
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
MAX_COMMENT_LENGTH=1000
MAX_BATCH_SIZE=100
PREDICTION_CONFIDENCE_THRESHOLD=0.6

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/youtube_intelligence
REDIS_URL=redis://localhost:6379/0

# Monitoring Configuration
ENABLE_METRICS=True
LOG_LEVEL=INFO
```

### Generate Secure Keys

```bash
# Generate new secret keys
python generate_secrets.py

# Or use one-liner
python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32)); print('JWT_SECRET_KEY=' + secrets.token_hex(32))"
```

## 🧪 Testing

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test Categories
```bash
python run_tests.py --type unit
python run_tests.py --type api
python run_tests.py --type security
```

### Test Environment Loading
```bash
python tests/test_env_loading.py
```

## 📊 API Endpoints

### Authentication Required Endpoints

All endpoints require API key in header: `X-API-Key: your_api_key`

#### POST /predict
Analyze sentiment of comments
```bash
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"comments": ["This is a great video!"]}'
```

#### POST /predict_with_timestamps
Analyze sentiment with timestamps
```bash
curl -X POST http://localhost:8080/predict_with_timestamps \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"comments": ["Great video!"], "timestamps": ["2024-01-01T10:00:00Z"]}'
```

#### POST /generate_chart
Generate sentiment distribution chart
```bash
curl -X POST http://localhost:8080/generate_chart \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"comments": ["Great video!", "Amazing content!"]}'
```

#### POST /generate_wordcloud
Generate word cloud data
```bash
curl -X POST http://localhost:8080/generate_wordcloud \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"comments": ["This is a great tutorial", "Very helpful video"]}'
```

#### POST /generate_trend_graph
Generate trend analysis
```bash
curl -X POST http://localhost:8080/generate_trend_graph \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"comments": ["Great!"], "timestamps": ["2024-01-01T10:00:00Z"]}'
```

### Public Endpoints

#### GET /health
Health check endpoint
```bash
curl -X GET http://localhost:8080/health
```

#### GET /docs
API documentation
```bash
curl -X GET http://localhost:8080/docs
```

## 🚀 Deployment Options

### **Development Environment**
```bash
# Simple local development
python run.py

# Or with Docker Compose
docker-compose up -d
```

### **Staging Environment**
```bash
# Deploy to staging
kubectl apply -f k8s/overlays/staging/

# Check deployment
kubectl get pods -n youtube-comment-intelligence
```

### **Production Environment**
```bash
# Deploy to production
kubectl apply -f k8s/overlays/prod/

# Monitor deployment
kubectl get pods -n youtube-comment-intelligence
```

## 🔧 Development

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Running in Development
```bash
# Start Flask API with auto-reload
FLASK_DEBUG=True python app.py

# Start Streamlit UI
streamlit run streamlit_app.py --server.port=8501
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build Docker images
docker build -t youtube-comment-intelligence .
docker build -t youtube-comment-intelligence-streamlit -f Dockerfile.streamlit .

# Run with Docker Compose
docker-compose up -d

# Or run individually
docker run -p 8080:8080 youtube-comment-intelligence
docker run -p 8501:8501 youtube-comment-intelligence-streamlit
```

### Production Deployment
```bash
# Build for production
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale api=3 --scale web=2
```

## ☸️ Kubernetes Deployment

### Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Create namespace
kubectl create namespace youtube-comment-intelligence
```

### Deploy Application
```bash
# Apply base configuration
kubectl apply -f k8s/base/

# Check deployment status
kubectl get pods -n youtube-comment-intelligence

# View logs
kubectl logs -f deployment/youtube-comment-intelligence-api -n youtube-comment-intelligence
```

### Scaling
```bash
# Scale API replicas
kubectl scale deployment youtube-comment-intelligence-api --replicas=5 -n youtube-comment-intelligence

# Scale web replicas
kubectl scale deployment youtube-comment-intelligence-web --replicas=3 -n youtube-comment-intelligence
```

## 📈 Monitoring & Observability

### **Grafana Dashboards**
- **Application Metrics**: Request rate, response time, error rate
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Sentiment analysis accuracy, user engagement

### **Prometheus Metrics**
- **Custom Metrics**: API request duration, prediction accuracy
- **System Metrics**: Container resource usage
- **Business Metrics**: Comments processed, sentiment distribution

### **ELK Stack Logs**
- **Application Logs**: API requests, errors, security events
- **System Logs**: Container logs, infrastructure events
- **Business Logs**: User interactions, feature usage

### **Health Checks**
```bash
# API Health
curl http://localhost:8080/health

# Web UI Health
curl http://localhost:8501

# Database Health
kubectl exec -it deployment/youtube-comment-intelligence-postgres -- pg_isready
```

## 🔄 CI/CD Pipeline

### **Automated Workflow**
1. **Code Push** → Triggers CI/CD pipeline
2. **Automated Testing** → Unit tests, integration tests
3. **Security Scanning** → Bandit, Safety checks
4. **Docker Build** → Multi-stage image building
5. **Staging Deployment** → Automated staging deployment
6. **Production Deployment** → Blue-green deployment
7. **Slack Notification** → Deployment status alerts

### **Pipeline Features**
- ✅ **Automated Testing** on every commit
- ✅ **Security Scanning** for vulnerabilities
- ✅ **Docker Image Building** with caching
- ✅ **Multi-environment Deployment** (staging/prod)
- ✅ **Rollback Capability** on failure
- ✅ **Slack Notifications** for deployment status

## 🛠️ Troubleshooting

### Common Issues

#### 1. Model Loading Error
**Error**: `[Errno 21] Is a directory: './models'`
**Solution**: Ensure model files exist in `models/` directory

#### 2. NLTK Data Missing
**Error**: `LookupError: Resource stopwords not found`
**Solution**: Run `python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"`

#### 3. Port Already in Use
**Error**: `Address already in use`
**Solution**: Kill existing processes or use different port

#### 4. Authentication Errors
**Error**: `401 Unauthorized`
**Solution**: Ensure API key is set in `.env` file

#### 5. Docker Issues
**Error**: `Cannot connect to the Docker daemon`
**Solution**: Start Docker service and ensure user is in docker group

#### 6. Kubernetes Issues
**Error**: `The connection to the server localhost:8080 was refused`
**Solution**: Start minikube or configure kubectl for your cluster

## 📚 Documentation

- **Security Guide**: `docs/SECURITY.md`
- **Testing Guide**: `docs/TESTING.md`
- **Project Overview**: `docs/PROJECT_OVERVIEW.md`
- **API Key Setup**: `API_KEY_GUIDE.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YouTube Data API v3
- LightGBM for sentiment classification
- Flask for API framework
- Streamlit for web interface
- NLTK for text preprocessing
- Docker for containerization
- Kubernetes for orchestration
- Prometheus & Grafana for monitoring

---

**🎉 Happy analyzing!** Your YouTube Comment Intelligence system is ready with enterprise-grade orchestration!
