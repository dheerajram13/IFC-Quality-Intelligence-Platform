# IFC Quality Intelligence Platform

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-in%20development-orange.svg)
![BIM](https://img.shields.io/badge/BIM-IFC-brightgreen.svg)
![3D](https://img.shields.io/badge/3D-Geometry-purple.svg)
![ML](https://img.shields.io/badge/ML-Anomaly%20Detection-red.svg)

> Automated BIM Model Validation, Quality Metrics, and Anomaly Detection for Digital Engineering Pipelines

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- [Usage](#usage)
  - [CLI Usage](#cli-usage)
  - [API Usage](#api-usage)
  - [Dashboard Usage](#dashboard-usage)
- [Project Structure](#project-structure)
- [Quality Metrics](#quality-metrics)
- [Configuration](#configuration)
- [Development](#development)
  - [Setting Up Development Environment](#setting-up-development-environment)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
- [Deployment](#deployment)
- [Examples](#examples)
- [Roadmap](#roadmap)
- [Contact](#contact)

## Overview

The **IFC Quality Intelligence Platform** is a lightweight, production-ready Python system that automatically validates IFC (Industry Foundation Classes) building models. It extracts geometry and metadata features, detects data quality issues using rule-based validation and machine learning, and produces actionable metrics through interactive dashboards and REST APIs.

This platform is designed to:
- **Improve data reliability** in BIM workflows
- **Reduce manual QA effort** by 60-80%
- **Enable scalable automation** across digital engineering pipelines
- **Provide actionable insights** for technical and business stakeholders

### Why This Matters

Modern construction and infrastructure projects rely on BIM models containing hundreds of thousands of elements. Poor data quality in these models propagates into cost estimation, scheduling, digital twin systems, and analytics—creating compounding operational risks. This platform addresses these challenges with automated, repeatable, and measurable quality validation.

## Problem Statement

**Current Challenges in BIM Quality Assurance:**

- ❌ **High labor cost** and slow review cycles
- ❌ **Inconsistent validation** standards across teams
- ❌ **Increased risk** of downstream rework and errors
- ❌ **Limited visibility** into model health over time
- ❌ **Poor scalability** as model complexity grows

**Our Solution:**

An intelligent, automated quality validation system that combines rule-based checks with ML-powered anomaly detection to identify issues early, standardize validation, and provide continuous quality monitoring.

## Key Features

### Core Capabilities

- **Automated IFC Parsing**: Extract element metadata and 3D geometry from IFC files
- **Feature Engineering**: Compute ML-ready features (bounding boxes, centroids, volumes, aspect ratios)
- **Rule-Based Validation**: Detect missing metadata, invalid geometry, duplicates, and scale issues
- **ML Anomaly Detection**: Identify unusual geometry patterns using Isolation Forest
- **Quality Scoring**: Business-friendly KPIs (Quality Score 0-100, issue rates, completeness metrics)
- **Interactive Dashboards**: Plotly/Streamlit visualizations with exportable HTML reports
- **REST API**: FastAPI endpoints for pipeline integration
- **CLI Tool**: Batch scanning with configurable outputs
- **MLOps Integration**: Experiment tracking and artifact management with MLflow

### Quality Checks

| Check Type | Description | Severity |
|------------|-------------|----------|
| Missing Metadata | Elements without Name or ObjectType | Major |
| Duplicate GlobalId | Duplicate unique identifiers (data integrity) | Critical |
| Degenerate Geometry | Zero or near-zero bounding box dimensions | Major |
| Scale Mismatches | Suspicious unit or scale issues | Critical |
| Coordinate Anomalies | Elements too far from origin | Minor |
| ML Anomalies | Unusual geometry patterns flagged by ML model | Variable |

## Architecture

```
                    IFC File
                       |
                       v
        +------------------------------+
        |        IFC Loader            |
        |   (IfcOpenShell Parser)      |
        +------------------------------+
                       |
          +------------+------------+
          |            |            |
          v            v            v
    +---------+  +---------+  +-------------+
    |Geometry |  |Metadata |  |Feature Store|
    |Extractor|  |Extractor|  | (DataFrame) |
    +---------+  +---------+  +-------------+
          |            |            |
          +------------+------------+
                       |
                       v
        +------------------------------+
        |     Validation Engine        |
        |  +----------+  +-----------+ |
        |  |  Rule    |  | ML Anomaly| |
        |  |  Engine  |  | Detection | |
        |  +----------+  +-----------+ |
        +------------------------------+
                       |
                       v
              +-----------------+
              | Metrics Engine  |
              |(Scoring & KPIs) |
              +-----------------+
                       |
          +------------+------------+
          |            |            |
          v            v            v
    +----------+  +--------+  +--------+
    |Dashboard |  |  API   |  |  CLI   |
    |(Streamlit)  |(FastAPI)  | (Typer)|
    +----------+  +--------+  +--------+
          |            |            |
          +------------+------------+
                       |
                       v
              +-----------------+
              | MLflow Tracking |
              | (Experiments &  |
              |   Artifacts)    |
              +-----------------+
```

## Technology Stack

### Core Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.10+ | Core implementation |
| **IFC Processing** | IfcOpenShell | Parse IFC files and extract geometry |
| **Data Processing** | pandas, NumPy | Feature engineering and aggregation |
| **Machine Learning** | scikit-learn | Anomaly detection (IsolationForest) |
| **Visualization** | Plotly, Streamlit | Interactive dashboards |
| **API Framework** | FastAPI | REST API endpoints |
| **MLOps** | MLflow | Experiment tracking and artifacts |
| **CLI** | Typer | Command-line interface |
| **Containerization** | Docker | Deployment packaging |

## Getting Started

### Prerequisites

- **Python**: 3.10 or higher
- **pip**: Latest version
- **Git**: For cloning the repository
- **Docker** (optional): For containerized deployment

### Installation

#### Option 1: Install from Source

```bash
# Clone the repository
git clone https://github.com/dheerajram13/ifc-quality-intelligence.git
cd ifc-quality-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

#### Option 2: Using Docker

```bash
# Build the Docker image
docker build -t ifcqi:latest .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data ifcqi:latest
```

### Quick Start

#### 1. Scan an IFC File (CLI)

```bash
# Basic scan
ifcqi scan examples/sample.ifc --out reports/

# Scan with all features enabled
ifcqi scan examples/sample.ifc --out reports/ --html --ml --mlflow

# Batch scan multiple files
ifcqi batch-scan data/*.ifc --out reports/batch/
```

#### 2. Launch Dashboard

```bash
# Start Streamlit dashboard
streamlit run apps/dashboard.py

# Access at http://localhost:8501
```

#### 3. Start API Server

```bash
# Start FastAPI server
uvicorn ifcqi.api.main:app --reload --host 0.0.0.0 --port 8000

# API documentation at http://localhost:8000/docs
```

## Usage

### CLI Usage

The CLI provides flexible options for scanning IFC files:

```bash
# Help
ifcqi --help

# Basic scan with JSON output
ifcqi scan model.ifc --out ./output

# Generate HTML report
ifcqi scan model.ifc --out ./output --html

# Enable ML anomaly detection
ifcqi scan model.ifc --out ./output --ml

# Track with MLflow
ifcqi scan model.ifc --out ./output --mlflow --experiment-name "project-alpha"

# Custom configuration
ifcqi scan model.ifc --out ./output --config config.yaml

# Verbose logging
ifcqi scan model.ifc --out ./output --verbose
```

### API Usage

#### Scan IFC File

```bash
# Upload and scan IFC file
curl -X POST "http://localhost:8000/scan" \
  -F "file=@path/to/model.ifc" \
  -F "enable_ml=true"
```

**Response:**

```json
{
  "status": "success",
  "metrics": {
    "total_elements": 15420,
    "quality_score": 87.5,
    "critical_issues": 3,
    "major_issues": 24,
    "minor_issues": 156,
    "issue_rate": 11.87,
    "metadata_completeness": 0.92,
    "anomaly_rate": 0.03
  },
  "issues": [...],
  "anomalies": [...],
  "processing_time": 12.45
}
```

#### Health Check

```bash
curl http://localhost:8000/health
```

### Dashboard Usage

The Streamlit dashboard provides interactive visualization:

1. **Upload IFC File**: Drag and drop or browse
2. **Configure Options**: Enable/disable ML, select quality thresholds
3. **View Results**:
   - Quality Score and KPI tiles
   - Issue distribution charts
   - Top offenders by element type
   - Anomaly scatter plots
   - Exportable reports

## Project Structure

```
ifc-quality-intelligence/
├── src/
│   └── ifcqi/
│       ├── __init__.py           # Package initialization
│       ├── config.py             # Configuration management
│       ├── logger.py             # Logging setup
│       ├── ifc_loader.py         # IFC file parsing
│       ├── geometry.py           # Geometry extraction
│       ├── features.py           # Feature engineering
│       ├── checks.py             # Rule-based validation
│       ├── metrics.py            # Metrics calculation
│       ├── viz.py                # Visualization utilities
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── train.py          # Model training
│       │   └── infer.py          # Inference pipeline
│       ├── api/
│       │   ├── __init__.py
│       │   └── main.py           # FastAPI application
│       └── cli.py                # CLI implementation
├── apps/
│   └── dashboard.py              # Streamlit dashboard
├── examples/
│   ├── sample.ifc                # Sample IFC files
│   └── notebooks/                # Jupyter notebooks
├── tests/
│   ├── test_loader.py
│   ├── test_checks.py
│   └── test_metrics.py
├── data/                         # Data directory (gitignored)
├── reports/                      # Output reports (gitignored)
├── mlruns/                       # MLflow tracking (gitignored)
├── config/
│   └── default.yaml              # Default configuration
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose setup
├── pyproject.toml                # Project metadata and dependencies
├── requirements.txt              # Python dependencies
├── .pre-commit-config.yaml       # Pre-commit hooks
├── .gitignore
└── README.md
```

## Quality Metrics

### Quality Score (0-100)

Composite score calculated from:
- **Issue severity** (weighted: Critical × 3, Major × 2, Minor × 1)
- **Metadata completeness**
- **Geometry coverage**
- **Anomaly rate**

**Formula:**
```
Quality Score = 100 - (weighted_issues / total_elements × 100)
```

### Key Performance Indicators

| Metric | Description | Target |
|--------|-------------|--------|
| **Total Elements** | Number of IFC entities scanned | N/A |
| **Quality Score** | Overall model health (0-100) | > 85 |
| **Critical Issues** | Blocking quality violations | 0 |
| **Issue Rate** | Issues per 1,000 elements | < 10 |
| **Metadata Completeness** | % elements with required fields | > 95% |
| **Anomaly Rate** | % geometry flagged as unusual | < 5% |

## Configuration

### Default Configuration (`config/default.yaml`)

```yaml
# IFC Processing
ifc:
  extract_geometry: true
  max_elements: null  # null = no limit

# Quality Checks
checks:
  severity_weights:
    critical: 3.0
    major: 2.0
    minor: 1.0
  geometry:
    min_dimension: 0.001  # meters
    max_dimension: 10000  # meters
    max_distance_from_origin: 100000  # meters

# ML Configuration
ml:
  enabled: true
  model_type: "IsolationForest"
  contamination: 0.05
  random_state: 42

# Output
output:
  formats: ["json", "csv", "html"]
  include_geometry: false

# MLflow
mlflow:
  tracking_uri: "./mlruns"
  experiment_name: "ifc-quality"
```

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ifcqi --cov-report=html

# Run specific test file
pytest tests/test_checks.py -v

# Run with markers
pytest -m "not slow"
```

### Code Quality

This project uses:
- **Black**: Code formatting
- **Ruff**: Linting
- **mypy**: Type checking
- **isort**: Import sorting

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t ifcqi:v1.0 .

# Run API server
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/mlruns:/app/mlruns \
  --name ifcqi-api \
  ifcqi:v1.0

# Run with docker-compose
docker-compose up -d
```

### Cloud Deployment

**Azure Container Instances** (example):

```bash
# Login to Azure
az login

# Create resource group
az group create --name ifcqi-rg --location eastus

# Deploy container
az container create \
  --resource-group ifcqi-rg \
  --name ifcqi-api \
  --image youracr.azurecr.io/ifcqi:latest \
  --cpu 2 --memory 4 \
  --ports 8000 \
  --environment-variables MLFLOW_TRACKING_URI=<uri>
```

## Examples

### Example Output

**Terminal Output:**
```
Scanning IFC file: sample.ifc
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:12

Quality Report Summary
═══════════════════════════════════════════════════════
Total Elements:        15,420
Quality Score:         87.5/100
Critical Issues:       3
Major Issues:          24
Minor Issues:          156
Issue Rate:            11.87 per 1,000 elements
Metadata Completeness: 92.3%
Anomaly Rate:          2.8%

Top Issues:
  1. Missing Name (Major): 18 occurrences
  2. Degenerate Geometry (Major): 6 occurrences
  3. Missing ObjectType (Minor): 142 occurrences

Reports generated in: ./reports/sample/
  ✓ metrics.json
  ✓ issues.csv
  ✓ anomalies.csv
  ✓ report.html
```

## Roadmap

### Planned Features

- [ ] **Enhanced Geometry Features**: Surface area, mesh complexity, volume calculations
- [ ] **Azure ML Integration**: Cloud-based training pipelines
- [ ] **Azure AI Search**: Semantic model search and retrieval
- [ ] **OpenAI Integration**: Natural language report summarization
- [ ] **Longitudinal Tracking**: Quality metrics across project timeline
- [ ] **Automated Regression Testing**: CI/CD quality validation
- [ ] **Multi-Model Comparison**: Compare versions of the same model
- [ ] **Custom Rule Builder**: UI for non-technical users to create checks
- [ ] **Real-time Monitoring**: WebSocket-based live validation

## Contact

**Dheeraj Srirama**

- Portfolio: [https://dheerajsrirama.netlify.app](https://dheerajsrirama.netlify.app)
- GitHub: [@dheerajram13](https://github.com/dheerajram13)
- LinkedIn: [Your LinkedIn Profile](#)

---

**Built with precision for the future of digital engineering**

*This project demonstrates production-oriented ML engineering practices applied to real-world AEC (Architecture, Engineering, Construction) data, showcasing feature extraction, validation pipelines, metrics tracking, experiment management, and API-based deployment.*
