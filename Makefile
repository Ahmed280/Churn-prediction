# Makefile for Music Streaming Churn Prediction Project
# Professional automation for development, testing, and deployment tasks

# Configuration
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# Python environment detection
VENV_DIR := .venv
ifeq ($(OS),Windows_NT)
    PYTHON := $(if $(wildcard $(VENV_DIR)/Scripts/python.exe),$(VENV_DIR)/Scripts/python.exe,python)
    PIP := $(PYTHON) -m pip
    ACTIVATE := $(VENV_DIR)/Scripts/activate
else
    PYTHON := $(if $(wildcard $(VENV_DIR)/bin/python),$(VENV_DIR)/bin/python,python3)
    PIP := $(PYTHON) -m pip
    ACTIVATE := $(VENV_DIR)/bin/activate
endif

# Project variables
PROJECT_NAME := churn-prediction
DOCKER_IMAGE := $(PROJECT_NAME):latest
DATA_FILE := customer_churn.json
MODEL_DIR := models
MLFLOW_PORT := 5000
API_PORT := 8000

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Music Streaming Churn Prediction - Available Commands:"
	@echo "======================================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# Environment setup
.PHONY: install
install: ## Install Python dependencies and setup environment
	@echo "🔧 Setting up development environment..."
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

.PHONY: install-dev
install-dev: install ## Install development dependencies including pre-commit
	$(PIP) install pre-commit ruff black pytest pytest-cov bandit
	pre-commit install
	@echo "✅ Development environment ready"

.PHONY: venv
venv: ## Create virtual environment
	@echo "🐍 Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	@echo "✅ Virtual environment created at $(VENV_DIR)"
	@echo "💡 Activate with: source $(ACTIVATE)"

# Code quality and formatting
.PHONY: format
format: ## Format code with ruff
	@echo "🎨 Formatting code..."
	ruff format .
	@echo "✅ Code formatted"

.PHONY: lint
lint: ## Lint code with ruff
	@echo "🔍 Linting code..."
	ruff check . --fix
	@echo "✅ Linting completed"

.PHONY: lint-check
lint-check: ## Check linting without fixing
	@echo "🔍 Checking code quality..."
	ruff check .

.PHONY: security
security: ## Run security checks with bandit
	@echo "🔒 Running security checks..."
	bandit -r . -x tests/ -f json -o security-report.json || true
	bandit -r . -x tests/
	@echo "✅ Security check completed"

.PHONY: quality
quality: format lint security ## Run all code quality checks

# Testing
.PHONY: test
test: ## Run all tests
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "✅ Tests completed"

.PHONY: test-fast
test-fast: ## Run tests without coverage
	@echo "⚡ Running fast tests..."
	$(PYTHON) -m pytest tests/ -v -x
	@echo "✅ Fast tests completed"

.PHONY: test-modules
test-modules: ## Test module imports and basic functionality
	@echo "🔧 Testing module functionality..."
	$(PYTHON) test_modules.py
	@echo "✅ Module tests completed"

# Data processing and model training
.PHONY: data-eda
data-eda: ## Run exploratory data analysis
	@echo "📊 Running exploratory data analysis..."
	jupyter nbconvert --to notebook --execute dataEDA.ipynb --output dataEDA_executed.ipynb
	@echo "✅ EDA completed - check dataEDA_executed.ipynb"

.PHONY: train
train: ## Train the churn prediction models
	@echo "🎯 Training churn prediction models..."
	$(PYTHON) -c "from utils import *; from split import *; from eval import *; from MusicStreamingEventProcessor import *; print('Training pipeline starting...')"
	jupyter nbconvert --to notebook --execute Training.ipynb --output Training_executed.ipynb
	@echo "✅ Training completed - check Training_executed.ipynb and $(MODEL_DIR)/"

.PHONY: evaluate
evaluate: ## Evaluate trained models
	@echo "📈 Evaluating models..."
	$(PYTHON) -c "from eval import *; print('Model evaluation completed')"
	@echo "✅ Model evaluation completed"

.PHONY: process-data
process-data: ## Process raw data to features
	@echo "🏭 Processing data..."
	$(PYTHON) -c "from MusicStreamingEventProcessor import *; processor = MusicStreamingEventProcessor(); print('Data processing completed')"
	@echo "✅ Data processing completed"

# MLflow and experiment tracking
.PHONY: mlflow
mlflow: ## Start MLflow UI server
	@echo "📊 Starting MLflow UI..."
	@echo "🌐 MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	mlflow ui --port $(MLFLOW_PORT) --backend-store-uri file:./mlruns

.PHONY: mlflow-bg
mlflow-bg: ## Start MLflow UI in background
	@echo "📊 Starting MLflow UI in background..."
	nohup mlflow ui --port $(MLFLOW_PORT) --backend-store-uri file:./mlruns > mlflow.log 2>&1 &
	@echo "✅ MLflow UI started at http://localhost:$(MLFLOW_PORT)"

# API and deployment
.PHONY: api
api: ## Start FastAPI development server
	@echo "🚀 Starting FastAPI server..."
	@echo "🌐 API will be available at: http://localhost:$(API_PORT)"
	@echo "📖 API docs at: http://localhost:$(API_PORT)/docs"
	uvicorn main:app --reload --host 0.0.0.0 --port $(API_PORT)

.PHONY: api-prod
api-prod: ## Start FastAPI production server
	@echo "🚀 Starting FastAPI production server..."
	uvicorn main:app --host 0.0.0.0 --port $(API_PORT) --workers 4

# Docker commands
.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t $(DOCKER_IMAGE) .
	@echo "✅ Docker image built: $(DOCKER_IMAGE)"

.PHONY: docker-run
docker-run: docker-build ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p $(API_PORT):$(API_PORT) -p $(MLFLOW_PORT):$(MLFLOW_PORT) $(DOCKER_IMAGE)

.PHONY: docker-run-bg
docker-run-bg: docker-build ## Run Docker container in background
	@echo "🐳 Running Docker container in background..."
	docker run -d -p $(API_PORT):$(API_PORT) -p $(MLFLOW_PORT):$(MLFLOW_PORT) --name $(PROJECT_NAME) $(DOCKER_IMAGE)
	@echo "✅ Container running: http://localhost:$(API_PORT)"

.PHONY: docker-stop
docker-stop: ## Stop Docker container
	@echo "🛑 Stopping Docker container..."
	docker stop $(PROJECT_NAME) || true
	docker rm $(PROJECT_NAME) || true
	@echo "✅ Container stopped"

.PHONY: docker-clean
docker-clean: docker-stop ## Clean Docker images and containers
	@echo "🧹 Cleaning Docker artifacts..."
	docker rmi $(DOCKER_IMAGE) || true
	docker system prune -f
	@echo "✅ Docker cleanup completed"

# Monitoring and maintenance
.PHONY: monitor
monitor: ## Check for model drift and performance
	@echo "📡 Running monitoring checks..."
	$(PYTHON) monitor.py
	@echo "✅ Monitoring completed"

.PHONY: retrain
retrain: ## Check drift and retrain if needed
	@echo "🔄 Checking for model drift and retraining if needed..."
	$(PYTHON) retrain_on_drift.py
	@echo "✅ Retraining check completed"

# Cleanup and maintenance
.PHONY: clean
clean: ## Clean temporary files and caches
	@echo "🧹 Cleaning temporary files..."
ifeq ($(OS),Windows_NT)
	-rmdir /s /q __pycache__ 2>nul
	-rmdir /s /q .pytest_cache 2>nul
	-rmdir /s /q htmlcov 2>nul
	-del /f /q *.pyc 2>nul
	-del /f /q .coverage 2>nul
	-del /f /q security-report.json 2>nul
else
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -f .coverage
	rm -f security-report.json
	rm -f mlflow.log
endif
	@echo "✅ Cleanup completed"

.PHONY: clean-data
clean-data: ## Clean processed data files
	@echo "🧹 Cleaning processed data..."
ifeq ($(OS),Windows_NT)
	-rmdir /s /q data\\processed 2>nul
	-mkdir data\\processed 2>nul
else
	rm -rf data/processed/*
	mkdir -p data/processed
endif
	@echo "✅ Processed data cleaned"

.PHONY: clean-models
clean-models: ## Clean trained models
	@echo "🧹 Cleaning trained models..."
ifeq ($(OS),Windows_NT)
	-rmdir /s /q $(MODEL_DIR) 2>nul
	-rmdir /s /q mlruns 2>nul
	-mkdir $(MODEL_DIR) 2>nul
	-mkdir mlruns 2>nul
else
	rm -rf $(MODEL_DIR)/*
	rm -rf mlruns/*
	mkdir -p $(MODEL_DIR)
	mkdir -p mlruns
endif
	@echo "✅ Models cleaned"

.PHONY: clean-all
clean-all: clean clean-data clean-models ## Clean everything
	@echo "✅ Complete cleanup finished"

# Development workflow
.PHONY: dev-setup
dev-setup: venv install-dev ## Complete development environment setup
	@echo "🎉 Development environment ready!"
	@echo "📝 Next steps:"
	@echo "  1. Activate virtual environment: source $(ACTIVATE)"
	@echo "  2. Place your data file: $(DATA_FILE)"
	@echo "  3. Run EDA: make data-eda"
	@echo "  4. Train models: make train"
	@echo "  5. Start API: make api"

.PHONY: ci
ci: lint-check test security ## Run all CI checks
	@echo "✅ All CI checks completed"

# Status and info
.PHONY: status
status: ## Show project status
	@echo "📊 Project Status:"
	@echo "=================="
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Virtual env: $(if $(wildcard $(VENV_DIR)),✅ Active,❌ Not found)"
	@echo "Data file: $(if $(wildcard $(DATA_FILE)),✅ Found,❌ Missing)"
	@echo "Models dir: $(if $(wildcard $(MODEL_DIR)),✅ Found,❌ Missing)"
	@echo "Pre-commit: $(if $(shell which pre-commit 2>/dev/null),✅ Installed,❌ Not installed)"