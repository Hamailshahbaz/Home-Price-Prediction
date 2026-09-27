# End-to-End House Price Prediction Engine

A production-grade Machine Learning pipeline and REST API designed to predict residential housing market values using engineered spatial, structural, and historical features.

---

## Key Results & Performance

* **Best Model:** Tuned XGBoost Regressor
* **Log-Price $R^2$ Score:** **0.8082**
* **Log-Price RMSE:** **0.2316**
* **Real Dollar MAE:** **$89,166.04 USD** (Tested across 910 held-out properties)

---

## Project Architecture

```
house-price-prediction/
├── data/                  # Raw and feature-engineered datasets (Git-ignored raw data)
├── notebooks/             # Exploratory Data Analysis & Model Experiments
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/                   # Modularized Python Core Package
│   ├── data_pipeline.py    # Raw data cleaning & transformation
│   ├── feature_engineering.py # Age, room ratio, & location encodings
│   ├── models.py          # Model training, CV & serialization
│   ├── evaluate.py        # Evaluation metrics & visual plots
│   └── train.py           # End-to-end training pipeline runner
├── api/                   # Production Web Service
│   ├── main.py            # FastAPI endpoints
│   └── schemas.py         # Pydantic data schemas
├── app/                   # Web Dashboard
│   └── streamlit_app.py   # Interactive user interface
├── Dockerfile             # Container configuration
└── requirements.txt
```

---

## Quickstart Guide

### 1. Local Setup

```bash
# Clone repository
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model Pipeline

Run the full end-to-end pipeline (ingestion → cleaning → feature engineering → training → evaluation):

```bash
python src/train.py
```

### 3. Launch REST API Server

```bash
uvicorn api.main:app --reload --port 8000
```

* Interactive OpenAPI Swagger Docs: http://localhost:8000/docs

### 4. Launch Streamlit UI

In a new terminal window:

```bash
streamlit run app/streamlit_app.py
```

### Docker Deployment

Build and launch the containerized application using Docker:

```bash
# Build image
docker build -t house-price-api .

# Run container
docker run -p 8000:8000 house-price-api
```