# api/main.py
import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status

# Ensure root directory is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.schemas import HousePredictionInput, PredictionResponse

app = FastAPI(
    title="House Price Prediction API",
    description="Production REST API serving XGBoost house price estimation models.",
    version="1.0.0"
)

# Paths to trained model artifacts
MODEL_PATH = os.path.join("models", "xgboost_housing_model.pkl")
FEATURES_PATH = os.path.join("models", "model_features.pkl")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_housing_data.csv")

# Global variables to cache model and lookup maps in memory
model = None
model_features = []
city_price_map = {}
zip_price_map = {}
zip_count_map = {}


@app.on_event("startup")
def load_artifacts():
    """Loads model artifacts and location lookup maps into memory on app startup."""
    global model, model_features, city_price_map, zip_price_map, zip_count_map

    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        raise RuntimeError(f"Artifacts not found in models/ directory. Run `python src/train.py` first.")

    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)

    # Build location mapping lookups from processed dataset
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        city_price_map = df.groupby('city')['price'].median().to_dict()
        zip_price_map = df.groupby('statezip')['price'].median().to_dict()
        zip_count_map = df.groupby('statezip')['statezip'].count().to_dict()


@app.get("/", tags=["Health Check"])
def health_check():
    """Health check endpoint."""
    return {"status": "online", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_house_price(payload: HousePredictionInput):
    """
    Accepts raw property features, transforms them into required model input schema,
    and returns estimated dollar valuation.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is not initialized."
        )

    try:
        # Convert payload to dictionary
        data = payload.dict()
        current_year = 2026

        # Compute engineered features
        sale_year = current_year
        sale_month = 6
        house_age = max(0, sale_year - data['yr_built'])
        is_renovated = 1 if data['yr_renovated'] > 0 else 0
        years_since_renovation = max(0, sale_year - data['yr_renovated']) if is_renovated else house_age

        total_rooms = data['bedrooms'] + data['bathrooms']
        bed_bath_ratio = data['bedrooms'] / (data['bathrooms'] + 0.001)
        sqft_per_room = data['sqft_living'] / (total_rooms + 0.001)
        living_to_lot_ratio = data['sqft_living'] / data['sqft_lot']
        basement_ratio = data['sqft_basement'] / data['sqft_living']
        has_basement = 1 if data['sqft_basement'] > 0 else 0

        # Map location aggregations with global median fallbacks
        global_median = 460000.0
        city_med = city_price_map.get(data['city'], global_median)
        zip_med = zip_price_map.get(data['statezip'], global_median)
        zip_cnt = zip_count_map.get(data['statezip'], 50)

        # Assemble full input record matching exact training columns
        input_dict = {
            'bedrooms': data['bedrooms'],
            'bathrooms': data['bathrooms'],
            'sqft_living': data['sqft_living'],
            'sqft_lot': data['sqft_lot'],
            'floors': data['floors'],
            'waterfront': data['waterfront'],
            'view': data['view'],
            'condition': data['condition'],
            'sqft_above': data['sqft_above'],
            'sqft_basement': data['sqft_basement'],
            'yr_built': data['yr_built'],
            'yr_renovated': data['yr_renovated'],
            'sale_year': sale_year,
            'sale_month': sale_month,
            'house_age': house_age,
            'is_renovated': is_renovated,
            'years_since_renovation': years_since_renovation,
            'total_rooms': total_rooms,
            'bed_bath_ratio': bed_bath_ratio,
            'sqft_per_room': sqft_per_room,
            'living_to_lot_ratio': living_to_lot_ratio,
            'basement_ratio': basement_ratio,
            'has_basement': has_basement,
            'city_median_price': city_med,
            'zip_median_price': zip_med,
            'zip_property_count': zip_cnt
        }

        # Convert to DataFrame with exact column ordering
        input_df = pd.DataFrame([input_dict])[model_features]

        # Predict log price and transform to real USD
        log_pred = float(model.predict(input_df)[0])
        real_pred_usd = float(np.expm1(log_pred))

        return PredictionResponse(
            predicted_price_usd=round(real_pred_usd, 2),
            log_price_prediction=round(log_pred, 4),
            city=data['city'],
            statezip=data['statezip']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")