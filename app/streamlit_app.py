import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="House Valuation Engine",
    layout="wide"
)

st.title("Real Estate Price Valuation Engine")
st.markdown("Predict house prices using an end-to-end Machine Learning pipeline powered by **XGBoost**.")

st.sidebar.header("Input Property Details")

# Form Inputs
city = st.sidebar.text_input("City", value="Seattle")
statezip = st.sidebar.text_input("State & Zip", value="WA 98103")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Spatial & Physical Characteristics")
    sqft_living = st.number_input("Living Area (sqft)", min_value=300, max_value=15000, value=2100, step=50)
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=100000, value=7500, step=100)
    sqft_above = st.number_input("Above Ground (sqft)", min_value=300, max_value=10000, value=1600, step=50)
    sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, max_value=5000, value=500, step=50)
    floors = st.selectbox("Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=2)

with col2:
    st.subheader("Rooms & Property Condition")
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("Bathrooms", min_value=0.5, max_value=8.0, value=2.25, step=0.25)
    condition = st.slider("Condition Rating (1-5)", min_value=1, max_value=5, value=3)
    view = st.slider("View Quality Rating (0-4)", min_value=0, max_value=4, value=0)
    waterfront = st.selectbox("Waterfront Property?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    yr_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=1985)
    yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2026, value=2010)

st.markdown("---")

# Load artifacts directly in Streamlit
MODEL_PATH = os.path.join("models", "xgboost_housing_model.pkl")
FEATURES_PATH = os.path.join("models", "model_features.pkl")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_housing_data.csv")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    
    city_price_map, zip_price_map, zip_count_map = {}, {}, {}
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        city_price_map = df.groupby('city')['price'].median().to_dict()
        zip_price_map = df.groupby('statezip')['price'].median().to_dict()
        zip_count_map = df.groupby('statezip')['statezip'].count().to_dict()
        
    return model, model_features, city_price_map, zip_price_map, zip_count_map

model, model_features, city_price_map, zip_price_map, zip_count_map = load_artifacts()

if st.button("Calculate Estimated Price", use_container_width=True):
    with st.spinner("Calculating valuation..."):
        current_year = 2026
        sale_year, sale_month = current_year, 6
        house_age = max(0, sale_year - yr_built)
        is_renovated = 1 if yr_renovated > 0 else 0
        years_since_renovation = max(0, sale_year - yr_renovated) if is_renovated else house_age
        total_rooms = bedrooms + bathrooms
        bed_bath_ratio = bedrooms / (bathrooms + 0.001)
        sqft_per_room = sqft_living / (total_rooms + 0.001)
        living_to_lot_ratio = sqft_living / sqft_lot
        basement_ratio = sqft_basement / sqft_living
        has_basement = 1 if sqft_basement > 0 else 0

        global_median = 460000.0
        city_med = city_price_map.get(city, global_median)
        zip_med = zip_price_map.get(statezip, global_median)
        zip_cnt = zip_count_map.get(statezip, 50)

        input_dict = {
            'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
            'sqft_lot': sqft_lot, 'floors': floors, 'waterfront': waterfront, 'view': view,
            'condition': condition, 'sqft_above': sqft_above, 'sqft_basement': sqft_basement,
            'yr_built': yr_built, 'yr_renovated': yr_renovated, 'sale_year': sale_year,
            'sale_month': sale_month, 'house_age': house_age, 'is_renovated': is_renovated,
            'years_since_renovation': years_since_renovation, 'total_rooms': total_rooms,
            'bed_bath_ratio': bed_bath_ratio, 'sqft_per_room': sqft_per_room,
            'living_to_lot_ratio': living_to_lot_ratio, 'basement_ratio': basement_ratio,
            'has_basement': has_basement, 'city_median_price': city_med,
            'zip_median_price': zip_med, 'zip_property_count': zip_cnt
        }

        input_df = pd.DataFrame([input_dict])[model_features]
        log_pred = float(model.predict(input_df)[0])
        real_pred_usd = float(np.expm1(log_pred))

        st.success(f"### Estimated Market Valuation: **${real_pred_usd:,.2f} USD**")
        
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Log Price Scale", f"{log_pred:.4f}")
        metric_col2.metric("Location Reference", f"{city}, {statezip}")