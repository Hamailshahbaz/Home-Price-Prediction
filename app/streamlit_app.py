# app/streamlit_app.py
import streamlit as st
import requests
import json

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
    st.subheader("🛋️ Rooms & Property Condition")
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("Bathrooms", min_value=0.5, max_value=8.0, value=2.25, step=0.25)
    condition = st.slider("Condition Rating (1-5)", min_value=1, max_value=5, value=3)
    view = st.slider("View Quality Rating (0-4)", min_value=0, max_value=4, value=0)
    waterfront = st.selectbox("Waterfront Property?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    yr_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=1985)
    yr_renovated = st.number_input("Year Renovated (0 if never)", min_value=0, max_value=2026, value=2010)

st.markdown("---")

if st.button("Calculate Estimated Price", use_container_width=True):
    payload = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "city": city,
        "statezip": statezip
    }

    API_URL = "http://localhost:8000/predict"
    
    with st.spinner("Calculating valuation..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            if response.status_code == 200:
                res_data = response.json()
                price = res_data["predicted_price_usd"]
                
                st.balloons()
                st.success(f"### Estimated Market Valuation: **${price:,.2f} USD**")
                
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Log Price Scale", res_data["log_price_prediction"])
                metric_col2.metric("Location Reference", f"{res_data['city']}, {res_data['statezip']}")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to FastAPI service at `{API_URL}`. Make sure the API server is running!\n\nError: {e}")