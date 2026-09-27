import os
import pandas as pd
import numpy as np


def add_time_and_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates sale year, house age, renovation status, and years since renovation."""
    df_feat = df.copy()

    # Extract transaction time details
    df_feat['sale_year'] = df_feat['date'].dt.year
    df_feat['sale_month'] = df_feat['date'].dt.month

    # House age at time of sale
    df_feat['house_age'] = (df_feat['sale_year'] - df_feat['yr_built']).clip(lower=0)

    # Renovation status & age
    df_feat['is_renovated'] = (df_feat['yr_renovated'] > 0).astype(int)
    df_feat['years_since_renovation'] = np.where(
        df_feat['is_renovated'] == 1,
        df_feat['sale_year'] - df_feat['yr_renovated'],
        df_feat['house_age']
    )
    df_feat['years_since_renovation'] = df_feat['years_since_renovation'].clip(lower=0)

    return df_feat


def add_structural_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Creates room ratios, density metrics, and basement indicators."""
    df_feat = df.copy()

    # Room totals and bed-to-bath ratio
    df_feat['total_rooms'] = df_feat['bedrooms'] + df_feat['bathrooms']
    df_feat['bed_bath_ratio'] = df_feat['bedrooms'] / (df_feat['bathrooms'] + 0.001)

    # Living space metrics
    df_feat['sqft_per_room'] = df_feat['sqft_living'] / (df_feat['total_rooms'] + 0.001)
    df_feat['living_to_lot_ratio'] = df_feat['sqft_living'] / df_feat['sqft_lot']

    # Basement metrics
    df_feat['basement_ratio'] = df_feat['sqft_basement'] / df_feat['sqft_living']
    df_feat['has_basement'] = (df_feat['sqft_basement'] > 0).astype(int)

    return df_feat


def add_location_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes city and zip code features based on median transaction prices."""
    df_feat = df.copy()

    # Map median prices and property counts
    city_price_map = df_feat.groupby('city')['price'].median().to_dict()
    zip_price_map = df_feat.groupby('statezip')['price'].median().to_dict()
    zip_count_map = df_feat.groupby('statezip')['statezip'].count().to_dict()

    df_feat['city_median_price'] = df_feat['city'].map(city_price_map)
    df_feat['zip_median_price'] = df_feat['statezip'].map(zip_price_map)
    df_feat['zip_property_count'] = df_feat['statezip'].map(zip_count_map)

    return df_feat


def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the full feature engineering pipeline sequentially and 
    creates the log_price target variable.
    """
    df_proc = df.copy()

    # 1. Log target transformation
    if 'price' in df_proc.columns:
        df_proc['log_price'] = np.log1p(df_proc['price'])

    # 2. Sequential feature transformations
    df_proc = add_time_and_age_features(df_proc)
    df_proc = add_structural_ratios(df_proc)
    df_proc = add_location_encodings(df_proc)

    return df_proc


if __name__ == "__main__":
    from data_pipeline import load_raw_data, clean_raw_data

    # Test full end-to-end data processing locally
    RAW_PATH = "../data/data.csv"
    OUTPUT_PATH = "../data/processed/cleaned_housing_data.csv"

    print("Loading raw data...")
    raw_df = load_raw_data(RAW_PATH)
    
    print("Cleaning data...")
    cleaned_df = clean_raw_data(raw_df)
    
    print("Engineering features...")
    processed_df = build_feature_pipeline(cleaned_df)

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    processed_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Feature Pipeline Test Successful!")
    print(f"Processed shape: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
    print(f"Saved processed dataset to: {OUTPUT_PATH}")