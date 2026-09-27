import os
import pandas as pd
import numpy as np


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Loads raw CSV data from the specified path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
    return pd.read_csv(data_path)


def clean_raw_data(df: pd.DataFrame, max_price: float = 5_000_000) -> pd.DataFrame:
    """
    Cleans raw dataset by removing invalid records and outliers.
    
    Cleaning Steps:
    1. Filter out zero prices and extreme high-end outliers (> max_price).
    2. Filter out invalid properties with 0 bedrooms AND 0 bathrooms.
    3. Parse date strings into datetime objects.
    """
    df_clean = df.copy()

    # 1. Price filtering
    valid_price_mask = (df_clean['price'] > 0) & (df_clean['price'] < max_price)

    # 2. Room validation
    valid_rooms_mask = ~((df_clean['bedrooms'] == 0) & (df_clean['bathrooms'] == 0))

    # Apply filters
    df_clean = df_clean[valid_price_mask & valid_rooms_mask].copy()

    # 3. Parse date
    df_clean['date'] = pd.to_datetime(df_clean['date'])

    return df_clean


if __name__ == "__main__":
    # Test script execution locally
    RAW_PATH = "../data/data.csv"
    raw_df = load_raw_data(RAW_PATH)
    clean_df = clean_raw_data(raw_df)
    print(f"✓ Data Pipeline Test Successful! Cleaned shape: {clean_df.shape}")