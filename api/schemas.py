# api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional


class HousePredictionInput(BaseModel):
    bedrooms: float = Field(..., ge=0, le=15, description="Number of bedrooms", example=3)
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms", example=2.25)
    sqft_living: int = Field(..., gt=0, description="Living area square footage", example=2100)
    sqft_lot: int = Field(..., gt=0, description="Total lot square footage", example=7500)
    floors: float = Field(..., ge=1.0, le=4.0, description="Number of floors", example=1.5)
    waterfront: int = Field(0, ge=0, le=1, description="Waterfront property (0 or 1)", example=0)
    view: int = Field(0, ge=0, le=4, description="View rating (0 to 4)", example=0)
    condition: int = Field(3, ge=1, le=5, description="Condition rating (1 to 5)", example=3)
    sqft_above: int = Field(..., gt=0, description="Square footage above ground", example=1600)
    sqft_basement: int = Field(0, ge=0, description="Square footage of basement", example=500)
    yr_built: int = Field(..., ge=1800, le=2026, description="Year built", example=1985)
    yr_renovated: int = Field(0, ge=0, le=2026, description="Year renovated (0 if never)", example=2010)
    city: str = Field(..., description="City name", example="Seattle")
    statezip: str = Field(..., description="State Zip code", example="WA 98103")


class PredictionResponse(BaseModel):
    predicted_price_usd: float = Field(..., description="Estimated home price in USD ($)")
    log_price_prediction: float = Field(..., description="Model target prediction on log scale")
    city: str
    statezip: str
    status: str = "success"