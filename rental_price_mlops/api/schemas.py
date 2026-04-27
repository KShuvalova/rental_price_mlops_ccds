from pydantic import BaseModel, Field
from typing import Optional, Literal


class PredictionRequest(BaseModel):
    neighbourhood_group: str
    neighbourhood: str
    latitude: float
    longitude: float
    room_type: Literal["Entire home/apt", "Private room", "Shared room"]
    minimum_nights: int = Field(ge=1)
    number_of_reviews: int = Field(ge=0)
    reviews_per_month: float = Field(ge=0)
    calculated_host_listings_count: int = Field(ge=1)
    availability_365: int = Field(ge=0, le=365)
    days_since_last_review: float = -1
    has_last_review: int = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_log_price: float
    predicted_price: float
    model_name: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_path: str
    target: str
    features_expected: list[str]


class RetrainResponse(BaseModel):
    status: str
    message: str


class PredictionLogItem(BaseModel):
    timestamp: str
    request_data: dict
    predicted_log_price: float
    predicted_price: float
    model_name: str
    model_version: str


class LatestMetricsResponse(BaseModel):
    source: str
    metrics: dict