# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os

ML_URL = os.getenv("ML_URL", "http://ml_service:8000") # access to datasets and modules
AI_URL = os.getenv("AI_URL", "http://predict_service:7000") # access to modules

app = FastAPI(title="api")

"""
    AI MODULE
    communication to http://predict_service:7000
"""

class CreditRiskInput(BaseModel):
    income: float
    debt: float
    age: int
    credit_score: int

# --- Investment Risk ---
class InvestmentRiskInput(BaseModel):
    portfolio_value: float
    asset_weights: List[float]
    asset_types: List[str]
    asset_volatility: List[float]
    expected_return: List[float]
    horizon_days: int
    risk_free_rate: float
    correlation_matrix: List[List[float]]
    leverage_ratio: Optional[float] = 1.0
    liquidity_constraints: Optional[str] = "No restrictions"
    stop_loss_limits: Optional[List[float]] = None
    historical_prices: Optional[List[List[float]]] = None

# --- Insurance Risk ---
class VehicleCharacteristics(BaseModel):
    make: str
    model: str
    year_of_manufacture: int
    vehicle_value: float
    engine_power: int
    body_type: str
    anti_theft_systems: Optional[List[str]] = None
    tuning: Optional[bool] = False

class DriverParameters(BaseModel):
    driver_age: int
    driving_experience: int
    driver_gender: Optional[str] = None
    region_registration: str
    additional_drivers: Optional[int] = 0
    marital_status: Optional[str] = None

class InsuranceRiskInput(BaseModel):
    vehicle: VehicleCharacteristics
    driver: DriverParameters

@app.post("/analysis_credit/")
async def credit_risk(data: CreditRiskInput):
    return await call_predict("predict/credit", data.dict())

@app.post("/analysis_investment/")
async def investment_risk(data: InvestmentRiskInput):
    return await call_predict("predict/investment", data.dict())

@app.post("/analysis_insurance/")
async def insurance_risk(data: InsuranceRiskInput):
    return await call_predict("predict/insurance", data.dict())

"""
    ML MODULE
    communication to http://ml_service:8000
"""
async def call_predict(route: str, payload: dict):
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.post(f"{AI_URL}/{route}", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


async def call_training(route: str):
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            response = await client.post(f"{ML_URL}/{route}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml_credit")
async def train_credit():
    return await call_training("train/credit")


@app.post("/ml_investment")
async def train_investment():
    return await call_training("train/investment")


@app.post("/ml_insurance")
async def train_insurance():
    return await call_training("train/insurance")

@app.get("/health")
def health():
    return {"status": "ok", "service": "backend"}
