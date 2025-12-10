# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os

ML_URL = os.getenv("ML_URL", "http://ml_service:9000")

app = FastAPI(title="api")

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

async def call_ml(route: str, payload: dict):
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.post(f"{ML_URL}/{route}", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk/credit")
async def credit_risk(data: CreditRiskInput):
    return await call_ml("predict/credit", data.dict())

@app.post("/risk/investment")
async def investment_risk(data: InvestmentRiskInput):
    return await call_ml("predict/investment", data.dict())

@app.post("/risk/insurance")
async def insurance_risk(data: InsuranceRiskInput):
    return await call_ml("predict/insurance", data.dict())

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "service": "backend"}
