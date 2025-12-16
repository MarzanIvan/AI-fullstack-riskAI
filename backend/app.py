# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
import httpx
import os


ML_URL = os.getenv("ML_URL", "http://ml_service:9000") # access to datasets and models
AI_URL = os.getenv("AI_URL", "http://predict_service:7000") # access to models

app = FastAPI(title="api")

"""
    AI MODULE
    communication to http://predict_service:7000
"""
class Loan(BaseModel):
    loan_id: str
    loan_type: str # consumer | credit_card | mortgage | etc
    open_date: date
    close_date: Optional[date]
    loan_amount: float
    remaining_balance: float
    current_status: str # active | closed | overdue
    max_delay_days: int

class CreditRiskInput(BaseModel):
    full_name: str
    birth_date: date
    citizenship: str
    capacity_status: str # дееспособен / ограничен
    #income
    monthly_income: float
    income_sources: List[str]
    #credit history
    loans: List[Loan]
    #loan request
    requested_loan_amount: float
    requested_loan_type: str

@app.post("/analysis_credit/")
async def credit_risk(data: CreditRiskInput):
    return await call_predict("predict/credit", data.dict())

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

@app.post("/analysis_investment/")
async def investment_risk(data: InvestmentRiskInput):
    return await call_predict("predict/investment", data.dict())

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

@app.post("/analysis_insurance/")
async def insurance_risk(data: InsuranceRiskInput):
    return await call_predict("predict/insurance", data.dict())

async def call_predict(route: str, payload: dict):
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.post(f"{AI_URL}/{route}", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

"""
    ML MODULE
    communication to http://ml_service:9000
"""
class TrainRequest(BaseModel):
    dataset_name: str
    dataset_path: str


async def call_training(route: str, payload: dict):
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            response = await client.post(
                f"{ML_URL}/{route}",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml_credit")
async def train_credit(req: TrainRequest):
    return await call_training("train/credit", req.dict())


@app.post("/ml_investment")
async def train_investment(req: TrainRequest):
    return await call_training("train/investment", req.dict())


@app.post("/ml_insurance")
async def train_insurance(req: TrainRequest):
    return await call_training("train/insurance", req.dict())


@app.get("/health")
def health():
    return {"status": "ok", "service": "backend"}
