# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

ML_URL = os.getenv("ML_URL", "http://ml_service:9000")

app = FastAPI(title="api")

class CreditRiskInput(BaseModel):
    income: float
    debt: float
    age: int
    credit_score: int

class InvestmentRiskInput(BaseModel):
    asset_volatility: float
    expected_return: float
    horizon_days: int
    portfolio_value: float

class InsuranceRiskInput(BaseModel):
    claim_history: int
    age: int
    vehicle_value: float
    region_code: str

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
