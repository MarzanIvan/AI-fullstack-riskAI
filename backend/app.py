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

class CreditRiskInput(BaseModel):
    education: str
    sex: str
    age: int
    car: str
    car_type: str
    decline_app_cnt: int
    good_work: int
    score_bki: float
    bki_request_cnt: int
    region_rating: int
    home_address: int
    work_address: int
    income: float
    sna: int
    first_time: int
    foreign_passport: str
    app_date: str  # "01FEB2014"

class CreditPredictResponse(BaseModel):
    pd: float                   # Probability of Default
    default_label: int           # 0 / 1
    risk_level: str              # LOW / MEDIUM / HIGH

@app.post("/analysis_credit/",response_model=CreditPredictResponse)
async def credit_risk(data: CreditRiskInput):
    return await call_predict(
        "predict/credit",
        data.dict()
    )


class InsuranceRiskInput(BaseModel):
    months_as_customer: int
    age: int
    policy_deductable: int
    policy_annual_premium: float
    umbrella_limit: int

    incident_hour_of_the_day: int
    number_of_vehicles_involved: int
    bodily_injuries: int
    witnesses: int

    injury_claim: float
    property_claim: float
    vehicle_claim: float

    policy_state: str
    insured_sex: str
    insured_education_level: str
    incident_type: str
    collision_type: str
    incident_severity: str
    authorities_contacted: str
    property_damage: str
    police_report_available: str
    auto_make: str

class InsurancePredictResponse(BaseModel):
    fraud_probability: float
    fraud_label: int
    risk_level: str

@app.post("/analysis_insurance/",response_model=InsurancePredictResponse)
async def insurance_risk(data: InsuranceRiskInput):
    return await call_predict(
        "predict/insurance",
        data.dict()
    )


class InvestmentRiskInput(BaseModel):
    # Данные о компании
    company_name: str
    company_category_list: Optional[str]
    company_market: Optional[str]
    company_country_code: Optional[str]
    company_state_code: Optional[str]
    company_region: Optional[str]
    company_city: Optional[str]

    # Данные о раунде инвестиций
    funding_round_type: Optional[str]
    funding_round_code: Optional[str]
    funded_at: Optional[str]  # "DD/MM/YYYY"
    funded_month: Optional[str]  # "YYYY-MM"
    funded_quarter: Optional[str]  # "YYYY-QX"
    funded_year: Optional[int]
    raised_amount_usd: Optional[float]

class InvestmentPredictResponse(BaseModel):
    predicted_next_investment: float
    expected_portfolio_return: Optional[float] = None
    risk_score: Optional[float] = None

@app.post("/analysis_investment/",response_model=InvestmentPredictResponse)
async def investment_risk(data: InvestmentRiskInput):
    return await call_predict(
        "predict/investment",
        data.dict()
    )


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
