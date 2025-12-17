import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import Optional
from s3_client import S3Client

app = FastAPI(title="predict")

S3_BUCKET = os.getenv("S3_BUCKET", "risk-model-storage")

OUTPUT_MODEL_CREDIT = os.getenv(
    "MODEL_CREDIT_PATH",
    "models/credit/credit.joblib"
)

OUTPUT_MODEL_INVEST = os.getenv(
    "MODEL_INVEST_PATH",
    "models/invest/invest.h5"
)

OUTPUT_MODEL_INSURANCE = os.getenv(
    "MODEL_INSURANCE_PATH",
    "models/insurance/insurance.joblib"
)

LOCAL_MODEL_DIR = "models"

s3 = S3Client(bucket=S3_BUCKET)

credit_model = None
investment_model = None
insurance_model = None

def load_model_from_s3(s3_key: str, loader: str):
    local_path = os.path.join(LOCAL_MODEL_DIR, s3_key)

    s3.download_model(
        key=s3_key,
        local_path=local_path
    )

    if loader == "joblib":
        return joblib.load(local_path)

    if loader == "keras":
        return load_model(local_path)

    raise ValueError(f"Unsupported loader type: {loader}")


@app.on_event("startup")
def load_all_models():
    global credit_model, investment_model, insurance_model
    try:
        credit_model = load_model_from_s3(
            OUTPUT_MODEL_CREDIT,
            loader="joblib"
        )
        investment_model = load_model_from_s3(
            OUTPUT_MODEL_INVEST,
            loader="keras"
        )
        insurance_model = load_model_from_s3(
            OUTPUT_MODEL_INSURANCE,
            loader="joblib"
        )
    except Exception as e:
        raise RuntimeError(f"model loading failed: {e}")


class CreditPredictRequest(BaseModel):
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
    app_date: str


class CreditPredictResponse(BaseModel):
    pd: float
    default_label: int
    risk_level: str

@app.post(
    "/predict/credit",
    response_model=CreditPredictResponse
)
def predict_credit(req: CreditPredictRequest):

    if credit_model is None:
        raise HTTPException(
            status_code=503,
            detail="Credit model not loaded"
        )

    try:
        X = pd.DataFrame([req.dict()])
        pd_default = credit_model.predict_proba(X)[0, 1]
        default_label = int(pd_default >= 0.5)

        if pd_default < 0.2:
            risk_level = "LOW"
        elif pd_default < 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return CreditPredictResponse(
            pd=round(float(pd_default), 4),
            default_label=default_label,
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

class InsurancePredictRequest(BaseModel):
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

@app.post(
    "/predict/insurance",
    response_model=InsurancePredictResponse
)
def predict_insurance(req: InsurancePredictRequest):
    if insurance_model is None:
        raise HTTPException(
            status_code=503,
            detail="Insurance model not loaded"
        )
    try:
        X = pd.DataFrame([req.dict()])
        fraud_prob = insurance_model.predict_proba(X)[0, 1]
        fraud_label = int(fraud_prob >= 0.5)
        if fraud_prob < 0.3:
            risk_level = "LOW"
        elif fraud_prob < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return InsurancePredictResponse(
            fraud_probability=round(float(fraud_prob), 4),
            fraud_label=fraud_label,
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Insurance prediction error: {str(e)}"
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


@app.post("/predict/investment", response_model=InvestmentPredictResponse)
def predict_investment(req: InvestmentRiskInput):
    if investment_model is None:
        raise HTTPException(status_code=503, detail="investment model not loaded")
    try:
        # Используем raised_amount_usd как вход модели
        seq = np.array([[req.raised_amount_usd or 0]])
        seq = seq.reshape(1, -1, 1)  # (batch=1, timesteps=1, features=1)

        pred = investment_model.predict(seq)[0, 0]
        risk_score = float(np.std(seq))
        expected_portfolio_return = float(pred)  # можно уточнить логику

        return InvestmentPredictResponse(
            predicted_next_investment=float(pred),
            expected_portfolio_return=expected_portfolio_return,
            risk_score=risk_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")