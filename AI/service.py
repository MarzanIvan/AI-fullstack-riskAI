import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import Optional
import traceback

import boto3
from botocore.client import Config

app = FastAPI(title="predict")


class S3Client:
    def __init__(
        self,
        bucket: str,
        endpoint: Optional[str] = "https://storage.yandexcloud.net",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.bucket = bucket

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=aws_access_key_id or os.getenv("YANDEX_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.getenv("YANDEX_SECRET_ACCESS_KEY"),
            config=Config(signature_version="s3v4"),
            region_name="ru-central1"
        )
    def read_json(self, key: str) -> pd.DataFrame:
        """Read JSON file """
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_json(io.BytesIO(obj["Body"].read()))

    def read_parquet(self, key: str) -> pd.DataFrame:
        """Read PARQUET file """
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    def download_model(self, key: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket, key, local_path)
        return local_path

    def upload_model(self, local_path: str, key: str):
        self.s3.upload_file(local_path, self.bucket, key)

    def upload_file(self, local_path: str, key: str):
        """Upload any file to S3."""
        self.s3.upload_file(local_path, self.bucket, key)

    def download_file(self, key: str, local_path: str):
        """Download any file """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket, key, local_path)

    def list_files(self, prefix: str = ""):
        """List objects in bucket."""
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        if "Contents" not in response:
            return []
        return [item["Key"] for item in response["Contents"]]

OUTPUT_MODEL_CREDIT = os.getenv("MODEL_CREDIT_PATH", "models/credit/credit.joblib") # joblib from sklearn
OUTPUT_MODEL_INVEST = os.getenv("MODEL_INVEST_PATH", "models/invest/invest.h5") # h5 models from keras and tensorflow
OUTPUT_MODEL_INSURANCE = os.getenv("MODEL_INSURANCE_PATH", "models/insurance/insurance.joblib") # joblib from sklearn
S3_BUCKET = os.getenv("S3_BUCKET", "risk-model-storage")

LOCAL_MODEL_DIR = "models"

s3 = S3Client(
    bucket=S3_BUCKET,
    aws_access_key_id=os.getenv("YANDEX_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("YANDEX_SECRET_ACCESS_KEY"),
)


def get_or_load_model(
    model_ref: dict,
    s3_key: str,
    loader: str
):
    if model_ref["model"] is not None:
        return model_ref["model"]

    try:
        local_path = os.path.join(LOCAL_MODEL_DIR, s3_key)

        if not os.path.exists(local_path):
            s3.download_model(
                key=s3_key,
                local_path=local_path
            )

        if loader == "joblib":
            model_ref["model"] = joblib.load(local_path)
        elif loader == "keras":
            model_ref["model"] = load_model(local_path)
        else:
            raise ValueError(f"Unsupported loader type: {loader}")

        return model_ref["model"]

    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")


credit_model_ref = {"model": None}
insurance_model_ref = {"model": None}
investment_model_ref = {"model": None}

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

@app.post("/predict/credit", response_model=CreditPredictResponse)
def predict_credit(req: CreditPredictRequest):
    try:
        model = get_or_load_model(
            model_ref=credit_model_ref,
            s3_key=OUTPUT_MODEL_CREDIT,
            loader="joblib"
        )
        X = pd.DataFrame([req.dict()])
        pd_default = model.predict_proba(X)[0, 1]
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
            detail=str(e)
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
@app.post("/predict/insurance", response_model=InsurancePredictResponse)
def predict_insurance(req: InsurancePredictRequest):
    try:
        model = get_or_load_model(
            model_ref=insurance_model_ref,
            s3_key=OUTPUT_MODEL_INSURANCE,
            loader="joblib"
        )
        X = pd.DataFrame([req.dict()])
        fraud_prob = model.predict_proba(X)[0, 1]
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
            detail=str(e)
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
    try:
        model = get_or_load_model(
            model_ref=investment_model_ref,
            s3_key=OUTPUT_MODEL_INVEST,
            loader="keras"
        )
        seq = np.array([[req.raised_amount_usd or 0]])
        seq = seq.reshape(1, -1, 1)  # (batch=1, timesteps=1, features=1)

        pred = model.predict(seq)[0, 0]
        risk_score = float(np.std(seq))
        expected_portfolio_return = float(pred)  # можно уточнить логику

        return InvestmentPredictResponse(
            predicted_next_investment=float(pred),
            expected_portfolio_return=expected_portfolio_return,
            risk_score=risk_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
