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


from sklearn.base import BaseEstimator, TransformerMixin

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.region_income_mean_ = (
            X.groupby("region_rating")["income"].mean().to_dict()
        )
        self.age_income_mean_ = (
            X.groupby("age")["income"].mean().to_dict()
        )
        self.age_bki_mean_ = (
            X.groupby("age")["score_bki"].mean().to_dict()
        )
        return self

    def transform(self, X):
        df = X.copy()

        df["income_to_request"] = df["income"] / (df["bki_request_cnt"] + 1)

        df["mean_income_region"] = df["region_rating"].map(
            self.region_income_mean_
        ).fillna(df["income"])

        df["mean_income_age"] = df["age"].map(
            self.age_income_mean_
        ).fillna(df["income"])

        df["mean_bki_age"] = df["age"].map(
            self.age_bki_mean_
        ).fillna(df["score_bki"])

        df["sex"] = df["sex"].map({"M": 1, "F": 0}).fillna(0)
        df["good_work"] = df["good_work"].astype(int)
        df["first_time"] = df["first_time"].astype(int)

        return df


def get_or_load_model(model_ref, local_path: str, model_type: str):
    if model_ref["model"] is not None:
        return model_ref["model"]

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        s3.download_file(local_path, local_path)

    if model_type == "keras":
        model_ref["model"] = load_model(local_path, compile=False)
    else:
        model_ref["model"] = joblib.load(local_path)

    return model_ref["model"]

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
            credit_model_ref,
            OUTPUT_MODEL_CREDIT,
            "joblib"
        )
        data = req.dict()
        data.pop("app_date", None)
        X = pd.DataFrame([data])
        pd_default = float(model.predict_proba(X)[0, 1])
        default_label = int(pd_default >= 0.5)
        risk_level = (
            "LOW" if pd_default < 0.2
            else "MEDIUM" if pd_default < 0.5
            else "HIGH"
        )

        return CreditPredictResponse(
            pd=round(pd_default, 4),
            default_label=default_label,
            risk_level=risk_level
        )

    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Credit prediction failed")



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
            insurance_model_ref,
            OUTPUT_MODEL_INSURANCE,
            "joblib"
        )

        X = pd.DataFrame([req.dict()])
        fraud_prob = float(model.predict_proba(X)[0, 1])
        fraud_label = int(fraud_prob >= 0.5)

        risk_level = (
            "LOW" if fraud_prob < 0.3
            else "MEDIUM" if fraud_prob < 0.6
            else "HIGH"
        )

        return InsurancePredictResponse(
            fraud_probability=round(fraud_prob, 4),
            fraud_label=fraud_label,
            risk_level=risk_level
        )

    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Insurance prediction failed")


SEQ_LENGTH = 20


class InvestmentRiskInpu(BaseModel):
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


class InvestmentPredictRespons(BaseModel):
    predicted_next_investment: float
    expected_portfolio_return: Optional[float] = None
    risk_score: Optional[float] = None

SEQ_LENGTH = 20

class InvestmentRiskInput(BaseModel):
    company_name: str
    company_category_list: Optional[str] = None
    company_market: Optional[str] = None
    company_country_code: Optional[str] = None
    company_state_code: Optional[str] = None
    company_region: Optional[str] = None
    company_city: Optional[str] = None

    funding_round_type: Optional[str] = None
    funding_round_code: Optional[str] = None
    funded_at: Optional[str] = None
    funded_month: Optional[str] = None
    funded_quarter: Optional[str] = None
    funded_year: Optional[int] = None

    raised_amount_usd: float


class InvestmentPredictResponse(BaseModel):
    predicted_next_investment: float
    expected_portfolio_return: float
    risk_score: float
    risk_level: str


def calc_size_risk(amount: float) -> float:
    return float(np.clip(amount / 50_000_000, 0, 1))


def calc_stage_risk(round_code: Optional[str]) -> float:
    mapping = {
        "seed": 0.2,
        "angel": 0.2,
        "pre-seed": 0.15,
        "a": 0.4,
        "b": 0.6,
        "c": 0.8,
        "d": 0.9,
        "growth": 1.0
    }
    if not round_code:
        return 0.5
    return mapping.get(round_code.lower(), 0.6)


def calc_volatility_risk(history: np.ndarray) -> float:
    if len(history) < 5:
        return 0.3
    returns = np.diff(history) / history[:-1]
    vol = np.std(returns)
    return float(np.clip(vol / 0.5, 0, 1))


@app.post("/predict/investment", response_model=InvestmentPredictResponse)
def predict_investment(req: InvestmentRiskInput):
    try:
        if req.raised_amount_usd <= 0:
            raise HTTPException(400, "raised_amount_usd must be > 0")

        model = get_or_load_model(
            investment_model_ref,
            OUTPUT_MODEL_INVEST,
            "keras"
        )

        value = float(req.raised_amount_usd)

        seq = np.full(
            (1, SEQ_LENGTH, 1),
            value,
            dtype=np.float32
        )

        pred = float(model.predict(seq, verbose=0)[0][0])

        expected_return = (pred - value) / value

        history = seq.flatten()

        size_risk = calc_size_risk(value)
        stage_risk = calc_stage_risk(req.funding_round_code)
        volatility_risk = calc_volatility_risk(history)

        risk_score = (
            0.4 * size_risk +
            0.4 * volatility_risk +
            0.2 * stage_risk
        )

        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return InvestmentPredictResponse(
            predicted_next_investment=round(pred, 2),
            expected_portfolio_return=round(expected_return, 4),
            risk_score=round(risk_score, 4),
            risk_level=risk_level
        )

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Investment prediction failed"
        )
