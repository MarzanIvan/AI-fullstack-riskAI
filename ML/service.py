import os
import io
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, PoissonRegressor, GammaRegressor
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss

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



# Числовые и категориальные признаки
NUM_COLS = [
    'age', 'decline_app_cnt', 'score_bki',
    'bki_request_cnt', 'income', 'first_time',
    'region_rating', 'mean_income_region',
    'mean_income_age', 'mean_bki_age', 'income_to_request'
]

CAT_COLS = [
    'education', 'sex', 'car', 'car_type',
    'good_work', 'home_address', 'work_address',
    'foreign_passport', 'sna'
]

def build_credit_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CAT_COLS),
            ('num', 'passthrough', NUM_COLS)
        ]
    )

    model = LogisticRegression(
        class_weight='balanced',
        C=500.5,
        max_iter=1000,
        penalty='l2',
        solver='lbfgs'
    )

    pipeline = Pipeline(steps=[
        ('features', CreditFeatureEngineer()),
        ('preprocess', preprocessor),
        ('model', model)
    ])

    return pipeline



from s3_client import S3Client

OUTPUT_MODEL_CREDIT = os.getenv("MODEL_CREDIT_PATH", "models/credit/credit.joblib") # joblib from sklearn
OUTPUT_MODEL_INVEST = os.getenv("MODEL_INVEST_PATH", "models/invest/invest.h5") # h5 models from keras and tensorflow
OUTPUT_MODEL_INSURANCE = os.getenv("MODEL_INSURANCE_PATH", "models/insurance/insurance.joblib") # joblib from sklearn
S3_BUCKET = os.getenv("S3_BUCKET", "risk-model-storage")

app = FastAPI(title="ML Service")

s3 = S3Client(
    bucket=S3_BUCKET,
    aws_access_key_id=os.getenv("YANDEX_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("YANDEX_SECRET_ACCESS_KEY"),
)

def load_dataset(s3_path: str, *, csv_sep: str = ";"):
    if s3_path.endswith(".csv"):
        return s3.read_csv(
            s3_path,
            sep=csv_sep
        )
    elif s3_path.endswith(".json"):
        return s3.read_json(s3_path)
    elif s3_path.endswith(".parquet"):
        return s3.read_parquet(s3_path)
    else:
        raise ValueError(f"Unsupported dataset format: {s3_path}")


class TrainRequest(BaseModel):
    dataset_name: str
    dataset_path: str

@app.post("/train/credit")
def train_credit(req: TrainRequest):
    try:
        return train_credit_model(req.dataset_name, req.dataset_path)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(500, tb)

@app.post("/train/investment")
def train_investment(req: TrainRequest):
    try:
        return train_investment_model(req.dataset_name, req.dataset_path)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(500, tb)



@app.post("/train/insurance")
def train_insurance(req: TrainRequest):
    try:
        return train_insurance_model(req.dataset_name, req.dataset_path)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(500, tb)



@app.get("/health")
def health():
    return {"status": "ok", "service": "training-service"}

def train_credit_model(dataset_name: str, dataset_path: str):
    df = load_dataset(
        dataset_path,
        csv_sep=","
    )

    y = df['default']
    X = df.drop(columns=['default', 'client_id'])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    pipeline = build_credit_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_score),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "logloss": log_loss(y_test, y_pred)
    }
    
    local_dir = "models/credit"
    os.makedirs(local_dir, exist_ok=True)

    model_filename = f"credit.joblib"
    local_path = os.path.join(local_dir, model_filename)
    s3_key = f"models/credit/{model_filename}"
    joblib.dump(pipeline, local_path)
    s3.upload_file(
        local_path=local_path,
        key=s3_key
    )
    return {
        "model_path": local_path,
        "s3_key": s3_key,
        "train rows": len(X_train),
        "test rows": len(X_test)
    }

def train_investment_model(dataset_name: str, dataset_path: str):
    try:
        df = load_dataset(
            dataset_path,
            csv_sep=";"
        )

        df["raised_amount_usd"] = (
            df["raised_amount_usd"]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
        )

        df["raised_amount_usd"] = pd.to_numeric(
            df["raised_amount_usd"],
            errors="coerce"
        ).fillna(0.0)

        df["funded_at"] = pd.to_datetime(
            df["funded_at"],
            errors="coerce",
            dayfirst=True
        )

        df = df.dropna(subset=["funded_at"])
        df = df.sort_values("funded_at")

        prices = df["raised_amount_usd"].values.reshape(-1, 1)

        seq_length = 20
        if len(prices) <= seq_length:
            raise ValueError(
                f"Not enough data for LSTM: {len(prices)} rows"
            )

        X_seq, y_seq = [], []
        for i in range(len(prices) - seq_length):
            X_seq.append(prices[i:i+seq_length])
            y_seq.append(prices[i+seq_length])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        model = Sequential([
            LSTM(64, input_shape=(seq_length, 1), activation="tanh"),
            Dense(1)
        ])

        model.compile(optimizer=Adam(0.001), loss="mse")
        model.fit(X_seq, y_seq, epochs=3, batch_size=64, verbose=1)

        local_path = "models/invest/invest.h5"
        s3_key = "models/invest/invest.h5"

        os.makedirs("models/invest", exist_ok=True)
        model.save(local_path)

        s3.upload_file(local_path=local_path, key=s3_key)

        return {
            "model": "investment_lstm",
            "dataset": dataset_name,
            "train rows": int(len(X_seq)),
            "saved": s3_key
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise RuntimeError(tb)


policy_features = [
    "months_as_customer",
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit"
]
incident_features = [
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses"
]
claim_features = [
    "injury_claim",
    "property_claim",
    "vehicle_claim"
]
categorical_features = [
    "policy_state",
    "insured_sex",
    "insured_education_level",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "property_damage",
    "police_report_available",
    "auto_make"
]



def build_insurance_pipeline():
    numeric_features = (
        policy_features +
        incident_features +
        claim_features
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])


    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])


def train_insurance_model(dataset_name: str, dataset_path: str):
    df = load_dataset(
        dataset_path,
        csv_sep=";"
    )

    df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

    X = df.drop(columns=[
        "fraud_reported",
        "policy_number",
        "policy_bind_date",
        "incident_date",
        "_c39"
    ], errors="ignore")

    y = df["fraud_reported"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    pipeline = build_insurance_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_score),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "fraud_rate": float(y.mean())
    }

    local_path = f"models/insurance/insurance.joblib"
    os.makedirs("models/insurance", exist_ok=True)
    joblib.dump(pipeline, local_path)

    s3.upload_file(local_path, local_path)
    
    return {
        "model": "insurance_fraud",
        "dataset": dataset_name,
        "train rows": len(X_train),
        "test rows": len(X_test),
        "saved": local_path
    }


