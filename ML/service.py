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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

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

def load_dataset(s3_path: str):
    if s3_path.endswith(".csv"):
        return s3.read_csv(s3_path)
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
        raise HTTPException(500, str(e))


@app.post("/train/investment")
def train_investment(req: TrainRequest):
    try:
        return train_investment_model(req.dataset_name, req.dataset_path)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/train/insurance")
def train_insurance(req: TrainRequest):
    try:
        return train_insurance_model(req.dataset_name, req.dataset_path)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "training-service"}

def train_credit_model(dataset_name: str, dataset_path: str):
    df = load_dataset(dataset_path)

    X = df[["income", "debt", "age", "credit_score"]]
    y = df["default"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    lr = LogisticRegression(max_iter=800).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400).fit(X_train, y_train)

    auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    auc_mlp = roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])

    best_model, best_auc, best_type = max(
        [(lr, auc_lr, "lr"), (rf, auc_rf, "rf"), (mlp, auc_mlp, "mlp")],
        key=lambda x: x[1]
    )

    local_path = f"models/credit/{dataset_name}_best.joblib"
    os.makedirs("models/credit", exist_ok=True)
    joblib.dump(best_model, local_path)

    s3.upload_file(local_path, f"models/credit/{dataset_name}.joblib")

    return {
        "model": "credit",
        "dataset": dataset_name,
        "auc": round(best_auc, 4),
        "saved": f"models/credit/{dataset_name}.joblib"
    }

def train_investment_model(dataset_name: str, dataset_path: str):
    df = load_dataset(dataset_path)

    prices = df["price"].values.reshape(-1, 1)

    seq_length = 20
    X_seq, y_seq = [], []

    for i in range(len(prices) - seq_length):
        X_seq.append(prices[i:i+seq_length])
        y_seq.append(prices[i+seq_length])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    model = Sequential([
        LSTM(64, input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="mse")
    model.fit(X_seq, y_seq, epochs=5, batch_size=32)

    local_path = f"models/invest/{dataset_name}.h5"
    os.makedirs("models/invest", exist_ok=True)
    model.save(local_path)

    s3.upload_file(local_path, local_path)

    return {
        "model": "investment",
        "dataset": dataset_name,
        "samples": len(X_seq),
        "saved": local_path
    }

def train_insurance_model(dataset_name: str, dataset_path: str):
    df = load_dataset(dataset_path)

    freq_model = PoissonRegressor().fit(
        df[["driver_age", "power", "region_risk"]],
        df["claims_count"]
    )

    df_sev = df[df["claim_amount"] > 0]
    sev_model = GammaRegressor().fit(
        df_sev[["driver_age", "power"]],
        df_sev["claim_amount"]
    )

    fraud_model = IsolationForest().fit(df[["power", "claim_amount"]])

    local_path = f"models/insurance/{dataset_name}.joblib"
    os.makedirs("models/insurance", exist_ok=True)

    joblib.dump(
        {"freq": freq_model, "severity": sev_model, "fraud": fraud_model},
        local_path
    )

    s3.upload_file(local_path, local_path)

    return {
        "model": "insurance",
        "dataset": dataset_name,
        "saved": local_path
    }



