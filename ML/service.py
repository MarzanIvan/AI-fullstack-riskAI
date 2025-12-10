import os
import io
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

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

S3_BUCKET = os.getenv("S3_BUCKET", "risk-model-storage")

s3 = S3Client(
    bucket=S3_BUCKET,
    aws_access_key_id=os.getenv("YANDEX_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("YANDEX_SECRET_ACCESS_KEY"),
)

CREDIT_DATASET_PATH = os.getenv("CREDIT_DATASET_PATH", "datasets/credit/credit.csv")
INVESTMENT_DATASET_PATH = os.getenv("INVESTMENT_DATASET_PATH", "datasets/invest/invest.csv")
INSURANCE_DATASET_PATH = os.getenv("INSURANCE_DATASET_PATH", "datasets/insurance/claims.csv")

OUTPUT_MODEL_CREDIT = os.getenv("MODEL_CREDIT_PATH", "models/credit/best.joblib")
OUTPUT_MODEL_INVEST = os.getenv("MODEL_INVEST_PATH", "models/invest/lstm.h5")
OUTPUT_MODEL_INSURANCE = os.getenv("MODEL_INSURANCE_PATH", "models/insurance/insurance.joblib")

app = FastAPI(title="ML Service")

@app.post("/train/credit")
def train_credit():
    try:
        return train_credit_model()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/train/investment")
def train_investment():
    try:
        return train_investment_model()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/train/insurance")
def train_insurance():
    try:
        return train_insurance_model()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "training-service"}

def load_dataset(s3_path: str):
    if s3_path.endswith(".csv"):
        return s3.read_csv(s3_path)
    elif s3_path.endswith(".json"):
        return s3.read_json(s3_path)
    elif s3_path.endswith(".parquet"):
        return s3.read_parquet(s3_path)
    else:
        raise ValueError(f"Unsupported dataset format: {s3_path}")

def train_credit_model():
    df = load_dataset(CREDIT_DATASET_PATH)

    X = df[["income", "debt", "age", "credit_score"]]
    y = df["default"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train models
    lr = LogisticRegression(max_iter=800).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400).fit(X_train, y_train)

    auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    auc_mlp = roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])

    best_model = max(
        [(lr, auc_lr, "lr"), (rf, auc_rf, "rf"), (mlp, auc_mlp, "mlp")],
        key=lambda x: x[1]
    )

    os.makedirs("models/credit", exist_ok=True)
    joblib.dump(best_model[0], "models/credit/best.joblib")

    # Upload to S3
    s3.upload_file("models/credit/best.joblib", OUTPUT_MODEL_CREDIT)

    return {"type": best_model[2], "auc": best_model[1], "saved": OUTPUT_MODEL_CREDIT}

def train_investment_model():
    df = load_dataset(INVESTMENT_DATASET_PATH)

    prices = df["price"].values.reshape(-1, 1)

    # PCA
    pca = PCA(n_components=1)
    pca.fit(df[["price", "volume"]])

    # LSTM
    seq_length = 20
    X_seq, y_seq = [], []

    for i in range(len(prices) - seq_length):
        X_seq.append(prices[i:i+seq_length])
        y_seq.append(prices[i+seq_length])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="mse")

    model.fit(X_seq, y_seq, epochs=5, batch_size=32, verbose=1)

    os.makedirs("models/invest", exist_ok=True)
    model.save("models/invest/lstm.h5")

    s3.upload_file("models/invest/lstm.h5", OUTPUT_MODEL_INVEST)

    return {"saved": OUTPUT_MODEL_INVEST, "samples": len(X_seq)}

def train_insurance_model():
    df = load_dataset(INSURANCE_DATASET_PATH)

    # Частота
    X_freq = df[["driver_age", "power", "region_risk"]]
    y_freq = df["claims_count"]

    freq_model = PoissonRegressor().fit(X_freq, y_freq)

    # Severity
    df_sev = df[df["claim_amount"] > 0]
    X_sev = df_sev[["driver_age", "power"]]
    y_sev = df_sev["claim_amount"]

    sev_model = GammaRegressor().fit(X_sev, y_sev)

    # Fraud
    fraud_model = IsolationForest().fit(df[["power", "claim_amount"]])

    os.makedirs("models/insurance", exist_ok=True)
    joblib.dump(
        {"freq": freq_model, "severity": sev_model, "fraud": fraud_model},
        "models/insurance/insurance.joblib"
    )
    s3.upload_file("models/insurance/insurance.joblib", OUTPUT_MODEL_INSURANCE)
    return {"saved": OUTPUT_MODEL_INSURANCE}



