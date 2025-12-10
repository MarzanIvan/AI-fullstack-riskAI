# ml_service/service.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional, Dict
import pandas as pd
import numpy as np
import joblib
import boto3

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# ---------- S3 Setup ----------
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3_client = boto3.client(
    service_name='s3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("./data", exist_ok=True)

app = FastAPI(title="ML Service")

# ---------- Schemas ----------
class TrainReq(BaseModel):
    csv_path: str
    target: Optional[str] = None
    config: Optional[dict] = None

class PredictReq(BaseModel):
    features: List[List[float]]

class TSReq(BaseModel):
    csv_path: str
    price_col: str = "close"
    lookback: int = 60
    epochs: int = 20

class TSPredictReq(BaseModel):
    series: List[List[float]]

# ---------- Utils ----------
def download_s3_csv(s3_path: str) -> str:
    """Download s3://bucket/key to local path and return local path"""
    key = s3_path.replace("s3://", "")
    local_path = os.path.join("data", os.path.basename(key))
    s3_client.download_file(S3_BUCKET, key, local_path)
    print(f"[S3] Dataset downloaded: {local_path}")
    return local_path

def upload_s3_model(local_path: str, model_name: str):
    """Upload model file to S3"""
    key = f"models/{os.path.basename(local_path)}"
    s3_client.upload_file(local_path, S3_BUCKET, key)
    print(f"[S3] Model uploaded: {key}")

def ensure_df(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        path = download_s3_csv(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def save_sklearn(obj, name: str):
    p = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(obj, p)
    upload_s3_model(p, name)
    return p

def load_sklearn(name: str):
    p = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return joblib.load(p)

def save_keras(model, name: str):
    p = os.path.join(MODEL_DIR, f"{name}.h5")
    model.save(p)
    upload_s3_model(p, name)
    return p

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "service": "ml_service"}

# --- Credit train/predict ---
@app.post("/train/credit")
def train_credit(req: TrainReq):
    df = ensure_df(req.csv_path)
    if not req.target:
        raise HTTPException(status_code=400, detail="target must be set")
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"target {req.target} not found")

    X = df.drop(columns=[req.target])
    y = df[req.target]
    X_num = X.select_dtypes(include=[np.number]).fillna(0)

    # Logistic Regression
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=500))])
    lr_pipe.fit(X_num, y)
    save_sklearn(lr_pipe, "credit_logreg")

    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_num, y)
    save_sklearn(rf, "credit_rf")

    # GradientBoosting
    gb = GradientBoostingClassifier(n_estimators=200)
    gb.fit(X_num, y)
    save_sklearn(gb, "credit_gb")

    # MLP (Keras)
    input_dim = X_num.shape[1]
    mlp = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    mlp.fit(X_num.values, y.values, epochs=20, batch_size=64, validation_split=0.1, callbacks=[EarlyStopping(patience=3)], verbose=0)
    save_keras(mlp, "credit_mlp")

    # Optional PD/LGD regression
    if 'pd' in df.columns:
        reg = GradientBoostingRegressor(n_estimators=200)
        reg.fit(X_num, df['pd'])
        save_sklearn(reg, "credit_pd_reg")
    if 'lgd' in df.columns:
        reg2 = GradientBoostingRegressor(n_estimators=200)
        reg2.fit(X_num, df['lgd'])
        save_sklearn(reg2, "credit_lgd_reg")

    return {"status": "ok", "models": ["credit_logreg", "credit_rf", "credit_gb", "credit_mlp"]}

@app.post("/predict/credit")
def predict_credit(req: PredictReq):
    try:
        model = load_sklearn("credit_logreg")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="credit models not found; train first")
    X = np.array(req.features)
    probs = model.predict_proba(X)[:,1].tolist()
    preds = (np.array(probs) >= 0.5).astype(int).tolist()
    return {"probability": probs, "prediction": preds}

# --- Investment train/predict and Insurance remain the same, but use ensure_df for S3 CSV and upload_s3_model for saving models ---
