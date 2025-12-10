import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

from s3_client import download_dataset, upload_model

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)
app = FastAPI(title="ML Service")

# --- Schemas ---
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

# --- Utils ---
def ensure_df(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        key = path.replace("s3://", "")
        local_path = os.path.join("data", os.path.basename(key))
        download_dataset(key, local_path)
        path = local_path
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def save_sklearn(obj, name: str):
    p = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(obj, p)
    upload_model(p, f"models/{name}.joblib")
    return p

def load_sklearn(name: str):
    p = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return joblib.load(p)

def save_keras(model, name: str):
    p = os.path.join(MODEL_DIR, f"{name}.h5")
    model.save(p)
    upload_model(p, f"models/{name}.h5")
    return p

# --- Endpoints ---
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

    lr_pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=500))])
    lr_pipe.fit(X_num, y)
    save_sklearn(lr_pipe, "credit_logreg")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_num, y)
    save_sklearn(rf, "credit_rf")

    gb = GradientBoostingClassifier(n_estimators=200)
    gb.fit(X_num, y)
    save_sklearn(gb, "credit_gb")

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