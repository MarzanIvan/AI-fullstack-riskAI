# ml_service/service.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional, Dict
import pandas as pd
import numpy as np
import joblib

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

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)

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
def ensure_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def save_sklearn(obj, name: str):
    p = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(obj, p)
    return p

def load_sklearn(name: str):
    p = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return joblib.load(p)

def save_keras(model, name: str):
    p = os.path.join(MODEL_DIR, f"{name}.h5")
    model.save(p)
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
    # numeric preprocessing
    X_num = X.select_dtypes(include=[np.number]).fillna(0)

    # Logistic
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

    # Optional PD/LGD regression if columns exist
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

# --- Investment (LSTM / VaR) ---
@app.post("/train/invest")
def train_invest(req: TSReq):
    df = ensure_df(req.csv_path)
    if req.price_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"{req.price_col} not in csv")
    prices = df[req.price_col].dropna().values.astype(float)
    if len(prices) < req.lookback + 2:
        raise HTTPException(status_code=400, detail="not enough data")
    returns = np.diff(np.log(prices))
    np.save(os.path.join(MODEL_DIR, "historical_returns.npy"), returns)

    # Create sequences
    lookback = req.lookback
    seqs, targets = [], []
    for i in range(len(prices) - lookback):
        seqs.append(prices[i:i+lookback])
        targets.append(prices[i+lookback])
    seqs = np.array(seqs)
    targets = np.array(targets)
    mean = float(seqs.mean())
    std = float(seqs.std() + 1e-9)
    seqs = (seqs - mean) / std
    targets = (targets - mean) / std
    X = seqs.reshape((seqs.shape[0], seqs.shape[1], 1))

    model = Sequential([
        LSTM(64, input_shape=(lookback,1)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, targets, epochs=req.epochs, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(patience=3)], verbose=0)

    save_keras(model, "invest_lstm")
    joblib.dump({"mean": mean, "std": std, "lookback": lookback}, os.path.join(MODEL_DIR, "invest_norm.joblib"))
    return {"status": "ok", "lstm": "invest_lstm.h5"}

@app.post("/predict/invest")
def predict_invest(req: TSPredictReq):
    norm = joblib.load(os.path.join(MODEL_DIR, "invest_norm.joblib"))
    model = load_model(os.path.join(MODEL_DIR, "invest_lstm.h5"))
    lookback = norm["lookback"]
    out = []
    for seq in req.series:
        if len(seq) != lookback:
            raise HTTPException(status_code=400, detail=f"each series must have length {lookback}")
        arr = (np.array(seq) - norm["mean"]) / norm["std"]
        arr = arr.reshape((1, lookback, 1))
        pred_n = model.predict(arr, verbose=0)[0,0]
        pred = float(pred_n * norm["std"] + norm["mean"])
        out.append(pred)
    return {"predictions": out}

@app.get("/var")
def compute_var(confidence: float = 0.95, horizon_days: int = 1, method: str = "historical"):
    rpath = os.path.join(MODEL_DIR, "historical_returns.npy")
    if not os.path.exists(rpath):
        raise HTTPException(status_code=400, detail="historical returns not found. Train invest first.")
    returns = np.load(rpath)
    if method == "historical":
        sorted_r = np.sort(returns)
        idx = int(max(0, (1 - confidence) * len(sorted_r)))
        var = -float(sorted_r[idx])
        return {"VaR": var, "method": "historical", "confidence": confidence}
    elif method == "montecarlo":
        mu = returns.mean()
        sigma = returns.std()
        sims = np.random.normal(mu, sigma, size=10000)
        var = -float(np.quantile(sims, 1 - confidence))
        cvar = -float(sims[sims <= np.quantile(sims, 1 - confidence)].mean())
        return {"VaR": var, "CVaR": cvar, "method": "montecarlo", "confidence": confidence}
    else:
        raise HTTPException(status_code=400, detail="unknown method")

# --- Insurance train/predict ---
@app.post("/train/insurance")
def train_insurance(req: TrainReq):
    df = ensure_df(req.csv_path)
    if not req.target:
        raise HTTPException(status_code=400, detail="target must be set")
    if req.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"{req.target} not found")

    X = df.drop(columns=[req.target])
    X_num = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[req.target]

    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(X_num, y)
    save_sklearn(clf, "ins_claim_gb")

    if 'claim_count' in df.columns:
        lambda_hat = float(df['claim_count'].mean())
        joblib.dump({"lambda": lambda_hat}, os.path.join(MODEL_DIR, "insurance_actuarial.joblib"))

    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_num)
    save_sklearn(iso, "insurance_fraud_iso")

    return {"status": "ok"}

@app.post("/predict/insurance")
def predict_insurance(req: PredictReq):
    try:
        clf = load_sklearn("ins_claim_gb")
        iso = load_sklearn("insurance_fraud_iso")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="models not found; train first")
    X = np.array(req.features)
    probs = clf.predict_proba(X)[:,1].tolist()
    preds = clf.predict(X).tolist()
    fraud_score = iso.decision_function(X).tolist()
    anomaly = iso.predict(X).tolist()
    return {"probability": probs, "prediction": preds, "fraud_score": fraud_score, "anomaly_flag": anomaly}
