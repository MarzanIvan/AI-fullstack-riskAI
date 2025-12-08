# backend/app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import os
import httpx
from pydantic import BaseModel
from typing import Any, Dict

ML_URL = os.environ.get("ML_URL", "http://ml_service:9000")

app = FastAPI(title="Business Backend (Proxy)")

# --- Simple health ---
@app.get("/health")
async def health():
    return {"status": "ok", "service": "backend"}

# --- Proxy endpoints to ML service ---
@app.post("/credit/train")
async def credit_train(payload: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{ML_URL}/train/credit", json=payload)
    return JSONResponse(status_code=r.status_code, content=r.json())

@app.post("/credit/predict")
async def credit_predict(payload: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{ML_URL}/predict/credit", json=payload)
    return JSONResponse(status_code=r.status_code, content=r.json())

@app.post("/invest/train")
async def invest_train(payload: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{ML_URL}/train/invest", json=payload)
    return JSONResponse(status_code=r.status_code, content=r.json())

@app.post("/invest/predict")
async def invest_predict(payload: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{ML_URL}/predict/invest", json=payload)
    return JSONResponse(status_code=r.status_code, content=r.json())

@app.get("/invest/var")
async def invest_var(confidence: float = 0.95, horizon_days: int = 1, method: str = "historical"):
    params = {"confidence": confidence, "horizon_days": horizon_days, "method": method}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{ML_URL}/var", params=params)
    return JSONResponse(status_code=r.status_code, content=r.json())

@app.post("/insurance/train")
async def insurance_train(payload: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{ML_URL}/train/insurance", json=payload)
    return JSONResponse(status_code=r.status_code, content=r.json())

@app.post("/insurance/predict")
async def insurance_predict(payload: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{ML_URL}/predict/insurance", json=payload)
    return JSONResponse(status_code=r.status_code, content=r.json())

# --- Generic error handler ---
@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})
