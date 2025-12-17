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
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.mean_income_region = X.groupby('region_rating')['income'].transform('mean')
            self.mean_income_age = X.groupby('age')['income'].transform('mean')
            self.mean_bki_age = X.groupby('age')['score_bki'].transform('mean')
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=X.columns if hasattr(X, 'columns') else None)

        df = X.copy()

        # Новые признаки
        df['income_to_request'] = df['income'] / (df['bki_request_cnt'] + 1)

        # Если расчет средних нужен на этапе трансформации
        if hasattr(self, 'mean_income_region'):
            df['mean_income_region'] = self.mean_income_region
        else:
            df['mean_income_region'] = df['income']  # fallback

        if hasattr(self, 'mean_income_age'):
            df['mean_income_age'] = self.mean_income_age
        else:
            df['mean_income_age'] = df['income']  # fallback

        if hasattr(self, 'mean_bki_age'):
            df['mean_bki_age'] = self.mean_bki_age
        else:
            df['mean_bki_age'] = df['score_bki']  # fallback

        # Преобразование категориальных признаков
        df['sex'] = df['sex'].map({'M': 1, 'F': 0}).fillna(0)
        df['good_work'] = df['good_work'].astype(int)
        df['first_time'] = df['first_time'].astype(int)

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
    'foreign_passport', 'sna', 'month'
]

def build_credit_pipeline():
    """
    Создает sklearn Pipeline для кредитного скоринга:
    1) Инжиниринг признаков (CreditFeatureEngineer)
    2) Препроцессинг (OneHotEncoder + passthrough)
    3) Логистическая регрессия
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CAT_COLS),
            ('num', 'passthrough', NUM_COLS)
        ]
    )

    model = LogisticRegression(
        class_weight='balanced',
        C=500.5,
        max_iter=400,
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

    model_path = f"ml/models/credit_{dataset_name}_v1.joblib"
    joblib.dump(pipeline, model_path)

    return {
        "model_path": model_path,
        "metrics": metrics,
        "features_count": pipeline.named_steps['preprocess'].get_feature_names_out().shape[0]
    }

def train_investment_model(dataset_name: str, dataset_path: str):
    df = load_dataset(dataset_path)

    df['raised_amount_usd'] = (
        df['raised_amount_usd']
        .astype(str)
        .str.replace(' ', '')  # удаляем неразрывный пробел
        .replace('', '0')
        .astype(float)
    )
    df['funded_at'] = pd.to_datetime(df['funded_at'], errors='coerce')
    df = df.dropna(subset=['funded_at'])
    df = df.sort_values('funded_at')

    prices = df['raised_amount_usd'].values.reshape(-1, 1)
    seq_length = 20
    X_seq, y_seq = [], []

    for i in range(len(prices) - seq_length):
        X_seq.append(prices[i:i+seq_length])
        y_seq.append(prices[i+seq_length])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    model = Sequential([
        LSTM(64, input_shape=(seq_length, 1), activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=1)

    local_path = f"models/invest/invest_{dataset_name}_v1.h5"
    os.makedirs("models/invest", exist_ok=True)
    model.save(local_path)
    s3.upload_file(local_path, local_path)

    y_pred = model.predict(X_seq)
    mse = np.mean((y_pred.flatten() - y_seq.flatten()) ** 2)

    return {
        "model": "investment_lstm",
        "dataset": dataset_name,
        "samples": len(X_seq),
        "mse": float(mse),
        "saved": local_path
    }



def train_investment_model_old(dataset_name: str, dataset_path: str):
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
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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
    df = load_dataset(dataset_path)

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

    local_path = f"models/insurance/insurance_{dataset_name}_v1.joblib"
    os.makedirs("models/insurance", exist_ok=True)
    joblib.dump(pipeline, local_path)

    s3.upload_file(local_path, local_path)

    return {
        "model": "insurance_fraud",
        "dataset": dataset_name,
        "saved": local_path,
        "metrics": metrics,
        "features": pipeline.named_steps["preprocess"].get_feature_names_out().shape[0]
    }


