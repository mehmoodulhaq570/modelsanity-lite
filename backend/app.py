# backend/app.py
import os
import io
import uuid
import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sqlalchemy
from sqlalchemy import text

DATABASE_URL = os.environ.get("DATABASE_URL")  # Neon/Postgres or fallback to sqlite:///:memory:
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./data.db"

engine = sqlalchemy.create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})

# Create runs table if not exists
with engine.connect() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        dataset_name TEXT,
        created_at TIMESTAMP,
        metrics_json TEXT
    )
    """))

app = FastAPI(title="ModelSanity Lite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, set to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeResult(BaseModel):
    dataset_name: str
    rows: int
    cols: int
    missing_per_column: dict
    dtypes: dict
    target_balance: dict | None
    sample_leakage_warnings: list
    
@app.get("/")
def home():
    return {"message": "ModelSanity Lite backend is running!"}


@app.post("/analyze", response_model=AnalyzeResult)
async def analyze_csv(file: UploadFile = File(...), target_column: str | None = None):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    rows, cols = df.shape

    # missing values
    missing = df.isnull().sum().to_dict()

    # dtypes
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    # simple leakage detection: target present in columns with high correlation (if numeric)
    leakage_warnings = []
    target_balance = None
    if target_column and target_column in df.columns:
        y = df[target_column]
        if y.nunique() <= 20:  # classification-ish
            target_balance = y.value_counts(normalize=True).to_dict()

        # numeric correlation check
        numeric = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors="ignore")
        if target_column in df.select_dtypes(include=[np.number]).columns:
            corr = numeric.corrwith(df[target_column]).abs().sort_values(ascending=False)
            high_corr = corr[corr > 0.9]
            for col in high_corr.index:
                leakage_warnings.append(f"High correlation ({corr[col]:.2f}) between {col} and target -> possible leakage")

    result = AnalyzeResult(
        dataset_name=file.filename,
        rows=int(rows),
        cols=int(cols),
        missing_per_column={k:int(v) for k,v in missing.items()},
        dtypes=dtypes,
        target_balance=target_balance,
        sample_leakage_warnings=leakage_warnings
    )
    return result

@app.post("/train")
async def train_model(file: UploadFile = File(...), target_column: str = "target"):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"target_column '{target_column}' not found")

    X = df.drop(columns=[target_column])
    # simple preprocessing: numeric only, drop non-numeric for baseline
    X_num = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[target_column]

    if y.nunique() < 2:
        raise HTTPException(status_code=400, detail="target has fewer than 2 classes")

    X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    # store run
    run_id = str(uuid.uuid4())
    run = {
        "id": run_id,
        "dataset_name": file.filename,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "metrics_json": str({"accuracy": acc, "n_train": len(X_train), "n_test": len(X_test)})
    }
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO runs (id, dataset_name, created_at, metrics_json) VALUES (:id, :dataset_name, :created_at, :metrics_json)"), **run)

    # save model file
    model_path = f"/tmp/{run_id}_model.joblib"
    joblib.dump(clf, model_path)

    return {"run_id": run_id, "accuracy": acc, "n_train": len(X_train), "n_test": len(X_test)}
    
@app.get("/runs")
async def list_runs():
    with engine.connect() as conn:
        res = conn.execute(text("SELECT id, dataset_name, created_at, metrics_json FROM runs ORDER BY created_at DESC LIMIT 50"))
        rows = [dict(r) for r in res]
    return {"runs": rows}
