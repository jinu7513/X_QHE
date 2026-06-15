from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import io
import json
import os
from pydantic import BaseModel
from typing import Optional, List

from he_hhl_solver import run_analysis

app = FastAPI(title="Quantum HE Regression API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    with open(os.path.join(frontend_dir, "index.html")) as f:
        return HTMLResponse(f.read())

def parse_x_axis(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    if pd.api.types.is_datetime64_any_dtype(series):
        dates = pd.to_datetime(series, errors="coerce")
        finite_dates = dates.dropna()
        if len(finite_dates) < 2:
            raise ValueError("x_column must contain at least two valid numeric or date values.")
        origin = finite_dates.min()
        values = ((dates - origin).dt.total_seconds() / 86400.0).to_numpy(dtype=float)
        return values, series.astype(str).tolist()

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= 2:
        return numeric.to_numpy(dtype=float), series.astype(str).tolist()

    dates = pd.to_datetime(series, errors="coerce")
    if dates.notna().sum() >= 2:
        origin = dates.dropna().min()
        values = ((dates - origin).dt.total_seconds() / 86400.0).to_numpy(dtype=float)
        return values, series.astype(str).tolist()

    raise ValueError("x_column must contain at least two valid numeric or date values.")

def parse_x_values(series: pd.Series) -> np.ndarray:
    values, _ = parse_x_axis(series)
    return values

def finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number

def build_plot_data(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_labels: list[str],
    result: dict,
) -> dict:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return {"points": [], "lines": {}}

    sorted_indices = indices[np.argsort(x_values[indices])]

    points = [
        {
            "x": float(x_values[i]),
            "x_label": x_labels[i] if i < len(x_labels) else str(i),
            "y": float(y_values[i]),
        }
        for i in sorted_indices
    ]

    line_sources = {
        "classical": "Classical",
        "hhl": "Quantum HHL",
        "he_hhl": "HE + HHL",
    }
    lines = {}
    for key, label in line_sources.items():
        method = result.get(key, {})
        intercept = finite_float(method.get("intercept"))
        slope = finite_float(method.get("slope"))
        if intercept is None or slope is None:
            continue
        lines[key] = {
            "label": label,
            "points": [
                {
                    "x": float(x_values[i]),
                    "y": float(intercept + slope * x_values[i]),
                }
                for i in sorted_indices
            ],
        }

    return {
        "points": points,
        "lines": lines,
    }

@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    target_columns: str = Form(None),
    x_column: str = Form(None)
):
    try:
        contents = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".csv") or filename.endswith(".txt"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
            
        # Parse targets
        if target_columns:
            targets = [col.strip() for col in target_columns.split(",") if col.strip()]
        else:
            # Auto-infer numeric columns
            targets = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if x_column in targets:
                targets.remove(x_column)
                
        if not targets:
            raise HTTPException(status_code=400, detail="No numeric target columns found.")
            
        # Parse x_column
        if x_column:
            if x_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"x_column {x_column} not found.")
            x_values, x_labels = parse_x_axis(df[x_column])
        else:
            x_values = np.arange(len(df), dtype=float)
            x_labels = [str(i) for i in range(len(df))]

        results = {}
        for target in targets:
            if target not in df.columns:
                continue
            y_values = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
            
            try:
                col_result = run_analysis(x_values, y_values)
                col_result["plot"] = build_plot_data(x_values, y_values, x_labels, col_result)
                results[target] = col_result
            except Exception as e:
                results[target] = {"error": str(e)}

        return {"status": "success", "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
