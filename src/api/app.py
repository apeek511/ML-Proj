"""
FastAPI application for speaker authentication.

Loads the trained GRU model and scaler, exposes a /predict endpoint.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent          # src/api/
PROJECT_ROOT = HERE.parent.parent                # repo root (two levels up)
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_classifier import GRUClassifier

# aths
MODEL_PATH = PROJECT_ROOT / "saved_models" / "gru_final.pth"
SCALER_PATH = PROJECT_ROOT / "saved_models" / "scaler.pkl"
PARAMS_PATH = PROJECT_ROOT / "saved_models" / "best_params.json"

# Load metadata 
with open(PARAMS_PATH) as f:
    BEST_PARAMS = json.load(f)

INPUT_SIZE = BEST_PARAMS["input_size"]      # 12
MAX_SEQ_LEN = 29                             # fixed padding length from training
N_CLASSES = BEST_PARAMS["n_classes"]         # 2
AUTH_SPEAKERS = BEST_PARAMS["authenticated_speakers"]

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The model was saved with torch.save(best_model, path) — whole-object save.
# torch.load() needs the GRUClassifier class to be importable, which it is
# because we imported it above.
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.to(DEVICE)
model.eval()

# Load scalere
scaler = joblib.load(SCALER_PATH)

# FastAPI app
app = FastAPI(
    title="Japanese Vowels Speaker Authentication",
    description=(
        "Classify a time-series of 12 LPC coefficients as belonging to "
        "an authenticated speaker (one of {}) or a stranger.".format(
            [s + 1 for s in AUTH_SPEAKERS]
        )
    ),
    version="1.0.0",
)

# Pydantic models 
class InputData(BaseModel):
    lpc_coefficients: list[list[float]]
    # Expected shape: (time_steps, 12)

class Prediction(BaseModel):
    authenticated: bool
    confidence_authenticated: float
    confidence_stranger: float

# Endpoints
@app.get("/")
def read_root():
    return {
        "message": "Speaker authentication API is running",
        "authenticated_speakers": [s + 1 for s in AUTH_SPEAKERS],
    }


@app.post("/predict", response_model=Prediction)
def predict(data: InputData):
    # Convert to numpy array — shape (time_steps, 12)
    arr = np.array(data.lpc_coefficients, dtype=np.float32)

    # Validate shape
    if arr.ndim != 2 or arr.shape[1] != INPUT_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Input must have shape (time_steps, {INPUT_SIZE}). "
                   f"Got {arr.shape}.",
        )

    # Pad or truncate to 29 time steps (same as training)
    seq = np.zeros((MAX_SEQ_LEN, INPUT_SIZE), dtype=np.float32)
    length = min(arr.shape[0], MAX_SEQ_LEN)
    seq[:length] = arr[:length]

    # Standardize using the training scaler
    seq = scaler.transform(seq)

    # Add batch dimension → shape (1, 29, 12)
    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Class 1 = authenticated, class 0 = stranger
    conf_auth = float(probs[1])
    conf_stranger = float(probs[0])
    is_auth = bool(np.argmax(probs) == 1)

    return Prediction(
        authenticated=is_auth,
        confidence_authenticated=conf_auth,
        confidence_stranger=conf_stranger,
    )
