import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import pandas as pd
import re
from logistic_model import LogisticRegressionModel
from linear_model import Sequential
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "file://"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vectorizer
logistic_vectorizer = joblib.load("tfidf_vectorizer.pkl")
linear_vectorizer = joblib.load("tfidf_vectorizer_linear.pkl")

# Build model
logistic_input_dim = len(logistic_vectorizer.get_feature_names_out())
logistic_model = LogisticRegressionModel(logistic_input_dim)
logistic_model.load_state_dict(torch.load("logistical_model.pth", map_location="cpu"))
logistic_model.eval()

linear_input_dim = len(linear_vectorizer.get_feature_names_out())
linear_model = Sequential(linear_input_dim)
linear_model.load_state_dict(torch.load("linear_model.pth", map_location="cpu"))
linear_model.eval()

class ReviewRequest(BaseModel):
    text: str
    
class AppIdRequest(BaseModel):
    appid: int

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"httpsS+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_text(text:str):
    cleaned = clean_text(text)
    
    # logisitic
    logistic_tfidf = logistic_vectorizer.transform([cleaned])
    logistic_tensor = torch.tensor(logistic_tfidf.toarray()).float()
    
    # linear
    linear_tfidf = linear_vectorizer.transform([cleaned])
    linear_tensor = torch.tensor(linear_tfidf.toarray()).float()
    
    with torch.no_grad():
        logistic_logits = logistic_model(logistic_tensor)
        logistic_prob = torch.sigmoid(logistic_logits).item()
        
        linear_log_prob = linear_model(linear_tensor)
        linear_prob = torch.exp(linear_log_prob)
        linear_prob = linear_prob[0][1].item()
        
    return {
        "logistic_prob": logistic_prob,
        "logistic_label": "positive" if logistic_prob >= 0.5 else "negative",

        "linear_prob": linear_prob,
        "linear_label": "positive" if linear_prob >= 0.5 else "negative"
    }

def get_app_name(appid: int) -> str:
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        info = data.get(str(appid), {})
        if info.get("success") and "data" in info:
            return info["data"].get("name", "Unknown App")
    except Exception:
        pass
    return "Unknown App"

@app.get("/")
def root():
    return {"message": "API running"}

@app.get("/metrics")
def get_metrics():
    try:
        linear_df = pd.read_csv("linear_model_metrics.csv", header=None)
        logistic_df = pd.read_csv("logistic_model_metrics.csv", header=None)
    except Exception as e:
        return {"error": str(e)}

    linear_vals = linear_df.iloc[0].tolist()
    logistic_vals = logistic_df.iloc[0].tolist()
    
    linear_vals = [x.item() if hasattr(x, "item") else x for x in linear_vals]
    logistic_vals= [x.item() if hasattr(x, "item") else x for x in logistic_vals]

    def parse(vals):
        return {
            "model_name": vals[0],
            "num_features": vals[1],
            "epochs": vals[2],
            "learning_rate": vals[3],
            "test_loss": vals[4],
            "test_accuracy": vals[5],
            "tn": vals[6],
            "fp": vals[7],
            "fn": vals[8],
            "tp": vals[9]
        }

    return {
        "linear": parse(linear_vals),
        "logistic": parse(logistic_vals)
    }

@app.post("/predict")
def predict(req: ReviewRequest):
    return predict_text(req.text)
    
@app.post("/predict_from_appid")
def predict_from_appid(req: AppIdRequest):
    app_name = get_app_name(req.appid)

    url = (
        f"https://store.steampowered.com/appreviews/{req.appid}"
        "?json=1&language=english&review_type=positive&num_per_page=100&filter=updated&day_range=30"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {
            "appid": req.appid,
            "name": app_name,
            "error": "Could not fetch reviews from Steam."
        }

    reviews = data.get("reviews", [])

    if not reviews:
        return {
            "appid": req.appid,
            "name": app_name,
            "message": "No reviews found",
            "total_reviews": 0,
            "logistic_percentage": None,
            "linear_percentage": None,
            "reviews": []
        }

    results = []
    logistic_pos = 0
    linear_pos = 0

    for r in reviews:
        text = r.get("review", "")
        pred = predict_text(text)

        if pred["logistic_label"] == "positive":
            logistic_pos += 1
        if pred["linear_label"] == "positive":
            linear_pos += 1

        results.append({
            "text": text,
            **pred
        })

    total = len(results)

    return {
        "appid": req.appid,
        "name": app_name,

        "total_reviews": total,

        "logistic_positive": logistic_pos,
        "logistic_percentage": (logistic_pos / total) * 100.0,

        "linear_positive": linear_pos,
        "linear_percentage": (linear_pos / total) * 100.0,

        "reviews": results
    }
    
@app.get("/linear_training_plot.png")
def get_linear_plot():
    path = "linear_plot.png"
    if not os.path.exists(path):
        return {"error": f"{path} not found"}
    return FileResponse(path, media_type="image/png")

@app.get("/logistic_training_plot.png")
def get_logistic_plot():
    path = "logistic_plot.png"
    if not os.path.exists(path):
        return {"error": f"{path} not found"}
    return FileResponse(path, media_type="image/png")

@app.get("/linear_confusion_matrix.png")
def get_linear_cm():
    path = "linear_confusion_matrix.png"
    if not os.path.exists(path):
        return {"error": f"{path} not found"}
    return FileResponse(path, media_type="image/png")

@app.get("/logistic_confusion_matrix.png")
def get_logistic_cm():
    path = "logistic_confusion_matrix.png"
    if not os.path.exists(path):
        return {"error": f"{path} not found"}
    return FileResponse(path, media_type="image/png")