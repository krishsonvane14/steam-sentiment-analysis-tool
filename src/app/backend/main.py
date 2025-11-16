import requests
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import re
from logistic_2 import LogisticRegressionModel
from fastapi.middleware.cors import CORSMiddleware

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
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Build model
input_dim = len(vectorizer.get_feature_names_out())
model = LogisticRegressionModel(input_dim)
model.load_state_dict(torch.load("logistical_model.pth", map_location="cpu"))
model.eval()

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

def predict_text(text:str) -> float:
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    x = torch.tensor(tfidf.toarray()).float()
    
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()
    return prob

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

@app.post("/predict")
def predict(req: ReviewRequest):
    text = clean_text(req.text)
    tfidf = vectorizer.transform([text])
    x = torch.tensor(tfidf.toarray()).float()

    # Predict
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    label = "positive" if prob >= 0.5 else "negative"

    return {
        "probability": prob,
        "label": label
    }
    
@app.post("/predict_from_appid")
def predict_from_appid(req: AppIdRequest):
    # Get app name
    app_name = get_app_name(req.appid)

    # Fetch reviews
    url = (
        f"https://store.steampowered.com/appreviews/{req.appid}"
        "?json=1&language=english&review_type=all&num_per_page=20&filter=recent"
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
            "positive_percentage": None,
            "total_reviews": 0,
            "reviews": []
        }

    results = []
    positive = 0

    for r in reviews:
        text = r.get("review", "")
        prob = predict_text(text)
        label = "positive" if prob >= 0.50 else "negative"

        if label == "positive":
            positive += 1

        results.append({
            "text": text,
            "probability": prob,
            "label": label
        })

    total = len(results)
    percentage = (positive / total) * 100.0

    return {
        "appid": req.appid,
        "name": app_name,
        "total_reviews": total,
        "positive_reviews": positive,
        "positive_percentage": percentage,
        "reviews": results
    }
