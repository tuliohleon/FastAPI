from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
import os

app = FastAPI(title="BERT Sentiment Analysis API", version="1.0.0")

# Model path
MODEL_PATH = "./model"

# Global model variable
sentiment_model = None

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    confidence: float

def load_model():
    global sentiment_model
    if sentiment_model is None:
        try:
            # Load the fine-tuned model
            sentiment_model = pipeline(
                "sentiment-analysis",
                model=MODEL_PATH,
                tokenizer=MODEL_PATH,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "BERT Sentiment Analysis API", "status": "running"}

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Perform sentiment analysis
        result = sentiment_model(request.text)[0]

        # Extract label and confidence
        label = result['label'].lower()
        confidence = round(result['score'], 4)

        return SentimentResponse(label=label, confidence=confidence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": sentiment_model is not None}