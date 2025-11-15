from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                "text-classification",
                model=MODEL_PATH,
                tokenizer="bert-base-uncased",
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
    logger.info(f"Received text: {request.text}")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        logger.info("Starting sentiment analysis")
        # Perform sentiment analysis
        result = sentiment_model(request.text)[0]
        logger.info(f"Model result: {result}")

        # Extract label and confidence
        raw_label = result['label']
        if raw_label == 'LABEL_0':
            label = 'negative'
        elif raw_label == 'LABEL_1':
            label = 'positive'
        else:
            label = raw_label.lower()
        confidence = round(result['score'], 4)
        logger.info(f"Processed label: {label}, confidence: {confidence}")

        return SentimentResponse(label=label, confidence=confidence)

    except Exception as e:
        logger.error(f"Analysis failed with exception: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": sentiment_model is not None}