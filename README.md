# BERT Sentiment Analysis API

A FastAPI application that provides sentiment analysis using a fine-tuned BERT model.

## Features

- Sentiment analysis endpoint with confidence scores
- Input validation and error handling
- Health check endpoint
- Containerized with Docker
- Ready for cloud deployment

## API Endpoints

### POST /analyze
Analyze sentiment of input text.

**Request Body:**
```json
{
  "text": "I love this product!"
}
```

**Response:**
```json
{
  "label": "positive",
  "confidence": 0.9876
}
```

### GET /health
Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn main:app --reload
```

3. Visit http://localhost:8000/docs for interactive API documentation.

## Docker Deployment

1. Build the image:
```bash
docker build -t bert-sentiment-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 bert-sentiment-api
```

## Cloud Deployment

### Render.com

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set the following:
   - **Runtime**: Docker
   - **Build Command**: (leave empty, uses Dockerfile)
   - **Start Command**: (leave empty, uses CMD from Dockerfile)
4. Add environment variable if needed (none required)
5. Deploy

### AWS (ECS/Fargate)

1. Build and push Docker image to ECR:
```bash
aws ecr get-login-password --region your-region | docker login --username AWS --password-stdin your-account.dkr.ecr.your-region.amazonaws.com
docker tag bert-sentiment-api:latest your-account.dkr.ecr.your-region.amazonaws.com/bert-sentiment-api:latest
docker push your-account.dkr.ecr.your-region.amazonaws.com/bert-sentiment-api:latest
```

2. Create ECS cluster, task definition, and service
3. Configure load balancer and security groups

### Google Cloud Platform (Cloud Run)

1. Build and push to GCR:
```bash
gcloud builds submit --tag gcr.io/your-project/bert-sentiment-api
```

2. Deploy to Cloud Run:
```bash
gcloud run deploy --image gcr.io/your-project/bert-sentiment-api --platform managed
```

### Azure (Container Instances)

1. Build and push to ACR:
```bash
az acr build --registry your-registry --image bert-sentiment-api .
```

2. Deploy to ACI:
```bash
az container create --resource-group your-rg --name bert-sentiment-api --image your-registry.azurecr.io/bert-sentiment-api --dns-name-label bert-sentiment-api --ports 8000
```

## Requirements

- Python 3.11+
- Fine-tuned BERT model in `./model/` directory
- Sufficient RAM for model loading (4GB+ recommended)

## Model

The API expects a fine-tuned BERT model for sentiment analysis saved in the `./model/` directory with the standard transformers format (config.json, pytorch_model.bin or model.safetensors, tokenizer files).# FastAPI
