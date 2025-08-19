# 🧠 Fake News Detection System

## Overview
This project is an **NLP + Machine Learning pipeline** to detect fake news articles. 
It preprocesses text data, trains ML models, and exposes a REST API for real-time predictions.

## Features
- Text preprocessing (tokenization, stopword removal, stemming).
- Train ML models (Logistic Regression, Random Forest, SVM).
- REST API using FastAPI (`/predict` endpoint).
- Save and load trained models with `joblib`.
- Dataset: [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news).

## Project Structure
```
fake_news_detection/
│── data/               # Dataset folder
│── artifacts/          # Saved models, vectorizers
│── src/                # Source code
│   ├── preprocess.py   # Text preprocessing utilities
│   ├── train.py        # Model training
│   ├── predict.py      # Prediction logic
│   ├── api.py          # FastAPI service
│── README.md           # Project documentation
```

## Installation
```bash
git clone <repo_url>
cd fake_news_detection
pip install -r requirements.txt
```

## Usage
1. **Train Model**
```bash
python -m src.train --csv data/train.csv --artifacts artifacts
```
2. **Run API**
```bash
uvicorn src.api:app --reload --port 8000
```
3. **Test Prediction**
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "Some news text here"}'
```

## Tech Stack
- Python, Pandas, Numpy
- Scikit-learn, NLTK
- FastAPI, Uvicorn
- Joblib

## Future Improvements
- Deep Learning models (LSTMs, Transformers)
- Web frontend for interactive use
- Deployment with Docker & CI/CD

---
✨ Author: Sriyaa  
