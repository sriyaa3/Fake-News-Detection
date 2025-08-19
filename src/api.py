from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class NewsItem(BaseModel):
    text: str

@app.post("/predict")
def get_prediction(item: NewsItem):
    result = predict(item.text)
    return {"prediction": result}
