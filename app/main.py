from fastapi import FastAPI

from schemas import ArticleRequest, PredictionResponse

from model import predict, predict_bert

app = FastAPI(title="Fake News Detector API")


# ================= HOME =================

@app.get("/")
def root():

    return {
        "status": "Fake News Detector is running!"
    }


# ================= TF-IDF =================

@app.post("/predict", response_model=PredictionResponse)
def predict_news(request: ArticleRequest):

    result = predict(request.text)

    return result


# ================= BERT =================

@app.post("/predict_bert", response_model=PredictionResponse)
def predict_news_bert(request: ArticleRequest):

    result = predict_bert(request.text)

    return result