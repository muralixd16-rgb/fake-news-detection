import joblib
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Fix path — always points to project root/model/ folder
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "vectorizer.pkl"))
model      = joblib.load(os.path.join(BASE_DIR, "model", "tfidf_model.pkl"))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

def predict(text: str):
    clean  = clean_text(text)
    vec    = vectorizer.transform([clean])
    pred   = model.predict(vec)[0]
    label  = "REAL" if pred == 1 else "FAKE"

    # PassiveAggressiveClassifier doesn't support predict_proba
    # Use decision function instead
    try:
        proba      = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)), 4)
    except:
        decision   = model.decision_function(vec)[0]
        confidence = round(float(abs(decision) / (1 + abs(decision))), 4)

    return {"label": label, "confidence": confidence}
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification

# ================= BERT =================

BERT_PATH = os.path.join(BASE_DIR, "model", "bert_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

bert_model = BertForSequenceClassification.from_pretrained(BERT_PATH)

bert_model.to(device)

bert_model.eval()


def predict_bert(text: str):

    encoding = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    input_ids = encoding["input_ids"].to(device)

    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probs      = F.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = round(float(probs[0][prediction].item()), 4)

    label = "REAL" if prediction == 1 else "FAKE"

    return {
        "label": label,
        "confidence": confidence
    }