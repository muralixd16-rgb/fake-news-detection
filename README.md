# 📰 Fake News Detection System

A full-stack **Fake News Detection** application that classifies news articles as **REAL** or **FAKE** using two different NLP approaches — **TF-IDF + Passive Aggressive Classifier** and **BERT Transformer** — with a **FastAPI** backend and **Streamlit** frontend.

---

## 🎯 Features

- **Dual Model Architecture** — Compare TF-IDF (fast, lightweight) vs BERT (deep learning, contextual)
- **Real-Time Predictions** — Instant classification with confidence scores
- **REST API** — FastAPI backend with interactive Swagger docs
- **Modern UI** — Streamlit frontend with model selector, progress bars, and styled result cards
- **Text Preprocessing** — Automated cleaning, stopword removal, and lemmatization

---

## 🛠️ Tech Stack

| Layer      | Technology                                      |
| ---------- | ----------------------------------------------- |
| **Frontend** | Streamlit                                     |
| **Backend**  | FastAPI + Uvicorn                             |
| **ML Model** | Scikit-learn (TF-IDF + Passive Aggressive)    |
| **DL Model** | PyTorch + HuggingFace Transformers (BERT)     |
| **NLP**      | NLTK (tokenization, lemmatization, stopwords) |
| **Data**     | Pandas, NumPy                                 |

---

## 📂 Project Structure

```
fake-news-detection/
│
├── app/
│   ├── main.py              # FastAPI application with /predict and /predict_bert endpoints
│   ├── model.py             # Model loading, text cleaning, and prediction logic
│   └── schemas.py           # Pydantic request/response schemas
│
├── model/
│   ├── tfidf_model.pkl      # Trained TF-IDF + Passive Aggressive model
│   ├── vectorizer.pkl       # Fitted TF-IDF vectorizer
│   ├── bert_model/          # Fine-tuned BERT model directory
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── requirements.txt
│   └── .gitignore
│
├── streamlit_app.py         # Streamlit frontend with model selector
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/muralixd16-rgb/fake-news-detection.git
cd fake-news-detection
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r model/requirements.txt
```

### 4️⃣ Download / Place Model Files

> **Note:** Large model files (`.pkl`, `.safetensors`) are excluded from Git via `.gitignore`.
> You need to place them manually in the `model/` directory after cloning.

| File | Location | Description |
| --- | --- | --- |
| `tfidf_model.pkl` | `model/` | Trained Passive Aggressive Classifier |
| `vectorizer.pkl` | `model/` | Fitted TF-IDF Vectorizer |
| `bert_model/` | `model/bert_model/` | Fine-tuned BERT model (config, weights, tokenizer) |

---

## 🚀 Running the Application

### Step 1: Start FastAPI Backend

```bash
cd app
python -m uvicorn main:app --port 8001
```

The API will be available at: **http://127.0.0.1:8001**

### Step 2: Start Streamlit Frontend

Open a **new terminal** and run:

```bash
streamlit run streamlit_app.py
```

The UI will be available at: **http://localhost:8501**

---

## 🌐 API Endpoints

| Method | Endpoint | Description | Model |
| ------ | -------- | ----------- | ----- |
| `GET` | `/` | Health check | — |
| `POST` | `/predict` | Predict using TF-IDF | TF-IDF + PAC |
| `POST` | `/predict_bert` | Predict using BERT | BERT Transformer |

### Request Body

```json
{
  "text": "Your news article text here..."
}
```

### Response

```json
{
  "label": "FAKE",
  "confidence": 0.9234
}
```

### Interactive API Docs

Visit **http://127.0.0.1:8001/docs** for Swagger UI.

---

## ⚖️ Model Comparison

| Feature               | TF-IDF + PAC | BERT            |
| --------------------- | ------------ | --------------- |
| Accuracy              | Moderate     | Very High       |
| Prediction Speed      | ⚡ Fast       | 🐢 Slow         |
| Memory Usage          | 🪶 Low       | 💾 High (~440MB) |
| Deployment Complexity | Easy         | Complex         |
| Context Understanding | Bag-of-words | Contextual      |
| Confidence Scores     | ✅ Yes        | ✅ Yes           |

---

## 📊 How It Works

### TF-IDF Pipeline
1. **Text Cleaning** — Lowercase, remove URLs/HTML/special characters
2. **Stopword Removal** — Remove common English stopwords (NLTK)
3. **Lemmatization** — Reduce words to base form (WordNet)
4. **TF-IDF Vectorization** — Convert text to numerical features
5. **Classification** — Passive Aggressive Classifier predicts REAL/FAKE

### BERT Pipeline
1. **Tokenization** — BERT WordPiece tokenizer (max 256 tokens)
2. **Encoding** — Generate input IDs and attention masks
3. **Inference** — Forward pass through fine-tuned `BertForSequenceClassification`
4. **Softmax** — Convert logits to probability distribution
5. **Classification** — Argmax to determine REAL/FAKE with confidence

---

## 📸 Screenshots

### Frontend — Model Selection
The Streamlit UI lets you choose between TF-IDF and BERT models before running predictions.

### Frontend — Prediction Result
After clicking **Detect**, results are displayed with label, confidence score, progress bar, and model info.

---

## 📋 Requirements

```
fastapi
uvicorn
streamlit
scikit-learn
pandas
numpy
nltk
joblib
transformers
torch
```

---

## 👨‍💻 Author

**Murali Manohar**

---

## 📄 License

This project is for educational and research purposes.
