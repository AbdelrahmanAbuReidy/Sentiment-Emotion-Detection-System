# Real-Time Sentiment & Emotion Analysis System

A production-deployed NLP web application that classifies the **sentiment** (Positive / Negative / Neutral) and **emotion** (Joy, Sadness, Anger, Fear, Love, Surprise) of text messages in real time — including a live multi-user chat where every message is analysed as it is sent.

Built as my Final Year Project for the BSc (Hons) Computer Science (Data Analytics) degree.

## What it does

- **Real-time chat analysis** — a Flask-SocketIO chat room where each message is classified for sentiment and emotion the moment it is delivered, with per-prediction confidence scores.
- **LLM-generated insight** — for analysed messages, an LLM (via the OpenRouter API) produces a short, human-readable description of how the sender is likely feeling.
- **REST prediction API** — `POST /predict` accepts raw text and returns sentiment + emotion labels with confidence scores.
- **User accounts** — registration and login with salted password hashing (Werkzeug), session management via Flask-Login.
- **Persistent storage** — users, chat sessions and messages stored in PostgreSQL through SQLAlchemy, with schema migrations managed by Flask-Migrate/Alembic.

## How it works

```
text message
   │
   ▼
Preprocessing (spaCy) ── lowercase → strip emojis/URLs/mentions → lemmatise → remove stop words
   │
   ▼
TF-IDF vectorisation ──► Naive Bayes classifiers (sentiment + emotion, trained separately)
   │                          │
   ▼                          ▼
labels + confidence      OpenRouter LLM → empathetic one-line description
   │
   ▼
broadcast to chat room over WebSocket (Socket.IO)
```

### The models

The research phase of the project compared classical ML models (Logistic Regression, SVM, Naive Bayes, Random Forest) against transformer models (BERT, DistilBERT) on a corpus of over 2 million tweets, using a CRISP-DM workflow: collection, cleaning, preprocessing, feature engineering, training and evaluation.

The **deployed** models are the tuned Multinomial Naive Bayes classifiers with TF-IDF features (`.joblib` artifacts in this repo). They were chosen for deployment because they give strong accuracy at a fraction of the inference cost — the app runs comfortably on a free-tier CPU host with sub-second predictions, which a transformer cannot do.

| Artifact | Role |
| --- | --- |
| `sentiment_nb_model.joblib` + `sentiment_vectorizer.joblib` | 3-class sentiment classifier |
| `emotion_nb_model_balanced.joblib` + `emotion_vectorizer.joblib` | 6-class emotion classifier (trained on a class-balanced set) |
| `*_label_encoder.joblib` | label ↔ class-name mapping |

## Tech stack

**Python, Flask, Flask-SocketIO (WebSockets), scikit-learn, spaCy, NumPy, joblib, PostgreSQL, SQLAlchemy, Flask-Migrate (Alembic), Flask-Login, OpenRouter API (LLM), Render (deployment)**

## Running locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# environment
export DATABASE_URL=postgresql://user:pass@localhost:5432/yourdb   # optional; defaults to local postgres
export OPENROUTER_API_KEY=sk-or-...                                # optional; only needed for LLM descriptions
export SECRET_KEY=a-strong-random-value

flask db upgrade        # create tables
python app.py           # serves on http://127.0.0.1:5000
```

### API example

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I finally got the internship, I cannot stop smiling!"}'
```

```json
{
  "emotion":   { "label": "joy",      "confidence": 0.93 },
  "sentiment": { "label": "positive", "confidence": 0.96 }
}
```

## Deployment

The repo is deploy-ready for Render: `Procfile` starts the Socket.IO server, `render_postbuild.sh` installs the spaCy model at build time, and the database URL, LLM key and secret key are read from environment variables.

## Author

**Abdelrahman Zakaria Abu Reidy** — [github.com/AbdelrahmanAbuReidy](https://github.com/AbdelrahmanAbuReidy)
