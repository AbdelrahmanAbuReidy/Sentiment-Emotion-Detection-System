from flask import Flask, request, jsonify, render_template, session
import joblib
import re
import spacy
import numpy as np
from flask_socketio import SocketIO, emit, join_room, leave_room
import uuid
import requests
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-only-secret-change-in-production')
socketio = SocketIO(app, cors_allowed_origins="*")

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/postgres')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Emoji removal pattern
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U00002700-\U000027BF"  # dingbats
    u"\U0001F900-\U0001F9FF"  # supplemental symbols
    u"\U00002600-\U000026FF"  # misc symbols
    "]+", flags=re.UNICODE)

# Preprocessing function (same as used in training)
def preprocess(text):
    """Preprocess text for model inference"""
    text = text.lower()
    text = emoji_pattern.sub('', text)  # remove emojis
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)  # URLs & mentions
    text = re.sub(r"[^a-z\s]", "", text)             # Remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and len(token) > 2]
    return " ".join(tokens)

# Load saved models
try:
    # Emotion models
    emotion_model = joblib.load('emotion_nb_model_balanced.joblib')
    emotion_vectorizer = joblib.load('emotion_vectorizer.joblib')
    emotion_label_encoder = joblib.load('emotion_label_encoder.joblib')
    
    # Sentiment models
    sentiment_model = joblib.load('sentiment_nb_model.joblib')
    sentiment_vectorizer = joblib.load('sentiment_vectorizer.joblib')
    sentiment_label_encoder = joblib.load('sentiment_label_encoder.joblib')
    
    print("✅ All models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# Use DeepSeek model via OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Use DeepSeek model via OpenRouter
OPENROUTER_MODEL = "deepseek-ai/deepseek-chat"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def get_gpt_description(message, emotion, sentiment):
    prompt = (
        f"Message: \"{message}\"\n"
        f"Emotion: {emotion}\n"
        f"Sentiment: {sentiment}\n"
        "Describe in 20 to 30 words how the sender is feeling or what their message conveys, in a human, empathetic way."
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct",  # <-- changed here
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[GPT error: {e}]"

LOGIN_TEMPLATE = 'login.html'
REGISTER_TEMPLATE = 'register.html'

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return render_template(LOGIN_TEMPLATE, message='You have been logged out.')

@app.route('/')
@login_required
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict emotion and sentiment for given text"""
    try:
        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Preprocess the text
        processed_text = preprocess(text)
        
        if not processed_text.strip():
            return jsonify({
                'error': 'Text contains no meaningful words after preprocessing'
            }), 400
        
        # Predict emotion
        emotion_features = emotion_vectorizer.transform([processed_text])
        emotion_prediction = emotion_model.predict(emotion_features)[0]
        emotion_label = emotion_label_encoder.inverse_transform([emotion_prediction])[0]
        emotion_confidence = np.max(emotion_model.predict_proba(emotion_features))
        
        # Predict sentiment
        sentiment_features = sentiment_vectorizer.transform([processed_text])
        sentiment_prediction = sentiment_model.predict(sentiment_features)[0]
        sentiment_label = sentiment_label_encoder.inverse_transform([sentiment_prediction])[0]
        sentiment_confidence = np.max(sentiment_model.predict_proba(sentiment_features))
        
        # Return results
        result = {
            'original_text': text,
            'processed_text': processed_text,
            'emotion': {
                'label': emotion_label,
                'confidence': float(emotion_confidence)
            },
            'sentiment': {
                'label': sentiment_label,
                'confidence': float(sentiment_confidence)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'emotion_model': 'emotion_nb_model_balanced.joblib',
        'sentiment_model': 'sentiment_nb_model.joblib'
    })

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        confirm_password = data.get('confirm_password')
        # Basic validation
        if not username or not email or not password or not confirm_password:
            return render_template(REGISTER_TEMPLATE, error='All fields are required.')
        if password != confirm_password:
            return render_template(REGISTER_TEMPLATE, error='Passwords do not match.')
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            return render_template(REGISTER_TEMPLATE, error='Username or email already exists.')
        # Hash the password
        password_hash = generate_password_hash(password)
        # Create and add user
        user = User(username=username, email=email, password_hash=password_hash)
        db.session.add(user)
        db.session.commit()
        return render_template(LOGIN_TEMPLATE, message='Registration successful! Please log in.')
    return render_template(REGISTER_TEMPLATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        identifier = data.get('identifier')  # username or email
        password = data.get('password')
        if not identifier or not password:
            return render_template(LOGIN_TEMPLATE, error='Username/email and password are required.')
        # Try to find user by username or email
        user = User.query.filter((User.username == identifier) | (User.email == identifier)).first()
        if not user or not check_password_hash(user.password_hash, password):
            return render_template(LOGIN_TEMPLATE, error='Invalid credentials.')
        login_user(user)
        return render_template('index.html', message='Login successful!')
    return render_template(LOGIN_TEMPLATE)

# --- SocketIO Real-Time Chat Logic ---
users = {}  # session_id -> username

@socketio.on('join')
def handle_join(data):
    username = data.get('username')
    if not username:
        emit('error', {'error': 'Username is required'})
        return
    session_id = request.sid
    users[session_id] = username
    emit('user_joined', {'username': username, 'session_id': session_id}, broadcast=True)
    print(f"{username} joined the chat. Session: {session_id}")

@socketio.on('send_message')
def handle_send_message(data):
    session_id = request.sid
    username = users.get(session_id, 'Unknown')
    message = data.get('message', '')
    analysis_enabled = data.get('analysis', False)
    analysis_result = None
    if analysis_enabled:
        try:
            processed_text = preprocess(message)
            if processed_text.strip():
                # Predict emotion
                emotion_features = emotion_vectorizer.transform([processed_text])
                emotion_prediction = emotion_model.predict(emotion_features)[0]
                emotion_label = emotion_label_encoder.inverse_transform([emotion_prediction])[0]
                emotion_confidence = float(np.max(emotion_model.predict_proba(emotion_features)))
                # Predict sentiment
                sentiment_features = sentiment_vectorizer.transform([processed_text])
                sentiment_prediction = sentiment_model.predict(sentiment_features)[0]
                sentiment_label = sentiment_label_encoder.inverse_transform([sentiment_prediction])[0]
                sentiment_confidence = float(np.max(sentiment_model.predict_proba(sentiment_features)))
                # Get GPT description
                gpt_desc = get_gpt_description(message, emotion_label, sentiment_label)
                analysis_result = {
                    'emotion': {
                        'label': emotion_label,
                        'confidence': emotion_confidence
                    },
                    'sentiment': {
                        'label': sentiment_label,
                        'confidence': sentiment_confidence
                    },
                    'gpt_description': gpt_desc
                }
        except Exception as e:
            analysis_result = {'error': f'Analysis failed: {str(e)}'}
    emit('receive_message', {
        'username': username,
        'message': message,
        'session_id': session_id,
        'analysis': analysis_result
    }, broadcast=True)
    print(f"{username}: {message} (analysis: {analysis_enabled})")

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    username = users.pop(session_id, None)
    if username:
        emit('user_left', {'username': username, 'session_id': session_id}, broadcast=True)
        print(f"{username} left the chat. Session: {session_id}")

# --- Database Models ---
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(512), nullable=False)
    messages = db.relationship('Message', backref='user', lazy=True)
    sessions = db.relationship('ChatSession', secondary='user_session', back_populates='participants')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime)
    messages = db.relationship('Message', backref='session', lazy=True)
    participants = db.relationship('User', secondary='user_session', back_populates='sessions')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)

# Association table for many-to-many relationship between users and sessions
user_session = db.Table('user_session',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('session_id', db.Integer, db.ForeignKey('chat_session.id'), primary_key=True)
)

if __name__ == '__main__':
    print("🚀 Starting Flask-SocketIO server...")
    print("📊 Models loaded:")
    print("   - Emotion: Naive Bayes (balanced)")
    print("   - Sentiment: Naive Bayes")
    print("🌐 Server will be available at: http://127.0.0.1:5000")
    print("🔗 API endpoint: http://127.0.0.1:5000/predict")
    print("💚 Health check: http://127.0.0.1:5000/health")
    
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, debug=False, host='0.0.0.0', port=port) 