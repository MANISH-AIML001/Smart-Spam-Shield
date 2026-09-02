"""
SMART SPAM SHIELD - Flask Backend
 AIML Project

Handles:
1. Authentication (signup / login / logout) with a local SQLite database
   and hashed passwords (werkzeug.security) — ships with Flask, no extra
   install needed.
2. The trained model (spam_model.pkl + tfidf_vectorizer.pkl) served at
   /predict, only reachable once a user is signed in.

Run: python app_flask.py
Then open: http://127.0.0.1:5000
"""

import re
import string
import pickle
import sqlite3
from functools import wraps

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

URGENCY_WORDS = [
    "free", "win", "winner", "cash", "prize", "urgent", "congratulations",
    "click", "claim", "offer", "limited", "act now", "call now", "credit",
    "loan", "guarantee", "risk-free", "bonus", "voucher", "selected"
]

DB_PATH = "users.db"

app = Flask(__name__)
# Change this before deploying anywhere public — fine for a local
# college demo but a real secret should not live in source code.
app.secret_key = "ab0d7daa7108ff5d7954c898df82c9b10deb409b7168778e"


# -------------------- Database setup --------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# -------------------- Load trained artifacts --------------------
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


def extract_features(msg):
    raw = str(msg)
    length = len(raw)
    num_digits = sum(c.isdigit() for c in raw)
    num_upper = sum(c.isupper() for c in raw)
    num_special = sum(c in "!$%*#@" for c in raw)
    num_urls = len(re.findall(r"http\S+|www\S+", raw))
    num_phone = len(re.findall(r"\b\d{10}\b", raw))
    exclm_count = raw.count("!")
    upper_ratio = num_upper / length if length > 0 else 0
    urgency_score = sum(1 for w in URGENCY_WORDS if w in raw.lower())
    return [length, num_digits, num_upper, num_special,
            num_urls, num_phone, exclm_count, upper_ratio, urgency_score]


FEATURE_LABELS = {
    "length": "Message length",
    "num_digits": "Digits",
    "num_upper": "Capital letters",
    "num_special": "Special symbols (!$%*#@)",
    "num_urls": "Links found",
    "num_phone": "Phone numbers found",
    "exclm_count": "Exclamation marks",
    "upper_ratio": "Capital letter ratio",
    "urgency_score": "Urgency-word score",
}


# -------------------- Auth routes --------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not username or not password:
            error = "Username and password can't be empty."
        elif len(password) < 4:
            error = "Password should be at least 4 characters."
        elif password != confirm:
            error = "Passwords don't match."
        else:
            conn = get_db()
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing:
                error = "That username is already taken."
                conn.close()
            else:
                conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                conn.commit()
                conn.close()
                return redirect(url_for("login", signed_up="1"))

    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    just_signed_up = request.args.get("signed_up") == "1"

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))
        error = "Incorrect username or password."

    return render_template("login.html", error=error, just_signed_up=just_signed_up)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -------------------- App routes --------------------
@app.route("/")
@login_required
def home():
    return render_template("index.html", username=session.get("username"))


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json(force=True)
    msg = data.get("message", "").strip()

    if not msg:
        return jsonify({"error": "empty_message"}), 400

    clean = clean_text(msg)
    behav = extract_features(msg)
    X_text = tfidf.transform([clean])
    X_combined = hstack([X_text, csr_matrix([behav])])

    pred = int(model.predict(X_combined)[0])
    proba = model.predict_proba(X_combined)[0]
    spam_conf = round(float(proba[1]) * 100, 1)
    safe_conf = round(100 - spam_conf, 1)

    # Three-tier verdict, based on how "safe" the message scores:
    #   80-100  -> safe
    #   50-79.9 -> suspicious
    #   below 50 -> spam
    if safe_conf >= 80:
        verdict = "safe"
    elif safe_conf >= 50:
        verdict = "suspicious"
    else:
        verdict = "spam"

    triggers = [w for w in URGENCY_WORDS if w in msg.lower()]

    feature_table = [
        {"label": FEATURE_LABELS[name], "value": round(val, 2) if isinstance(val, float) else val}
        for name, val in zip(FEATURE_LABELS.keys(), behav)
    ]

    return jsonify({
        "verdict": verdict,
        "spam_confidence": spam_conf,
        "safe_confidence": safe_conf,
        "triggers": triggers,
        "features": feature_table,
    })


if __name__ == "__main__":
    app.run(debug=True)