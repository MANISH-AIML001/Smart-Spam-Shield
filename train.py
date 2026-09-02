"""
SMART SPAM SHIELD - Training Script
 AIML Project

Ye script normal "TF-IDF + Naive Bayes" wale generic spam detector se alag hai.
Ismein hum 2 tarah ke features combine kar rahe hain:
   1) TF-IDF text features (jo word patterns pakadte hain)
   2) Hand-crafted "behavioural" features (jo spam messages ke structure/behaviour
      ko pakadte hain - jaise URL count, currency symbol, capital letters ratio,
      urgency words, phone numbers, etc.)

Isse model sirf words pe depend nahi karta, balki message ke "style" ko bhi
samajhta hai - isliye naye / unseen spam patterns pe bhi better generalize karta hai.

Author: <Manish Kumar>
"""

import re
import string
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# -----------------------------------------------------------------------
# STEP 1: Load dataset
# -----------------------------------------------------------------------
# Dataset: SMS Spam Collection Dataset (UCI / Kaggle)
# Download link diya gaya hai README.md mein.
# File format expected: CSV with 2 columns -> label (ham/spam), message
# -----------------------------------------------------------------------

def load_data(path="spam.csv"):
    df = pd.read_csv(path, encoding="latin-1")
    # Kaggle wali file mein extra unnamed columns hote hain, unko hata rahe hain
    df = df[[df.columns[0], df.columns[1]]]
    df.columns = ["label", "message"]
    df.drop_duplicates(inplace=True)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


# -----------------------------------------------------------------------
# STEP 2: Text cleaning
# -----------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # urls hatao
    text = re.sub(r"\d+", " ", text)                      # numbers hatao
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


# -----------------------------------------------------------------------
# STEP 3: Hand-crafted "behavioural" features (ye hi unique part hai)
# -----------------------------------------------------------------------
URGENCY_WORDS = [
    "free", "win", "winner", "cash", "prize", "urgent", "congratulations",
    "click", "claim", "offer", "limited", "act now", "call now", "credit",
    "loan", "guarantee", "risk-free", "bonus", "voucher", "selected"
]

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

    return [
        length, num_digits, num_upper, num_special,
        num_urls, num_phone, exclm_count, upper_ratio, urgency_score
    ]


FEATURE_NAMES = [
    "length", "num_digits", "num_upper", "num_special",
    "num_urls", "num_phone", "exclm_count", "upper_ratio", "urgency_score"
]


# -----------------------------------------------------------------------
# STEP 4: Build the full pipeline (TF-IDF + handcrafted features)
# -----------------------------------------------------------------------
def main():
    print("Loading dataset...")
    df = load_data("spam.csv")

    print("Cleaning text...")
    df["clean_message"] = df["message"].apply(clean_text)

    print("Extracting behavioural features...")
    behav_features = np.array(df["message"].apply(extract_features).tolist())

    print("Building TF-IDF vectors...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(df["clean_message"])

    # Combine TF-IDF sparse matrix + handcrafted dense features
    X_combined = hstack([X_text, csr_matrix(behav_features)])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training ensemble model (LogReg + RandomForest + SVM Voting)...")
    clf1 = LogisticRegression(max_iter=1000, C=2.0)
    clf2 = RandomForestClassifier(n_estimators=200, random_state=42)
    clf3 = SVC(probability=True, kernel="linear")

    model = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("svc", clf3)],
        voting="soft"
    )
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("Saving model + vectorizer...")
    with open("spam_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("\nDone! Files created: spam_model.pkl, tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()