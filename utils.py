import ast
import email
import html
import re
import zipfile
from email.policy import default
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

URGENCY_WORDS = [
    "free", "win", "winner", "cash", "prize", "urgent", "congratulations",
    "click", "claim", "offer", "limited", "act now", "call now", "credit",
    "loan", "guarantee", "risk-free", "bonus", "voucher", "selected"
]


def load_dataset(path, encoding="utf-8") -> pd.DataFrame:
    """Load a CSV using the encoding specified for its source dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding=encoding)


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

def ensure_label_message_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize common dataset headers to the label/message pair.
    df = df.copy()

    if {"label", "message"}.issubset(df.columns):
        return df
    if {"text", "target"}.issubset(df.columns):
        return df.rename(columns={"text": "message", "target": "label"})
    if {"label", "text"}.issubset(df.columns):
        return df.rename(columns={"text": "message"})

    if len(df.columns) < 2:
        raise ValueError("Dataset must contain at least two columns.")

    df = df.iloc[:, :2].copy()
    df.columns = ["label", "message"]
    return df


def dataset_preprocessing(
    datasets: list[tuple[pd.DataFrame, str]],
    saving_path: str | Path | None = None,
) -> pd.DataFrame:
    # Combine datasets and normalize their labels and message text.
    if not datasets:
        raise ValueError("Pass at least one dataset as a (dataframe, source) pair.")

    normalized = [
        ensure_label_message_columns(frame)
        for frame, _source in datasets
    ]
    data = pd.concat(normalized, ignore_index=True)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.drop(columns=["Unnamed:0"], errors="ignore")
    data = data.dropna(axis=1, how="all")
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    data = data.dropna(subset=["message"])
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].map(clean_raw_email)
    data = data[data["label"].isin({"ham", "spam", "0", "1"})]
    data["label"] = data["label"].map({"spam": 1, "ham": 0, "0": 0, "1": 1})
    data["label"] = data["label"].astype(int)
    data = data.drop_duplicates()

    if saving_path:
        data.to_csv(saving_path, index=False)
    return data


def clean_raw_email(raw_email_string) -> str:
    # Normalize plain text, serialized Enron lines, and raw emails.
    if raw_email_string is None or (
        not isinstance(raw_email_string, (list, tuple, dict))
        and pd.isna(raw_email_string)
    ):
        return ""

    if isinstance(raw_email_string, (list, tuple)):
        text = "\n".join(str(line) for line in raw_email_string)
    else:
        text = str(raw_email_string).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                lines = ast.literal_eval(text)
                if isinstance(lines, list):
                    text = "\n".join(str(line) for line in lines)
            except (ValueError, SyntaxError):
                pass

    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"(?i)\b(?:https?://|www\.)\S+", "URL_TOKEN", text)
    text = re.sub(r"\d+", "NUMBER_TOKEN", text)
    text = re.sub(
        r"(?im)(^|\n)(disclaimer|original message|forwarded by|thanks,)\b.*$",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?i)^subject\s*:\s*", "", text, count=1).strip()
    return text

def model_evaluation(Y_test, Y_pred, model=None):
    metrics = {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, zero_division=0),
        "recall": recall_score(Y_test, Y_pred, zero_division=0),
        "f1": f1_score(Y_test, Y_pred, zero_division=0),
    }
    print('='*50)
    print('Prediction model ', model or 'unknown')
    print('='*50)
    print('Classification report:\n', classification_report(Y_test, Y_pred, digits=4, zero_division=0))
    print('Confusion Matrix: \n', confusion_matrix(Y_test, Y_pred))
    print("Performance Matrix")
    for name, value in metrics.items():
        print(f"{name.title()}: {value:.4f}")
    return metrics

def check_dataset(df):
    # Check the dataset
    print(df.shape)
    print(df.info())
    print(df.isnull().sum())
    print(df.duplicated().sum())
    print(df.columns)
    print(df['label'].dtype)
    print(df['label'].value_counts(), df['message'].value_counts())
    print(df['message'].dtype)
    print(df.head())
