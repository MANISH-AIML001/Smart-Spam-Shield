import re
import zipfile
import pandas as pd 
import numpy as np
import ast
from pathlib import Path
from sklearn.metrics import( accuracy_score, classification_report, confusion_matrix, 
precision_score, f1_score, recall_score
)
import email
from email.policy import default
import re

URGENCY_WORDS = [
    "free", "win", "winner", "cash", "prize", "urgent", "congratulations",
    "click", "claim", "offer", "limited", "act now", "call now", "credit",
    "loan", "guarantee", "risk-free", "bonus", "voucher", "selected"
]


def load_dataset(path, encoding="utf-8") -> pd.DataFrame:
    """Load a CSV dataset from disk or from a ZIP archive."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() != ".zip":
        return pd.read_csv(path, encoding=encoding)

    with zipfile.ZipFile(path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"ZIP archive contains no CSV file: {path}")
        with archive.open(csv_members[0]) as csv_file:
            return pd.read_csv(csv_file, encoding=encoding)

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
    df = df.copy()

    if {"label", "message"}.issubset(df.columns):
        return df
    if {'text', 'target'}.issubset(df.columns):
        return df.rename(columns={'text': 'message', 'target': 'label'})
    if {'label', 'text'}.issubset(df.columns):
        return df.rename(columns={'text': 'message'})
    
    if len(df.columns) < 2:
        raise ValueError("Dataset must contain at least two columns.")

    # Both current datasets store label and message in the first two columns.
    df = df.iloc[:, :2].copy()
    df.columns = ["label", "message"]
    return df

def dataset_preprocessing(dataset : list[tuple[pd.DataFrame, str]], saving_path :str = None) -> pd.DataFrame:
    if not dataset :
        raise ValueError("DataFrame is empty. Pass atleat one panda dataframe.")

    normalized = []
    for df, path in dataset:

        df = ensure_label_message_columns(df)
        normalized.append(df)

    if len(dataset) > 1:
        df = pd.concat(normalized, ignore_index=True)
    else:
        df = normalized[0]
    #df = df.copy()

    df = df.replace([np.inf, -np.inf], np.nan)  # Repalce non-finite values inf with nan
    
    df= df.drop(columns= ["Unnamed:0"],errors="ignore")    # Drop the unnamed columns
   
    df = df.dropna(axis=1, how='all')    #Drop completely empty columns
    
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]    # Drop the columns which name is stating with Unnamed
    
    df = df.dropna(subset=['message'])      # Drop the column missing the message

    df["label"] = df["label"].astype(str).str.strip().str.lower()   # Lowercase the labels and remove whitespaces from messages
    df['message'] = df['message'].astype(str)
    df["message"] = df["message"].apply(clean_raw_email)   # Clean the raw emails in the dataset
    
    df = df[df['label'].isin(['ham', 'spam', 0, 1, '0', '1'])]  # Drop the column thats label is not ham or spam
    df = df.dropna(subset=['label'])
    
    df = df.drop_duplicates()   # Drop the duplicate columns
    
    df["label"]= df["label"].map({  
    'spam':1,
    'ham':0,
     '0':0,
     '1':1
    })  # Map 0 to ham and 1 to spam
    df['label'] = df['label'].astype(int)   # Ensures that labels are all of int type
    if saving_path:
        df.to_csv(saving_path, index=False)    # Save dataset to the given path
   
    return df

def clean_email_message(text):
    text = str(text).strip()
    
    if not (text.startswith("[") and text.endswith("]")):
        return text
    try:
        parsed = ast.literal_eval(text) # Clean the emails in enron dataset that are stored as a list of strings

        if isinstance(parsed, list):
            text = " ".join(str(x) for x in parsed)
        
    except (ValueError, SyntaxError):
        pass

    return text.strip()


def clean_raw_email(raw_email_string):
    text = str(raw_email_string or "").strip()
    if not text:
        return ""

    # Handle list-like strings such as "['Subject: ...']" and similar serialized values.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                text = " ".join(str(x) for x in parsed)
        except (ValueError, SyntaxError):
            pass

    # Remove common mbox separators.
    text = re.sub(r'(?m)^From .*\r?\n?', '', text).strip()

    # Prefer the actual body when the raw text is a full email with headers.
    if re.search(r'\r?\n\r?\n', text):
        parts = re.split(r'\r?\n\r?\n', text, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            text = parts[1].strip()

    # If the remaining content still has only header-like lines, keep the text after
    # stripping leading metadata lines. This covers list-backed Enron-style records.
    if not re.search(r'\r?\n\r?\n', text):
        lines = text.splitlines()
        body_lines = []
        header_block_done = False

        for line in lines:
            stripped = line.strip()
            if not header_block_done:
                if re.match(r'^(from|to|cc|bcc|subject|date|received|message-id|mime-version|content-type|content-transfer-encoding|reply-to|sender|return-path|delivered-to|errors-to|x-)', stripped, flags=re.IGNORECASE):
                    continue
                header_block_done = True
            body_lines.append(line)

        if body_lines:
            text = "\n".join(body_lines).strip()

    # Clean text formatting.
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\.\S+', 'URL_TOKEN', text)
    text = re.sub(r'\d+', 'NUMBER_TOKEN', text)
    text = re.sub(r'(?im)(^|\n)(disclaimer|original message|forwarded by|thanks,).*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Subject-only records should keep their subject text without the Subject: prefix.
    text = re.sub(r'(?i)^subject\s*:\s*', '', text, count=1).strip()

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







