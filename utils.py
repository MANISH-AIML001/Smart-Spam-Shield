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
import html
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

    return text


def clean_raw_email(raw_email_string):
    """Extract and normalize the readable body of a raw or flattened email."""
    if raw_email_string is None or (not isinstance(raw_email_string, (list, tuple, dict)) and pd.isna(raw_email_string)):
        return ""

    text = str(raw_email_string).strip()
    if not text:
        return ""

    # Some datasets serialize a list of message lines as a Python literal.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                text = "\n".join(str(item) for item in parsed).strip()
        except (ValueError, SyntaxError):
            pass

    # Strip an mbox separator only when it is a standalone first line, or a
    # recognizable flattened mbox prefix. Never apply `From .*` to a whole
    # newline-free record: several source CSVs flatten the complete email.
    mbox_prefix = re.compile(
        r"^From\s+\S+\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
        r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}\s*"
    )
    text = re.sub(r"(?m)^From [^\r\n]*\r?\n", "", text, count=1)
    text = mbox_prefix.sub("", text, count=1).strip()

    # Recreate line breaks between headers in CSVs that flattened RFC822
    # messages to one line. Consume only the leading run of header fields.
    header_names = (
        r"return-path|delivered-to|delivery-date|received|message-id|"
        r"resent-message-id|resent-from|resent-to|from|to|cc|bcc|reply-to|"
        r"sender|subject|date|mime-version|content-type|content-transfer-encoding|"
        r"content-disposition|user-agent|x-[a-z0-9-]+|errors-to|lines|references|"
        r"in-reply-to|mail-followup-to|organization|newsgroups|path|list-id|precedence|status|comments|keywords|url|authentication-results|dkim-signature"
    )
    header_line = re.compile(rf"(?i)^\s*(?:{header_names})\s*:")
    looks_like_email = bool(header_line.match(text))
    if looks_like_email and "\n" not in text and "\r" not in text:
        # Match candidate header names once with a lightweight scanner, then
        # insert boundaries only for recognized fields.
        # Match candidate header names once with a lightweight scanner, then
        # insert boundaries only for recognized fields.
        boundary = re.compile(r"(?i)(?<!\S)([a-z][a-z0-9-]{0,60})\s*:")
        known_headers = set(header_names.lower().split("|"))
        head, tail = text[:8192], text[8192:]
        pieces = []
        cursor = 0
        for match in boundary.finditer(head):
            name = match.group(1).lower()
            if (name in known_headers or name.startswith("x-")) and match.start() > cursor:
                pieces.append(head[cursor:match.start()])
                pieces.append("\n")
                cursor = match.start()
        pieces.extend((head[cursor:], tail))
        text = "".join(pieces)

    body_text = text
    if looks_like_email:
        lines = text.splitlines()
        header_lines = []
        first_body_line = len(lines)
        for index, line in enumerate(lines):
            if header_line.match(line):
                header_lines.append(line.strip())
            else:
                first_body_line = index
                break

        if header_lines:
            body_text = "\n".join(lines[first_body_line:]).strip()
            if not body_text:
                # Some flattened archives attach body text to the last MIME
                # transfer-encoding header without a visible line break.
                encodings = list(re.finditer(
                    r"(?i)content-transfer-encoding\s*:\s*(?:7bit|8bit|binary|base64|quoted-printable)\b\s*",
                    text,
                ))
                if encodings:
                    body_text = text[encodings[-1].end():].strip()
                else:
                    beenthere = re.search(
                        r"(?i)x-beenthere\s*:\s*[^\s]+\s+(.+)$", text
                    )
                    if beenthere:
                        body_text = beenthere.group(1).strip()
            message_source = "\n".join(header_lines) + "\n\n" + body_text
            try:
                message = email.message_from_string(message_source, policy=default)
                plain_parts = []
                html_parts = []
                for part in message.walk():
                    if part.is_multipart() or part.get_content_disposition() == "attachment":
                        continue
                    if part.get_content_maintype() != "text":
                        continue
                    try:
                        payload = part.get_content()
                    except (LookupError, UnicodeError, KeyError):
                        payload = part.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            payload = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                    if not isinstance(payload, str) or not payload.strip():
                        continue
                    if part.get_content_subtype() == "plain":
                        plain_parts.append(payload)
                    elif part.get_content_subtype() == "html":
                        html_parts.append(payload)
                if plain_parts:
                    body_text = "\n".join(plain_parts)
                elif html_parts:
                    body_text = "\n".join(html_parts)
                elif not body_text:
                    body_text = str(message.get("Subject", ""))
            except (ValueError, TypeError, LookupError):
                # Keep the separated body when malformed headers defeat parsing.
                pass

    # Remove HTML markup, decode entities, and normalize volatile values.
    text = html.unescape(re.sub(r"<[^>]+>", " ", body_text))
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
