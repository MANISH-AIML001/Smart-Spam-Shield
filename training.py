from pathlib import Path
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import dataset_preprocessing, load_dataset, model_evaluation


ROOT = Path(__file__).resolve().parent
DATASETS = (( ROOT / "dataset/training/training_balanced.csv", "utf-8"),)
MODEL_PATH = ROOT / "models/v1/lr_model.pkl"
BEST_PARAMS = {
    "tfidf__ngram_range": (1, 2),
    "tfidf__sublinear_tf": True,
    "model__C": 2.0,
    "model__class_weight": "balanced",
}
DEFAULT_CLEAN_DATASET = ROOT / "dataset/training/clean_dataset.csv"

datasets = [
    (load_dataset(path, encoding), str(path))
    for path, encoding in DATASETS
]
data = dataset_preprocessing(datasets, DEFAULT_CLEAN_DATASET)
if data.empty or data["label"].nunique() < 2:
    raise ValueError("Training data must contain both ham and spam messages.")

X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"],
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=BEST_PARAMS["tfidf__ngram_range"],
        sublinear_tf=BEST_PARAMS["tfidf__sublinear_tf"],
    )),
    ("model", LogisticRegression(
        C=BEST_PARAMS["model__C"],
        class_weight=BEST_PARAMS["model__class_weight"],
        max_iter=2000,
        random_state=42,
    )),
])
pipeline.fit(X_train, y_train)

print("Best parameters:", BEST_PARAMS)
model_evaluation(y_test, pipeline.predict(X_test), pipeline)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with MODEL_PATH.open("wb") as model_file:
    pickle.dump(pipeline, model_file)
print("Model saved:", MODEL_PATH)
