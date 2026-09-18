import argparse
import pickle
from pathlib import Path

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from utils import dataset_preprocessing, load_dataset, model_evaluation

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    ROOT / "dataset/training/processed_data.csv.zip",
    ROOT / "dataset/training/archive (1).zip",
    ROOT / "dataset/training/spam.csv",
)
DEFAULT_MODEL = ROOT / "models/v2/lr_model.pkl"
DEFAULT_CLEAN_DATASET = ROOT / "dataset/training/clean-dataset.csv"


def build_search(random_state=42):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])
    parameter_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1],
        "tfidf__sublinear_tf": [True],
        "model__C": [0.5, 2.0],
        "model__class_weight": [None, "balanced"],
    }
    return GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="f1",
        cv=3,
        n_jobs=1,
        refit=True,
        verbose=1,
    )


def train_model(dataset_paths=DEFAULT_DATASETS, model_path=DEFAULT_MODEL,
                clean_dataset_path=DEFAULT_CLEAN_DATASET, random_state=42,
                max_samples=None):
    datasets = [(load_dataset(path, encoding="cp1252" if str(path).endswith("spam.csv") else "utf-8"), str(path))
                for path in dataset_paths]
    data = dataset_preprocessing(datasets, str(clean_dataset_path))
    if data.empty or data["label"].nunique() < 2:
        raise ValueError("Training data must contain both ham and spam labels.")
    if max_samples is not None and max_samples < len(data):
        data = data.groupby("label", group_keys=False).sample(
            n=max_samples // data["label"].nunique(), random_state=random_state,
        )
        print(f"Using a stratified sample of {len(data)} messages.")

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        data["message"], data["label"], test_size=0.2,
        random_state=random_state, stratify=data["label"],
    )
    search = build_search(random_state)
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    print("Cross-validation F1: %.4f" % search.best_score_)
    model_evaluation(y_holdout, search.predict(X_holdout), search.best_estimator_)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(search.best_estimator_, model_file)
    print(f"Model saved to {model_path}")
    return search.best_estimator_


def parse_args():
    parser = argparse.ArgumentParser(description="Tune and train the spam classifier.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional stratified sample size for faster local training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(model_path=args.model_path, random_state=args.random_state,
                max_samples=args.max_samples)