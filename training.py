from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from utils import dataset_preprocessing, load_dataset, model_evaluation


ROOT = Path(__file__).resolve().parent
DATASETS = (( ROOT / "dataset/training/training_balanced.csv", "utf-8"),)
MODEL_PATH = ROOT / "models/v1/lr_model.joblib"
param_grid = {
    "tfidf__max_features": [30000, 50000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "model__C": [1, 4]
}
DEFAULT_CLEAN_DATASET = ((ROOT / "dataset/training/clean_dataset.csv", "utf-8"),)
# protected the files from being run when the multiple worker process the training in parallel
if __name__ == "__main__":
    datasets = [
        (load_dataset(path, encoding), str(path))
        for path, encoding in DATASETS
    ]
    data = dataset_preprocessing(datasets)
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
        ("tfidf", TfidfVectorizer(dtype=np.float32)),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        pre_dispatch=1,
        verbose=2,
        return_train_score=False
    )
    grid.fit(X_train,y_train)
    best_model = grid.best_estimator_

    print("Best parameters:", grid.best_params_)
    model_evaluation(y_test, best_model.predict(X_test), best_model)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True) 
    joblib.dump(best_model, MODEL_PATH, compress=3)
    print("Model saved:", MODEL_PATH)
