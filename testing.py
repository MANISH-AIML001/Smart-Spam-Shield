import joblib
from pathlib import Path

from utils import dataset_preprocessing, load_dataset, model_evaluation


ROOT = Path(__file__).resolve().parent
DATASETS = (
    (ROOT / "dataset/testing/spam_messages_test.csv", "utf-8"),
    (ROOT / "dataset/testing/spam_assassin.csv", "utf-8"),
)
MODEL_PATH = ROOT / "models/v1/lr_model.joblib"
CLEAN_DATASET_PATH = ROOT / "dataset/testing/training_set_1.csv"


pipeline = joblib.load(MODEL_PATH)

datasets = [
    (load_dataset(path, encoding), str(path))
    for path, encoding in DATASETS
]
data = dataset_preprocessing(datasets)
if data.empty:
    raise ValueError("The test dataset is empty after preprocessing.")


model_evaluation(data["label"], pipeline.predict(data["message"]), pipeline)
