import pickle
from pathlib import Path

from utils import dataset_preprocessing, load_dataset, model_evaluation

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset/testing/spam_assassin.csv.zip"
DEFAULT_MODEL = ROOT / "models/v1/lr_model.pkl"
DEFAULT_CLEAN_DATASET = ROOT / "dataset/testing/testing_set_1.csv"


def evaluate_model(dataset_path=DEFAULT_DATASET, model_path=DEFAULT_MODEL,
                   clean_dataset_path=DEFAULT_CLEAN_DATASET):
    with Path(model_path).open("rb") as model_file:
        pipeline = pickle.load(model_file)

    raw_data = load_dataset(dataset_path)
    data = dataset_preprocessing([(raw_data, str(dataset_path))], str(clean_dataset_path))
    if data.empty:
        raise ValueError("The test dataset is empty after preprocessing.")

    predictions = pipeline.predict(data["message"])
    return model_evaluation(data["label"], predictions, pipeline)


if __name__ == "__main__":
    evaluate_model()