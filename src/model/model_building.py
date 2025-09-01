import numpy as np
import pandas as pd
import os 
import pickle 
import yaml
import logging
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_building_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retreived from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded and NaNs filled from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data: %s", e)
        raise

def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        x_train = train_data['text']
        y_train = train_data['label']

        x_train_tfidf = vectorizer.fit_transform(x_train)

        logger.debug(f"TF-IDF transformation complete. Train shape: {x_train_tfidf.shape}")

        with open(os.path.join(get_root_dir(), 'tfidf_vectorizer.pkl'), "wb") as f:
            pickle.dump(vectorizer, f)

        logger.debug("TF-IDF applied and data transformed")
        return x_train_tfidf, y_train
    except Exception as e:
        logger.error("Error during TF-IDF transformation %s", e)
        raise

def train_xgb(x_train: np.ndarray, y_train: np.ndarray, lr: float, max_depth: int, n_estimators: int) -> xgb.XGBClassifier:
    try:
        best_model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, random_state=42)
        best_model.fit(x_train, y_train)
        logger.debug("XGBoost model training completed")
        return best_model
    except Exception as e:
        logger.error("Error during XGBoost model training: %s", e)

def get_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../"))

def save_model(model, file_path: str) -> None:
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug("Model, saved to %s", file_path)
    except Exception as e:
        logger.error("Error occured while saving the model: %s", e)
        raise

def main():
    try: 
        root_dir = get_root_dir()

        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])

        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        train_data = load_data(os.path.join(root_dir, 'data/raw/train.csv'))

        x_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        best_model = train_xgb(x_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        save_model(best_model, os.path.join(root_dir, "xgb_model.pkl"))
    except Exception as e:
        logger.error("Failed to complete the feature engineering and model building process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
