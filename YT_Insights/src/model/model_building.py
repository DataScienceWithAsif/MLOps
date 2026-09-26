from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import os
import yaml
import pickle
import logging

# logging configuration
logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
file_handler = logging.FileHandler("model_building_errors.log")
file_handler.setLevel("ERROR")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug(f"params loaded from {file_path}")
        return params

    except yaml.YAMLError as e:
        logger.error("yaml error: {e}")
        raise
    
    except Exception as e:
        logger.error(f"unxpected behavior occured, {e}")
        raise

def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        df.dropna(inplace=True)
        logger.debug(f"train data loaded from path: {data_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"failed to parse the file: {data_path}")
        raise
    except Exception as e:
        logger.error(f"error occured while loading data from {data_path}")
        raise

def apply_vectorization(train_data: pd.DataFrame, ngram_range: tuple, max_featres: int):
    try:
        vectorizer = TfidfVectorizer(max_features=max_featres, ngram_range=ngram_range)

        x_train = train_data['clean_comment']
        y_train = train_data['category']

        x_train = vectorizer.fit_transform(x_train)
        logger.debug(f"TF-IDF vectorization complete. shape is: {x_train.shape}")

        with open(os.path.join(get_root_dir(), "trained_model/tfidf_vctorizer.pkl"), 'wb') as f:
            pickle.dump(vectorizer, f)

        return x_train, y_train
    except Exception as e:
        logger.error(f"error occured while apply transformation")
        raise

def build_model(x_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int):
    try:
        lgbm_model = LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        lgbm_model.fit(x_train, y_train)
        logger.debug("LGBMClassifier model training complete.")
        return lgbm_model
    except Exception as e:
        logger.error("error occured during model training")
        raise RuntimeError("model training failed") from e


def save_model(model, file_path: str):
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug("model saved successfully")
    except Exception as e:
        logger.error("error occured while saving model")
        raise

def get_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../"))

def main():
    try:
        root_dir = get_root_dir()

        logger.debug("Model building phase started.............")

        params = load_params(file_path=os.path.join(root_dir, "params.yaml"))
        max_features = params['model_building']['max_features']
        n_estimators = params['model_building']['n_estimators']
        max_depth = params['model_building']['max_depth']
        learning_rate = params["model_building"]['learning_rate']
        ngram_range = tuple(params['model_building']['ngram_range'])

        train_data = load_data(os.path.join(root_dir, "data/preprocessing/preprocessed_train.csv"))

        x_train, y_train = apply_vectorization(train_data=train_data, max_featres=max_features, ngram_range=ngram_range)

        model = build_model(x_train=x_train, y_train=y_train, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

        save_model(model=model, file_path=os.path.join(root_dir, "trained_model/lgbm_model.pkl"))
        logger.debug("model building phase completed!")        

    except Exception as e:
        logger.error(f"error occured while building model, {e}")
        print(f"error: {e}")


if __name__ == "__main__":
    main()