import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pickle
import yaml
import json
import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("model_evaluation_errors.log")
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
        logger.debug(f"parameters loaded from {params_path}")

        return params
            
    
    except Exception as e:
        logger.error(f"error occured while loading params, \n{e}")
        print(f"params loading error: {e}")


def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        df.dropna(inplace=True)
        logger.debug(f"test data loaded of shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"error occured while loading data, \n{e}")
        print(f"data loading error: {e}")

def load_vectorizer(vec_path: str) -> TfidfVectorizer:
    try:
        with open(vec_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug(f"TF-IDF vctorizer loaded from {vec_path}")
        return vectorizer

    except Exception as e:
        logger.error(f"error occured while loading vectorizer, \n{e}")
        print(f"vectorizer loading error: {e}")

def apply_vectorizer(vectorizer, x_test: pd.DataFrame):
    try:
        x_test_vec = vectorizer.transform(x_test)
        logger.debug(f"TF-IDF transformation applied now it is shape of {x_test_vec.shape}")
        return x_test_vec
    except Exception as e:
        logger.error(f"error occured while vectorization, \n{e}")
        print(f"vectorization error: {e}")


def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"model loaded from {model_path}")
        return model

    except Exception as e:
        logger.error(f"error occured while loading model, \n{e}")
        print(f"model loading error: {e}")


def evaluate_model(model, x_test_vec, y_test):
    try:
        y_pred = model.predict(x_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.debug(f"Model Evaluation completed with accuracy: {accuracy}")
        return classification_rep, cm, accuracy

    except Exception as e:
        logger.error(f"error occured while model evaluation, \n{e}")
        print(f"Model evaluation error: {e}")

def log_cm(cm, dataset_name):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion matrix for - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_file_path = f"confusion_matrix_{dataset_name}.png"
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {
            "run_id": run_id,
            "model_path": model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug(f"model inf saved to {file_path}")

    except Exception as e:
        logger.error(f"error occured while saving model info, \n{e}")
        raise



def get_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../"))


def main():
    
    logger.debug("Model Evaluation phase started...........")

    mlflow.set_tracking_uri("http://ec2-13-126-176-114.ap-south-1.compute.amazonaws.com:5000/")
    mlflow.set_experiment("dvc-pipeline-runs")

    with mlflow.start_run() as run:
        try:

            root_dir = get_root_dir()
            params = load_params(params_path=os.path.join(root_dir, "params.yaml"))
            for param, val in params.items():
                mlflow.log_param(param, val)


            vectorizer = load_vectorizer(vec_path=os.path.join(root_dir, "trained_model/tfidf_vctorizer.pkl"))
            test_data = load_data(data_path=os.path.join(root_dir, "data/preprocessing/preprocessed_test.csv"))
            model = load_model(model_path=os.path.join(root_dir, "trained_model/lgbm_model.pkl"))
            x_test = test_data['clean_comment'].values
            y_test = test_data['category'].values
            x_test_vec = apply_vectorizer(vectorizer=vectorizer, x_test=x_test)

            input_exp = pd.DataFrame(x_test_vec.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_exp, model.predict(x_test_vec[:5]))

            mlflow.sklearn.log_model(
                model,
                "LGBM_model",
                signature=signature,
                input_example=input_exp
            )

            artifact_uri = mlflow.get_artifact_uri()
            model_path = f"{artifact_uri}/LGBM_model"
            save_model_info(run.info.run_id, model_path=model_path, file_path="model_info.json")

            mlflow.log_artifact(os.path.join(root_dir, "trained_model/tfidf_vctorizer.pkl"))

            classification_rep, cm, accuracy = evaluate_model(model=model, x_test_vec=x_test_vec, y_test=y_test)
            mlflow.log_metric("accuracy", accuracy)

            for label, metrics in classification_rep.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f'test_{label}_precision': metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            log_cm(cm, "Test_Data")

            mlflow.set_tag("model_type", "LGBM_model")
            mlflow.set_tag("task", "YT sentiment analysis")
            mlflow.set_tag("dataset", "youtube comments")
      

        

            logger.debug("\nModel evaluation phase completed!")


        except Exception as e:
            logger.error(f"error occured while model evaluation, \n{e}")
            print(f"Model evaluation error occured: {e}")


if __name__ == "__main__":
    main()

