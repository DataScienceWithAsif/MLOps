import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import logging
import pickle
import yaml
import os
import mlflow
import mlflow.sklearn

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
            params = yaml.safe_load(params_path)
        logger.debug(f"parameters loaded from {params_path}")
        # when mlflow setup is activated
        # for param, value in params.items():
        #     mlflow.log_param(param, value)

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

def load_vectorizer(vec_path: str):
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
        classification_rep = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap="Blues")
        plt.title("LGBM Model Confusion Matrix")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.savefig("LGBM_Confusion_Matrix.png")
        plt.show()

        # when mlflow setup is activated
        # mlflow.log_artifact("LGBM_Confusion_Matrix.png")
        # mlflow.log_metric("accuracy", accuracy)
        # for label, metrics in classification_rep.items():
        #     if isinstance(metrics, dict):
        #         for metric, value in metrics.items():
        #             mlflow.log_metric(f"{label}_{metric}", value)

        return accuracy, classification_rep
        logger.debug(f"Model Evaluation completed with accuracy: {accuracy}")

    except Exception as e:
        logger.error(f"error occured while model evaluation, \n{e}")
        print(f"Model evaluation error: {e}")


def get_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../"))


def main():
    try:
        logger.debug("Model Evaluation phase started...........")

        root_dir = get_root_dir()

        params = load_params(params_path=os.path.join(root_dir, "params.yaml"))
        vectorizer = load_vectorizer(vec_path=os.path.join(root_dir, "trained_model/tfidf_vctorizer.pkl"))
        test_data = load_data(data_path=os.path.join(root_dir, "data/preprocessing/preprocessed_test.csv"))
        model = load_model(model_path=os.path.join(root_dir, "trained_model/lgbm_model.pkl"))

        x_test = test_data['clean_comment']
        y_test = test_data['category']

        x_test_vec = apply_vectorizer(vectorizer=vectorizer, x_test=x_test)

        accuracy, classification_rep = evaluate_model(model=model, x_test_vec=x_test_vec, y_test=y_test)
        print(f"accuracy: \n{accuracy}")
        print("-------------------------------")
        print(f"classification report: \n{classification_rep}")

        # mlflow.sklearn.log_model(
        #     model,
        #     name="LGBMClassifier_model",
        #     skops_trusted_types=[
        #         "sklearn.tree._tree.Tree",
        #         "collections.OrderedDict",
        #         "lightgbm.basic.Booster",
        #         "lightgbm.sklearn.LGBMClassifier",
        #     ],
        # )

        logger.debug("\nModel evaluation phase completed!")


    except Exception as e:
        logger.error(f"error occured while model evaluation, \n{e}")
        print(f"Model evaluation error occured: {e}")


if __name__ == "__main__":
    main()

