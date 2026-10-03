import json
import mlflow
import logging
import os

mlflow.set_tracking_uri("http://ec2-3-110-56-241.ap-south-1.compute.amazonaws.com:5000/")

#logging configuration
logger = logging.getLogger("model_registration")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug(f"model info loaded from {file_path}")
        return model_info
    
    except Exception as e:
        logger.error(f"error occured during loading json file from {file_path}, error is: \n{e}")
        raise


def register_model(model_name: str, model_info: dict) -> None:
    try:
        model_path = model_info["model_path"]
        if model_path.startswith(("s3://", "file://", "dbfs:/")):
            model_path = model_path.rsplit("/", 1)[-1]
        model_uri = f"runs:/{model_info['run_id']}/{model_path}"

        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="staging"
        )

        logger.debug(f"Model {model_name} version {model_version} registered and transitioned to staging")

    except Exception as e:
        logger.error(f"error occured while registering model, error is: \n{e}")
        raise


def main():
    try:
        model_info_path = "model_info.json"
        model_info = load_model_info(file_path=model_info_path)

        model_name = "yt_chrome_plugin_model"
        register_model(model_name=model_name, model_info=model_info)

    except Exception as e:
        logger.error(f"failed to complete model registration, error is: \n{e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
