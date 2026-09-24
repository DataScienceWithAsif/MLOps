import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import logging
import os

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.logs")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"parameters retrieved from path: {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"file not found at {params_path}")
        raise
    except Exception as e:
        logger.error(f"unexpected error: {e}")
        raise

def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"data loaded from: {data_url}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"failed to load the .csv file, {e}")
        raise
    except Exception as e:
        logger.error(f"while loading data unexpected error occured, {e}")
        raise

def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug(f"basic preprocessing of data of shape: {df.shape} done.")
        return df
    except KeyError as e:
        logger.error(f'missing required col in the dataset, {e}')
        raise
    except Exception as e:
        logger.error(f"unexpected behavior occured: {e}")
        raise 

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test_data.csv"), index=False)

        logger.debug(f"train and test data saved to {raw_data_path}")
    except Exception as e:
        logger.error(f"unexpected error occured, {e}")
        raise

def main():
    try:
        params = load_params(params_path=r"C:\Users\as296\Desktop\MLOps\YT_Insights\params.yaml")
        test_size = params['data_ingestion']['test_size']

        data_url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        df = load_data(data_url=data_url)

        df = basic_preprocess(df=df)

        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data"))
    except Exception as e:
        logger.error(f"failed t complete data ingestion phase, \n{e}")
        raise

if __name__ == "__main__":
    main()
