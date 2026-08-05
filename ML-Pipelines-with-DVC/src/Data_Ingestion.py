import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
#Configuring logger
logger = logging.getLogger("Data_Ingestion")
logger.setLevel("DEBUG")

consule_handler = logging.StreamHandler()
consule_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("error.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
consule_handler.set_name(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(consule_handler)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------

def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        return df
    except pd.errors.ParserError as e:
        print(f"Error: failed to parse the csv file from path: {data_url}")
        print(e)
        raise
    except Exception as e:
        print(e)
        raise

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=['tweet_id'])
        df = df[df['sentiment'].isin(["happiness", "sadness"])]
        shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_content = pd.DataFrame()
        df_content['content'] = shuffled_df['content'].values
        df_sentiment = pd.DataFrame()
        df_sentiment['sentiment'] = shuffled_df['sentiment'].map({"happiness":1, "sadness":0})
        final_df = pd.concat([df_content, df_sentiment], axis=1)

        return final_df
    except KeyError as e:
        print(e)
        raise
    except Exception as e:
        print(e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train_data.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test_data.csv"), index=False)
    except Exception as e:
        print(e)
        raise


def main():
    try:
        df = load_data(data_url="https://raw.githubusercontent.com/entbappy/Branching-tutorial/refs/heads/master/tweet_emotions.csv")
        final_df = data_preprocessing(df=df)
        train_data, test_data = train_test_split(final_df, test_size=.2, random_state=42)
        save_data(train_data=train_data, test_data=test_data, data_path="data")
    except Exception as e:
        print(f"Error: {e}")
        print("n\nFailed to complete the data ingestion process")


if __name__ == "__main__":
    main()