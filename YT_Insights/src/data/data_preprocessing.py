import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
file_handler = logging.FileHandler("preprocessing_errors.log")
file_handler.setLevel("ERROR")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download("stopwords")
nltk.download("wordnet")

def preprocess_comment(comment):
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # intially we did not remove stop words if accuracy drops then we will do
        # stop_words = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
        # comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        logger.debug("intial preprocessing done")
        return comment

    except Exception as e:
        logger.error(f"error occured in prerocessing, {e}")
        return comment

def normalize_text(df):
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug("text normalization completed")
        return df
    except Exception as e:
        logger.error(f"during normalization error occured, {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        preprocessing_data_path = os.path.join(data_path, 'preprocessing')
        os.makedirs(preprocessing_data_path, exist_ok=True)
        logger.debug(f"Directory: {preprocessing_data_path} created or already existed")

        train_data.to_csv(os.path.join(preprocessing_data_path, "preprocessed_train.csv"), index=False)
        test_data.to_csv(os.path.join(preprocessing_data_path, "preprocessed_test.csv"), index=False)
        logger.debug(f"processed data saved to {preprocessing_data_path}")
    except Exception as e:
        logger.error(f"error occured during saving data, {e}")
        raise

def main():
    try:
        logger.debug("Preprocessing started.................")


        train_data = pd.read_csv('./data/raw/train_data.csv')
        test_data = pd.read_csv('./data/raw/test_data.csv')

        processed_train_data = normalize_text(train_data)
        processed_test_data = normalize_text(test_data)

        save_data(processed_train_data, processed_test_data, data_path="./data")
        logger.debug("Preprocessing done!!!")

    except Exception as e:
        logger.error(f"failed to complete preprocessing step, \n{e}")
        print(f"Error: \n{e}")
        raise

if __name__ == "__main__":
    main()