import os
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

train_data = pd.read_csv("./data/raw/train_data.csv")
test_data = pd.read_csv("./data/raw/test_data.csv")

nltk.download("wordnet")
nltk.download("stopwords")

def lemmmatization(text):
    lemmatizer = WordNetLemmatizer()

    text = text.split()

    text = [lemmatizer.lemmatize(y) for y in text]

    return " ".join(text)

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    text = text.split()
    text = [i for i in text if i not in stop_words]

    return " ".join(text)

def remove_num(text):
    text = [i for i in text if not i.isdigit()]
    return "".join(text)

def lower_case(text):
    text = text.split()

    text = [t.lower() for t in text]

    return " ".join(text)

def remove_punctuations(text):
    text = re.sub('[%s]' % re.escape("""!"`~#$%^&*()_-+=;',<>?/\\{|}"""), ' ', text)
    text = text.replace(':','')
    text = re.sub('\s+',' ', text)
    text = " ".join(text.split())

    return text.strip()

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def normalize_text(df):
    df.content = df.content.apply(lambda content: lower_case(content))
    df.content = df.content.apply(lambda content: remove_stopwords(content))
    df.content = df.content.apply(lambda content: remove_num(content))
    df.content = df.content.apply(lambda content: remove_punctuations(content))
    df.content = df.content.apply(lambda content: remove_urls(content))
    df.content = df.content.apply(lambda content: lemmmatization(content))

    return df

train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)

data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)

train_processed_data.to_csv(os.path.join(data_path, 'train_processed_data.csv'))
test_processed_data.to_csv(os.path.join(data_path, "test_processed_data.csv"))