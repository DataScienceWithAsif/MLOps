import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

train_data = pd.read_csv("./data/processed/train_processed_data.csv")
test_data = pd.read_csv("./data/processed/test_processed_data.csv")

train_data.fillna('',inplace=True)
test_data.fillna('', inplace=True)

x_train = train_data['content'].values
y_train = train_data['sentiment'].values
x_test = test_data['content'].values
y_test = test_data['sentiment'].values

count_vectorizer = CountVectorizer(max_features=50)

x_train_bow = count_vectorizer.fit_transform(x_train)
x_test_bow = count_vectorizer.transform(x_test)

train_df = pd.DataFrame(x_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(x_test_bow.toarray())
test_df['label'] = y_test

data_path = os.path.join("data", "features")
os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path, "train_bow.csv"))
test_df.to_csv(os.path.join(data_path, "test_bow.csv"))