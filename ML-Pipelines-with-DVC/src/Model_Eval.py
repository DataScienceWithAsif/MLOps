import os
import pandas as pd
import numpy as np
import pickle
import json

from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score

test_data = pd.read_csv("./data/features/test_bow.csv")
model = pickle.load(open("model.pkl","rb"))

x_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

eval_metrics ={
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "AUC": auc
}

with open("eval_metrics.json", "w") as file:
    json.dump(eval_metrics, file, indent=4)