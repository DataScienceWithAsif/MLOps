import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier

train_df = pd.read_csv("./data/features/train_bow.csv")

x_train = train_df.iloc[:,0:-1].values
y_train = train_df.iloc[:,-1].values

model = GradientBoostingClassifier(n_estimators=50)

model.fit(x_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))