import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

model = sys.argv[1]

file = f"models/{model}/dataframe.csv"

print("Preprocessing: Preparing for stratified sampling")
df = pd.read_csv(file, header=0)
y = df["label"]
del df["label"]

X_train, X_test, y_train, y_test = train_test_split(df, y, stratify=y, test_size=0.30)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
train.to_csv(f'models/{model}/train.csv',index=False)
test.to_csv(f'models/{model}/test.csv',index=False)