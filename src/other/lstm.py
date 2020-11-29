import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
from sklearn.model_selection import train_test_split

def concat_data():
    script_dir = os.path.dirname(__file__)  # Script directory

    full_path = os.path.join(script_dir, './2007')

    all_folders = glob(full_path+"/*")

    frames = []
    for folder in all_folders:
        all_files = glob(folder + "/*.csv")
        li = []
        for filename in all_files:
            print(filename)
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        frame = pd.concat(li, axis=0, ignore_index=True)
        frames.append(frame)
    result = pd.concat(frames)

    result.to_csv(r'combined.csv', index=False)

def load_data():
    data = pd.read_csv("combined.csv", index_col=None, header=0)
    return data

df = load_data()
print(df.head())
print(df.tail())

# shuffle and reset the index
df = df.sample(frac=1).reset_index(drop=True)

print(df.head(20))

# split the data into test and train set
train_data, test_data = train_test_split(df, test_size=0.2)

print(train_data[1,:])

# training_set = train_data.iloc[:, 1:2].values

