import pandas as pd

train = pd.read_csv("../dataset/train.csv")
test = pd.read_csv("../dataset/test.csv")

print(train.head())
print(train.info())
print(train['Category'].value_counts())
print(train['Misconception'].value_counts())
total_count = train['Misconception'].value_counts().sum()
print("Total count of 'Misconception':", total_count)
