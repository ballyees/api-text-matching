import numpy as np
import pandas as pd

df = pd.read_csv('keyword.csv', sep='|')

# print(df.head())
word = df['word'].apply(lambda s: s.split('/sp/')).tolist()
tmp = []
for wd in word:
    for w in wd:
        tmp.append(w)
word = np.unique(tmp)
word = pd.Series(word, name='unique')
word.to_csv('unique_keyword.csv', index=False)