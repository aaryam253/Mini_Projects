import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from pprint import pprint

cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("./trainingandtestdata/training.1600000.processed.noemoticon.csv", header=None, names=cols, encoding='latin-1')
# above line will be different depending on where you saved your data, and your file name
df.head()
df.sentiment.value_counts()

# drop columns that aren't needed for specific purpose for sentiment analysis
df.drop(['id','date','query_string','user'],axis=1,inplace=True)

# NOTE: Sentiment values are ordered as follows:
# 0 - Negative
# 2 - Neutral
# 4 - Positive

# check sentiment values of 0 and 4 
df[df.sentiment == 0].head(10)
df[df.sentiment == 4].head(10)

# check the length of each string in text column in each entry
df['pre_clean_len'] = [len(t) for t in df.text]

data_dict = {
    'sentiment':{
        'type':df.sentiment.dtype,
        'description':'sentiment class - 0:negative, 1:positive'
    },
    'text':{
        'type':df.text.dtype,
        'description':'tweet text'
    },
    'pre_clean_len':{
        'type':df.pre_clean_len.dtype,
        'description':'Length of the tweet before cleaning'
    },
    'dataset_shape':df.shape
}
pprint(data_dict)

# plot the distribution of length of strings in each entry
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.pre_clean_len)
plt.show()

# ensure string lengths are within 140 characters
df[df.pre_clean_len > 140].head(10)
df.text[279]

# DATA CLEANING PROCEDURES
# Step 1: Decode any HTMl encodings to general text
example1 = BeautifulSoup(df.text[279], 'lxml')
print example1.get_text()


