import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from pprint import pprint
from wordcloud import WordCloud

csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.head()
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df.info()

# First Step of Visualisation: WorldCloud
# Provides a quick and crude form of textual analysis on all the tweets data
neg_tweets = my_df[my_df.target == 0]
neg_string = []
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')

wordcloud = WordCloud(width=1600,height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Explore some clear false positives
for t in neg_tweets.text[:200]:
	if 'love' in t:
		print (t)

# Now look at the positive tweets and create its own worldcloud
pos_tweets = my_df[my_df.target == 1]
pos_string = []
for t in pos_tweets.text:
	pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')

worldcloud = WorldCloud(width=1600,height=800,max_font_size=200).generate(pos_string)
plt.figure(figsize=(12,10))
plt.imshow(worldcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
