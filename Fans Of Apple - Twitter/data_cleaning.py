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
print (example1.get_text())

# Step 2: Remove redundant '@' mention
df.text[343]
re.sub(r'@[A-Za-z0-9]+','',df.text[343])

# Step 3: Remove redundant URL links
df.text[0]
re.sub('https?://[A-Za-z0-9./]+','',df.text[0])

# Step 4: Decode UTF-8 encoding
testing = df.text[226]
testing
testing.replace(u'\ufffd','?')

# Step 5: Remove any hashtag '#' characters
df.text[175]
re.sub('[^a-zA-Z]', ' ', df.text[175])

# Create a data cleaning function
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

testing = df.text[:100]
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))
test_result

# clean the tweets data
nums = [0,400000,800000,1200000,1600000]
print ("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(nums[0],nums[1]):
    if( (i+1)%10000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, nums[1] ))                                                                  
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))

# save the cleaned data in CSV format
clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = df.sentiment
clean_df.head()

clean_df.to_csv('clean_tweet.csv',encoding='utf-8')
csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.head()
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df.info()





