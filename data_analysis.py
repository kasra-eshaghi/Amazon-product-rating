# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:43:10 2020

@author: Kasra
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer 

# import data
df = pd.read_json('train.json', lines=True)
# drop unnecessary columns
df.drop('image', inplace = True, axis = 1)
df.drop('reviewTime', inplace = True, axis = 1)
df.drop('reviewHash', inplace = True, axis = 1)

# clean up price data:
df['price'] = df['price'].replace({'\$':''}, regex = True)

# clean up review text and summary text:
df['reviewText'] = df['reviewText'].replace(np.nan,'', regex = True)
df['summary'] = df['summary'].replace(np.nan,'', regex = True)

#%% Look at rating for each music category over time
categories = df['category'].unique()
for category in categories:
    colors = ['red', 'blue', 'green', 'magenta','gold']
    data = []
    for overall in range(1,6,1):
        # find dataframe for specific overall and category:
        temp_df = df.loc[(df['category'] == category) & (df['overall'] == overall)]
        data.append(temp_df.unixReviewTime)



    plt.hist(data, color = colors, label = ['1','2','3','4','5'], bins = [0.9e9, 1e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9, 1.5e9, 1.6e9])
    plt.title(category)
    plt.xlim(0.9e9, 1.6e9)
    plt.ylim(0, 30000)
    plt.xlabel('unixReviewTime ')
    plt.ylabel('Frequency')

    plt.show()
    
#%% Look at statistics for review and summary data:
cv = CountVectorizer(ngram_range = (1,1), lowercase = True, analyzer = 'word', binary = True)
analyzer = cv.build_analyzer()

# count number of words in the summary and reviews
summary_texts = list(df['summary'])
summary_lengths = []
for summary in summary_texts:
    # tokenize:
    summary_tok = analyzer(summary)
    summary_lengths.append(len(summary_tok))

review_texts = list(df['reviewText'])
review_lengths = []
for review in review_texts:
    # tokenize:
    review_tok = analyzer(review)
    review_lengths.append(len(review_tok))

plt.boxplot([summary_lengths], vert = False, labels = ['Summary'])
plt.title('Summary Length')
plt.xlim(0,40)
plt.show()

plt.boxplot([review_lengths], vert = False, labels = ['Review'])
plt.title('Review Length')
plt.xlim(0, 5500)
plt.show()

# count number of unique words in summary and review
cv_fit_summary = cv.fit(summary_texts)
print('number of unique words in summaries:', len(cv_fit_summary.vocabulary_.keys()))

cv_fit_review = cv.fit(review_texts)
print('number of unique words in reviews:', len(cv_fit_review.vocabulary_.keys()))

#%% look at common words in summary and review data:
# get common stop words
custom_stop_words = set(stopwords.words('english'))
# add additional stop words:
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer() 
additional_stop_words = {'pop', 'classical', 'jazz', 'dance', 'electronic', 'rock'}
additional_stop_words_normalized=set()
for additional_word in additional_stop_words:
    category_lem = lemmatizer.lemmatize(additional_word)
    category_stem = ps.stem(additional_word)
    additional_stop_words_normalized.update({category_lem,category_stem})

custom_stop_words.update(additional_stop_words_normalized)
summary_texts = list(df['summary'])


cv_summary = CountVectorizer(ngram_range=(1,1), lowercase = True, stop_words=list(custom_stop_words), analyzer = 'word', binary = True, max_features = 30)
cv_summary_fit = cv_summary.fit_transform(summary_texts)

word_count = np.sum(cv_summary_fit.toarray(), axis = 0)

df_temp = pd.DataFrame({'word':cv_summary.get_feature_names(), 'word_count':word_count})
df_temp.sort_values(by='word_count', ascending=False, inplace = True)
df_temp.plot(x='word', y='word_count', kind = 'bar')
plt.xlabel('Top 30 Frequent Unigrams')
plt.ylabel('Frequency')
plt.legend('')
plt.title('Summary Word Distribution')
plt.ylim(0, 30000)
plt.show()
#%%
review_texts = list(df['reviewText'])
cv_review = CountVectorizer(ngram_range=(1,1), lowercase = True, stop_words=list(custom_stop_words), analyzer = 'word', binary = True, max_features = 30)
cv_review_fit = cv_review.fit_transform(review_texts)

word_count = np.sum(cv_review_fit.toarray(), axis = 0)

df_temp = pd.DataFrame({'word':cv_review.get_feature_names(), 'word_count':word_count})
df_temp.sort_values(by='word_count', ascending=False, inplace = True)
df_temp.plot(x='word', y='word_count', kind = 'bar')
plt.xlabel('Top 30 Frequent Unigrams')
plt.ylabel('Frequency')
plt.legend('')
plt.ylim(0, 70000)
plt.title('Review Word Distribution')
plt.show()














