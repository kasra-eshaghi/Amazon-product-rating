# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from nltk.corpus import stopwords
from tensorflow import keras

def create_text_processor(train_data, n_vocab):
    ''' 
        This function creates the tokenizer for processing text data
        OUTPUT:
            tokenizer - tokenizer instance that can be used to create text sequences
    '''
    
    summary_train = list(train_data['summary'])
    review_train = list(train_data['reviewText'])
    
    text_all = summary_train+ review_train
    
    tokenizer = keras.preprocessing.text.Tokenizer(num_words = n_vocab)
    tokenizer.fit_on_texts(text_all)
    
    return tokenizer

def use_text_process(data, tokenizer, max_sent_len):
    '''
        This function turns the summary and reviewText data into sequences that are padded
        tokenizer must be provided
        OUTPUTS:
            train_summaries
            train_reviews
    '''
    summary = list(data['summary'])
    review = list(data['reviewText'])
    
    # turn into sequence
    summary_numbers = tokenizer.texts_to_sequences(summary)
    review_numbers = tokenizer.texts_to_sequences(review)
    
    # pad:
    summary_numbers = keras.preprocessing.sequence.pad_sequences(summary_numbers, padding = 'post', maxlen = max_sent_len)
    review_numbers = keras.preprocessing.sequence.pad_sequences(review_numbers, padding = 'post', maxlen = max_sent_len)
    
    return summary_numbers, review_numbers
    

def create_data_processors(train_data, n_bigrams_summary, n_unigram_review):
    '''
        This function create the preprocessors necessary for enconding the time, categorical, and text data
        INTPUT:
            train_data - all training data (in pd format)
        OUTPUT:
            scalar_fit, cat_encoder_fit, cv_summary_fit, cv_review_fit - preprocessing objects to be used
            
    '''
    pass

    scalar = MinMaxScaler(feature_range = (0,1), copy = False)
    scalar_fit = scalar.fit(pd.DataFrame(train_data['unixReviewTime']))
    
    cat_encoder = OneHotEncoder()
    cat_encoder_fit = cat_encoder.fit(pd.DataFrame(train_data['category']))
    
    custom_stop_words = set(stopwords.words('english'))
    
    summary_train = list(train_data['summary'])
    cv_summary = CountVectorizer(ngram_range = (2,2), lowercase = True, stop_words=list(custom_stop_words), analyzer = 'word', binary = True, max_features = n_bigrams_summary)
    cv_summary_fit = cv_summary.fit(summary_train)
    
    review_train = list(train_data['reviewText'])
    cv_review = CountVectorizer(ngram_range = (1,1), lowercase = True, stop_words=list(custom_stop_words), analyzer = 'word', binary = True, max_features = n_unigram_review)
    cv_review_fit = cv_review.fit(review_train)

    return scalar_fit, cat_encoder_fit, cv_summary_fit, cv_review_fit
        
def clean_data(df):
    '''
        This function cleans up the dataset
    '''
    # drop unnecessary columns
    df.drop('image', inplace = True, axis = 1)
    df.drop('reviewTime', inplace = True, axis = 1)
    df.drop('reviewHash', inplace = True, axis = 1)
    df.drop('price', inplace = True, axis = 1)
    
    # clean up review text and summary text:
    df['reviewText'] = df['reviewText'].replace(np.nan,'', regex = True)
    df['summary'] = df['summary'].replace(np.nan,'', regex = True)

    return df

def embed_data(corp, cv, emb_sentence_length):
    '''
        This function prepares the text data for the embedding layer:
        INPUTS:
            corp: list of strings, each of which is a datapoint
            cv - count vectorizer instance used
            emb_sentence_length - number of words to be considered in each sentence
        OUTPUTS:
            corp_for_embedding
    '''
    analyzer = cv.build_analyzer()
    corp_emb = []
    
    # embed the sentence based on the words in dictionary
    for sent in corp:
        # tokenize sentence
        sent_tokenized = analyzer(sent)
        sent_emb = []
        for i, word in enumerate(sent_tokenized):
            #onlt keep first few words in the sentence
            if i > emb_sentence_length:
                break
            # check if word is in the vocabulary
            if word in cv.vocabulary_:
                sent_emb.append(cv.vocabulary_[word]+1) # add one to avoid 0, as the vocabulary includes index 0 as well
                
        # add sentences to embedded corpus
        corp_emb.append(sent_emb)
            
    # padd corpus to make them all the same length
    corp_pad = keras.preprocessing.sequence.pad_sequences(corp_emb, emb_sentence_length, padding = 'post')
    
    return corp_pad

def embed_categories(categories_1hot):
    '''
        This function embeds the categories
    '''
    categories_embedded = np.where(categories_1hot == 1)[1]
    categories_embedded = categories_embedded.reshape((-1,1))
    
    return categories_embedded

def create_inputs(data, scalar_fit, cat_encoder_fit, cv_summary_fit, cv_review_fit, embed_cat, use_embed_summary, use_embed_review):
    '''
        This function creates the training data based on the preprocessing objects scalar_fit, cat_encoder_fit, cv_summary_fit, cv_review_fit
        use_embed_summary, use_embed_review are used to indicate how text data should be processed
    '''
    # create time data
    review_time_normalized = scalar_fit.transform(pd.DataFrame(data['unixReviewTime']))
    
    # create category data
    categories_1hot = cat_encoder_fit.transform(pd.DataFrame(data['category'])).toarray()
    if embed_cat:
       categories_1hot = embed_categories(categories_1hot)
    
    # create summary data
    summary = list(data['summary'])
    if use_embed_summary:
        emb_sentence_length = 15
        summaries_encoded = embed_data(summary, cv_summary_fit, emb_sentence_length)
    else:
        summaries_encoded = cv_summary_fit.transform(summary)

    # create review data
    review = list(data['reviewText'])
    if use_embed_review:
        emb_sentence_length = 300
        reviews_encoded = embed_data(review, cv_review_fit, emb_sentence_length)
    else:
        reviews_encoded = cv_review_fit.transform(review)
        
    return review_time_normalized, categories_1hot, summaries_encoded, reviews_encoded


def cap_results(y):
    '''
        This function caps the results, namely, ones over 5 are set to 5, and ones below 1 are set to 1
    '''
    y[np.where(y>5)] = 5
    y[np.where(y<1)] = 1

    return y