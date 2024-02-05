import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow import keras


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


############## Preprocessing #################
# read data
all_data = pd.read_json('train.json', lines=True)
all_data = clean_data(all_data)

# split training data
train_data, val_data = train_test_split(all_data, test_size = 0.3, random_state = 0)
y_train = np.array(train_data['overall'])
y_val = np.array(val_data['overall'])

# create tokenizer for text data
n_vocab = 5000
tokenizer = create_text_processor(train_data, n_vocab)
vocab_size = len(tokenizer.word_index)+1

# turn text data into sequences
max_sent_len = 100
train_summaries, train_reviews = use_text_process(train_data, tokenizer, max_sent_len)
val_summaries, val_reviews = use_text_process(val_data, tokenizer, max_sent_len)

train_texts = np.concatenate((train_reviews, train_summaries), axis = 1)
val_texts = np.concatenate((val_reviews, val_summaries), axis = 1)

# create one-hot encoder for categories, and scalar transform for review time. NOTE: cv_summary_fit, cv_review_fit are not used.
scalar_fit, cat_encoder_fit, cv_summary_fit, cv_review_fit= create_data_processors(train_data, 200, 200)
train_review_time, train_categories, _, _ = create_inputs(train_data, scalar_fit, cat_encoder_fit, cv_summary_fit, cv_review_fit, False, True, True)
val_review_time, val_categories, _, _ = create_inputs(val_data, scalar_fit, cat_encoder_fit, cv_summary_fit, cv_review_fit, False, True, True)

################# Fit model ##################
# hyperparameters
e_size = 16
l_size = 64
d_size = 200

            
# add early stopping criterion
model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    mode = 'min',
    )
call_backs = [model_early_stopping_callback]

# inputs:
input_text = keras.layers.Input(shape = (train_texts.shape[1],), name = 'text_input')
input_time = keras.layers.Input(shape = (train_review_time.shape[1],), name = 'review_time')
input_cat = keras.layers.Input(shape = (train_categories.shape[1],), name = 'categories')

# text embedding and LSTM:
embed_words = keras.layers.Embedding(input_dim = vocab_size, output_dim = e_size, input_length = max_sent_len*2, mask_zero=True, name = 'word_embedding')(input_text)
LSTM_review = keras.layers.LSTM(units = l_size, return_sequences=False)(embed_words)

# concatenate all processed inputs
concat_all = keras.layers.concatenate([input_time, input_cat, LSTM_review], name = 'concat_all')

# dense layer # 1
dense_layer = keras.layers.Dense(units = d_size, activation = 'relu')(concat_all)

# dense layer # 2
output = keras.layers.Dense(1)(dense_layer)

model = keras.Model(inputs=[input_text, input_time, input_cat], outputs = [output])
model.compile(loss = 'mse', optimizer = 'Adam') 

# fit model
history = model.fit((train_texts, train_review_time, train_categories), y_train, batch_size=32, epochs = 100, validation_data = (((val_texts, val_review_time, val_categories)),y_val), callbacks = call_backs)

########## THANK YOU #########