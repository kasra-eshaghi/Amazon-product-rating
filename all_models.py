# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

def model_1(train_review_time, train_categories, train_summaries, train_reviews, cv_summary_fit, cv_review_fit):
    # model 1 - min val = 0.5619
    input_time = keras.layers.Input(shape = [train_review_time.shape[1]], name='time_input')
    input_cat = keras.layers.Input(shape = [train_categories.shape[1]], name='cat_input')
    input_summary = keras.layers.Input(shape = [train_summaries.shape[1]], name='summary_input')
    input_review = keras.layers.Input(shape = [train_reviews.shape[1]], name='review_input')
    
    embed_summary = keras.layers.Embedding(input_dim = (len(cv_summary_fit.vocabulary_)+1), output_dim = 5, input_length = train_summaries.shape[1], name = 'embed_summary')(input_summary)
    embed_review = keras.layers.Embedding(input_dim = (len(cv_review_fit.vocabulary_)+1), output_dim = 32, input_length = train_reviews.shape[1], name = 'embed_review')(input_review)
    
    flatten_summary = keras.layers.Flatten()(embed_summary)
    flatten_review = keras.layers.Flatten()(embed_review)
    
    concat_all = keras.layers.concatenate([input_time, input_cat, flatten_summary, flatten_review])
    
    hidden_1 = keras.layers.Dense(units = 100, activation = 'relu', name = 'hidden')(concat_all)
    
    output = keras.layers.Dense(units = 1, name = 'output')(hidden_1)
    
    model = keras.Model(inputs=[input_time, input_cat, input_summary, input_review], outputs = [output])
    
    model.compile(loss = 'mse', optimizer = 'Adam') 
    
    return model

def model_2(train_reviews, cv_review_fit):
    #min val = 0.6
    model = keras.models.Sequential([
    keras.layers.Embedding(input_dim = (len(cv_review_fit.vocabulary_)+1), output_dim = 32, input_length = train_reviews.shape[1], name = 'embed_review'),
    keras.layers.LSTM(units = 128),
    keras.layers.Dense(units = 1, name = 'output'),
    ])
    
    model.compile(loss = 'mse', optimizer = 'Adam') 
    
    return model
    
def model_3(train_review_time, train_categories, train_summaries, train_reviews, cv_summary_fit, cv_review_fit):
    # model 3 - min val = 0.58
    input_time = keras.layers.Input(shape = [train_review_time.shape[1]], name='time_input')
    input_cat = keras.layers.Input(shape = [train_categories.shape[1]], name='cat_input')
    input_summary = keras.layers.Input(shape = [train_summaries.shape[1]], name='summary_input')
    input_review = keras.layers.Input(shape = [train_reviews.shape[1]], name='review_input')
    
    embed_cat = keras.layers.Embedding(input_dim = 5, output_dim = 5, input_length = train_categories.shape[1], name = 'embed_cat')(input_cat)
    embed_summary = keras.layers.Embedding(input_dim = (len(cv_summary_fit.vocabulary_)+1), output_dim = 5, input_length = train_summaries.shape[1], name = 'embed_summary')(input_summary)
    embed_review = keras.layers.Embedding(input_dim = (len(cv_review_fit.vocabulary_)+1), output_dim = 32, input_length = train_reviews.shape[1], name = 'embed_review')(input_review)
    
    flatten_cat = keras.layers.Flatten()(embed_cat)
    flatten_summary = keras.layers.Flatten()(embed_summary)
    flatten_review = keras.layers.Flatten()(embed_review)
    
    concat_all = keras.layers.concatenate([input_time, flatten_cat, flatten_summary, flatten_review])
    
    hidden_1 = keras.layers.Dense(units = 100, activation = 'relu', name = 'hidden')(concat_all)
    
    output = keras.layers.Dense(units = 1, name = 'output')(hidden_1)
    
    model = keras.Model(inputs=[input_time, input_cat, input_summary, input_review], outputs = [output])
    
    model.compile(loss = 'mse', optimizer = 'Adam') 
            
    return model

def model_4(train_review_time, train_categories, train_summaries, train_reviews, cv_summary_fit, cv_review_fit):
    # model 4 - min val = 0.86
    input_time = keras.layers.Input(shape = (train_review_time.shape[1],), name='time_input')
    input_cat = keras.layers.Input(shape = (train_categories.shape[1],), name='cat_input')
    
    embed_cat = keras.layers.Embedding(input_dim = 5, output_dim = 5, input_length = train_categories.shape[1], name = 'embed_cat')(input_cat)
    flatten_cat = keras.layers.Flatten(name = 'flatten_cat')(embed_cat)
    concat_time_cat = keras.layers.concatenate([input_time, flatten_cat], name = 'concat_time_cat')
    dense_time_cat = keras.layers.Dense(units = 10, activation = 'relu')(concat_time_cat)
    
    
    input_summary = keras.layers.Input(shape = (train_summaries.shape[1],), name='summary_input')
    embed_summary = keras.layers.Embedding(input_dim = (len(cv_summary_fit.vocabulary_)+1), output_dim = 32, input_length = train_summaries.shape[1], name = 'embed_summary')(input_summary)
    lstm_summary = keras.layers.LSTM(units = 128)(embed_summary)
    dense_summary = keras.layers.Dense(units = 1)(lstm_summary)

    input_review = keras.layers.Input(shape = (train_reviews.shape[1],), name='review_input')
    embed_review = keras.layers.Embedding(input_dim = (len(cv_review_fit.vocabulary_)+1), output_dim = 32, input_length = train_reviews.shape[1], name = 'embed_review')(input_review)
    lstm_review = keras.layers.LSTM(units = 128)(embed_review) 
    dense_review = keras.layers.Dense(units = 1)(lstm_review)   
    
    concat_all = keras.layers.concatenate([dense_time_cat, dense_summary, dense_review])
    
    
    output = keras.layers.Dense(units = 1, name = 'output')(concat_all)
    
    model = keras.Model(inputs=[input_time, input_cat, input_summary, input_review], outputs = [output])
    
    model.compile(loss = 'mse', optimizer = 'Adam') 
            
    return model

def model_5(train_review_time, train_categories, train_summaries, train_reviews, cv_summary_fit, cv_review_fit):
    # model 5 - min val =0.55
    input_time = keras.layers.Input(shape = [train_review_time.shape[1]], name='time_input')
    input_cat = keras.layers.Input(shape = [train_categories.shape[1]], name='cat_input')
    input_summary = keras.layers.Input(shape = [train_summaries.shape[1]], name='summary_input')
    input_review = keras.layers.Input(shape = [train_reviews.shape[1]], name='review_input')
    
    embed_summary = keras.layers.Embedding(input_dim = (len(cv_summary_fit.vocabulary_)+1), output_dim = 16, input_length = train_summaries.shape[1], name = 'embed_summary')(input_summary)
    embed_review = keras.layers.Embedding(input_dim = (len(cv_review_fit.vocabulary_)+1), output_dim = 32, input_length = train_reviews.shape[1], name = 'embed_review')(input_review)
    
    conv1d_summary = keras.layers.Conv1D(filters = 128, kernel_size=5)(embed_summary)
    conv1d_review = keras.layers.Conv1D(filters = 128, kernel_size=5)(embed_review)
    
    flatten_summary = keras.layers.Flatten()(conv1d_summary)
    flatten_review = keras.layers.Flatten()(conv1d_review)
    
    concat_all = keras.layers.concatenate([input_time, input_cat, flatten_summary, flatten_review])
    
    hidden_1 = keras.layers.Dense(units = 100, activation = 'relu', name = 'hidden')(concat_all)
    
    output = keras.layers.Dense(units = 1, name = 'output')(hidden_1)
    
    model = keras.Model(inputs=[input_time, input_cat, input_summary, input_review], outputs = [output])
    
    model.compile(loss = 'mse', optimizer = 'Adam') 
    
    return model

def model_6(train_summaries, train_reviews, vocab_size, max_sent_len):
    # create model, val error = 0.63
    input_reviews = keras.layers.Input(shape = (train_reviews.shape[1],), name = 'review_input')
    input_summaries = keras.layers.Input(shape = (train_summaries.shape[1],), name = 'summary_input')
    
    embed_words = keras.layers.Embedding(input_dim = vocab_size, output_dim = 8, input_length = max_sent_len, mask_zero=True, name = 'word_embedding')
    
    input_reviews_encoded = embed_words(input_reviews)
    input_summaries_encoded = embed_words(input_summaries)
    
    
    concat_embeded_sentences = keras.layers.Concatenate(axis = -1, name = 'summary_and_review')([input_reviews_encoded, input_summaries_encoded])
        
    lstm_sentences = keras.layers.LSTM(64, return_sequences = False, name = 'lstm_out')(concat_embeded_sentences)
    
    output = keras.layers.Dense(1)(lstm_sentences)
    
    model = keras.Model(inputs=[input_reviews, input_summaries], outputs = [output])
    
    model.compile(loss = 'mse', optimizer = 'Adam') 

def model_7(train_reviews, vocab_size, max_sent_len):
    #%% create model, val error = 0.5
    input_reviews = keras.layers.Input(shape = (train_reviews.shape[1],), name = 'review_input')
    
    embed_review = keras.layers.Embedding(input_dim = vocab_size, output_dim = 32, input_length = max_sent_len, mask_zero=False)(input_reviews)
    
    gru_review = keras.layers.GRU(units = 128, return_sequences=False)(embed_review)
    #gru_review_2 = keras.layers.GRU(units = 128)(gru_review)
    
    output = keras.layers.Dense(1)(gru_review)
    
    model = keras.Model(inputs=[input_reviews], outputs = [output])
    
    model.compile(loss = 'mse', optimizer = 'Adam') 

def build_model(n_hidden = 1, n_neurons = 50, learning_rate = 3e-3,  inputshape= 5):
    model = keras.models.Sequential()
    # add input layer
    model.add(keras.layers.InputLayer(input_shape = (inputshape,)))
    
    model.add(keras.layers.Dense(n_neurons, activation = 'relu', input_shape = (inputshape,)))
    # add desired number of hidden layers:
    for layer in range(n_hidden-1):
        model.add(keras.layers.Dense(n_neurons, activation = 'relu'))
        
    # add last dense layer to get output:
    model.add(keras.layers.Dense(1))
    
    # compile model:
    model.compile(loss ='mse', optimizer = keras.optimizers.SGD(lr = learning_rate))
    
    return model
    
