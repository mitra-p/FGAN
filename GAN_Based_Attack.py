# -*- coding: utf-8 -*-
 
"""
Created on Fri Sep 18 12:03:10 2020

@author: Rahim Taheri

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import functools
import glob
import os
import PIL
import time
from scipy import sparse
import pandas as pd
import numpy as np
import random
import argparse
import math
from numpy import *
import os.path as osp
import scipy.sparse as sp
import pickle

#******************************************************************************
CLASS = 'class'
CLASS_BEN = 'B'
CLASS_MAL = 'M'
DATA = 'data'
#********************************************Functions ************************
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-tables', nargs='*', dest='input_tables')

    args = parser.parse_args()

    return args
#******************************************************************************
def read_table(table_file):
    
        table = dict()
        
        with open(table_file, 'rb') as handle:
            while True:
                   try:
                           table = pickle.load(handle)
                   except EOFError:
                           break
        
        f_set=set()
        
        for k,v in table.items():
             for feature in v[DATA]:
                f_set.add(feature)
               
        return table , f_set
#******************************************************************************
def build_table(tables):
    full_table = dict()

    file_set = set()
    
    for table in tables:
        file_set.update(table.keys())
        for key, val in table.items():
            full_table[key] = val
              
    files = list(file_set)
    return full_table, files
#******************************************************************************
def convert_to_matrix(table, features, files):
    mat = sp.lil.lil_matrix((len(files), len(features)), dtype=np.int8)

    print("Input Data Size =  ", mat.get_shape())
    # the response vector
   
    cl = [0]*len(files)
    
    for key, val in table.items():
        k = files.index(key)
    
        if val[CLASS] is CLASS_BEN:
            cl[k] = 1
       
        for v in val[DATA]:
            try:
                idx = features.index(v)
                mat[k, idx] = 1
            except Exception as e:
                print(e)
                pass              
        
    return mat, cl
#******************************************************************************
def delete_row_lil(mat, i):
    if not isinstance(mat, sp.lil.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])
#******************************************************************************
def relevant_features(data, response_vector, features):
    rel_features = list()
    ranked_index=list()
    
    model =RandomForestRegressor()
    rfe = RFE(model, 1)
    fit = rfe.fit(data, response_vector)
    old_features=features

    for i in fit.ranking_:
        if i<len(features):
              rel_features.append(features[i])
    ranked_index=[old_features.index(x) for x in rel_features if x in old_features]
       
    return rel_features ,ranked_index
#*************************************Main Function****************************
def main():
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()

    tables = []
    f_set = set()
    
    #*************************************read the data************************
    for t_files in args.input_tables:
        table, features = read_table(t_files)
        f_set = f_set.union(features)
        tables.append(table)
    print("                                                                    ")
    print("                                                                    ")
    print("********************************************************************")
    print("**************************GAN Based Attack**************************") 
    print("********************************************************************")

    #******************build table from data and convert to matrix************* 
    full_table, files = build_table(tables)
    files.sort()
    features = list(f_set)
    features.sort()
    mat, cl = convert_to_matrix(full_table, features, files) 
   
    #**************Doing feature Ranking on all of the Data********************
    print("****************Doing feature Ranking on all of the Data************")
    r_features,ranked_index = relevant_features(mat, cl, features)
    print("********************************************************************")
 
    original_selected=ranked_index[1:257]
    data = sparse.lil_matrix(sparse.csr_matrix(mat)[:,original_selected])
    seed = 10
    test_size = 0.3
    X_train, X_test, Y_train, Y_test= train_test_split(data, cl, test_size= test_size, random_state=seed)
    #***************Datapreparation***************
    X_train=X_train.todense()
    X_test=X_test.todense()
    
    X_train=np.array(X_train)
    #row_X_train=X_train.shape[0]
    #col_X_train=X_train.shape[1]
    
    X_test=np.array(X_test)
    #row_X_test=X_test.shape[0]
    #col_X_test=X_test.shape[1]
    
    Y_train=np.asanyarray(Y_train)
    #row_Y_train=Y_train.shape[0]
    
    Y_test=np.asanyarray(Y_test)
    #row_Y_test=Y_test.shape[0]
    
    X_train = X_train.reshape(X_train.shape[0],16,16,1)
    X_test = X_test.reshape(X_test.shape[0],16,16,1)
    #**************************************************************************  
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    EPOCHS = 150
    noise_dim = 256
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    #**************************************************************************
    state = np.random.get_state()
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(Y_train)
    #**************************************************************************   
    def make_generator_model():
        model = keras.Sequential()
        
        model.add(keras.layers.Dense(4*4*256, use_bias=False, input_shape=(256,)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU())
    
        model.add(keras.layers.Reshape((4, 4, 256)))
        assert model.output_shape == (None, 4, 4, 256)  # Batch size is not limited
    
        model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 4, 4, 128)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU())
    
        model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 64)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU())
    
        model.add(keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 16, 16, 1)
    
        return model
    #**************************************************************************
    generator = make_generator_model()
    #**************************************************************************

    def make_discriminator_model():
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[16, 16, 1]))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.3))
    
        model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.3))
    
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(2))
        return model
    #**************************************************************************
    malicious_discriminator = make_discriminator_model()
    malicious_discriminator.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    malicious_discriminator.fit(X_train, Y_train, epochs=20, batch_size=256)   
    test_loss, test_acc = malicious_discriminator.evaluate(X_test, Y_test, verbose=0)
    print('\nTest accuracy:', test_acc, 'Test loss:', test_loss)   
    cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #**************************************************************************
    def discriminator_loss(real_output, fake_output, real_labels):
        real_loss = cross_entropy(real_labels, real_output)
        
        fake_result = np.zeros(len(fake_output))
        # Attack label
        for i in range(len(fake_result)):
            fake_result[i] = 1
        fake_loss = cross_entropy(fake_result, fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    #**************************************************************************
    def generator_loss(fake_output):
        ideal_result = np.zeros(len(fake_output))
        # Attack label
        for i in range(len(ideal_result)):
            ideal_result[i] = 0
        
        return cross_entropy(ideal_result, fake_output)
    #**************************************************************************
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    #**************************************************************************
    @tf.function
    def train_step(data_sample_batch, data_labels):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_sample = generator(noise, training=True)
            print(generated_sample.shape)
    
            # real_output is the probability of the mimic number
            real_output = malicious_discriminator(data_sample_batch, training=False)
            fake_output = malicious_discriminator(generated_sample, training=False)
            
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output, real_labels = data_labels)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, malicious_discriminator.trainable_variables)
    
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, malicious_discriminator.trainable_variables))
    
    #**************************************************************************
    def train(dataset, labels, epochs):
        for epoch in range(epochs):
            start = time.time()
            for i in range(round(len(dataset)/BATCH_SIZE)):
                sample_batch = dataset[i*BATCH_SIZE:min(len(dataset), (i+1)*BATCH_SIZE)]
                labels_batch = labels[i*BATCH_SIZE:min(len(dataset), (i+1)*BATCH_SIZE)]
                train_step(sample_batch, labels_batch)
    
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
        
    #**************************************************************************   
 
    attack_samples = X_train[Y_train==0]
    attack_labels = Y_train[Y_train==0]
    
    train(attack_samples, attack_labels, EPOCHS)


    #**************************************************************************
    print("********************************************************************")
#******************************************************************************
if __name__ == "__main__":
    main()
#******************************************************************************
