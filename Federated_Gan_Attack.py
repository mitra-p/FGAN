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
#*********************Functions that will be used in this program**************
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
#******************************Main Function***********************************
def main():
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()

    tables = []
    f_set = set()
    
    #read the data
    for t_files in args.input_tables:
        table, features = read_table(t_files)
        f_set = f_set.union(features)
        tables.append(table)
    print("                                                                    ")
    print("                                                                    ")
    print("********************************************************************")
    print("********************************Federated GAN Based Attack**********") 
    print("********************************************************************")

    #*build table from data and convert to matrix 
    full_table, files = build_table(tables)
    files.sort()
    features = list(f_set)
    features.sort()
    mat, cl = convert_to_matrix(full_table, features, files) 
   
    #Doing feature Ranking on all of the Data
    print("************************Doing feature Ranking on all of the Data****")
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
    #Y_train=Y_train.reshape(Y_train.shape[0],16,16,1)
    X_test = X_test.reshape(X_test.shape[0],16,16,1)
    #Y_test=Y_test.reshape(Y_test.shape[0],16,16,1)
    #************************************************************************** 
    Round = 5
    Clinets_per_round = 10
    Batch_size = 16
    Gan_epoch = 20
    Test_accuracy = []
    Models = { }
    Client_data = {}
    Client_labels = {}

    BATCH_SIZE = 256
    noise_dim = 256
    num_examples_to_generate = 36
    num_to_merge = 500
    # num_to_merge = 50
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    seed_merge = tf.random.normal([num_to_merge, noise_dim])

    #**************************************************************************
    #******************************Load Data***********************************
    #**************************************************************************
    
    # Data
    state = np.random.get_state()
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(Y_train)

    for i in range(Clinets_per_round):  
        Client_data.update({i:X_train[Y_train==i]})
        Client_labels.update({i:Y_train[Y_train==i]})

        state = np.random.get_state()
        np.random.shuffle(Client_data[i])
        np.random.set_state(state)
        np.random.shuffle(Client_labels[i])

    attack_ds = np.array(Client_data[0])
    attack_l = np.array(Client_labels[0])


    #**************************************************************************
    #******************************Models Prepared*****************************
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

    def make_generator_model():
        model = keras.Sequential()
        
        model.add(keras.layers.Dense(4*4*256, use_bias=False, input_shape=(256,)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())
    
        model.add(keras.layers.Reshape((4, 4, 256)))
        assert model.output_shape == (None, 4, 4, 256)  # Batch size is not limited
    
        model.add(keras.layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 4, 4, 128)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())
    
        model.add(keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 64)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())
    
        model.add(keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 16, 16, 1)
    
        return model

    model = make_discriminator_model()

    for i in range(Clinets_per_round):
        Models.update({i:make_discriminator_model()})
        Models[i].compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    #**************************************************************************
    #*****************************Attack setup*********************************
    #**************************************************************************

    generator = make_generator_model()
    malicious_discriminator = make_discriminator_model()

    cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output, real_labels):
        real_loss = cross_entropy(real_labels, real_output)
        
        fake_result = np.zeros(len(fake_output))
        for i in range(len(fake_result)):
            fake_result[i] = 0
        fake_loss = cross_entropy(fake_result, fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        ideal_result = np.zeros(len(fake_output))
        for i in range(len(ideal_result)):
            ideal_result[i] = 1
        
    generator_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, decay=1e-7)
    discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, decay=1e-7)

    @tf.function
    def train_step(data_sample_batch, data_labels):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_samples = generator(noise, training=True)
            
            real_output = malicious_discriminator(data_sample_batch, training=False)
            fake_output = malicious_discriminator(generated_samples, training=False)
            
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output, real_labels =data_labels)
    
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, malicious_discriminator.trainable_variables)
    
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, malicious_discriminator.trainable_variables))

    def train(dataset, labels, epochs):
        for epoch in range(epochs):
            start = time.time()
            for i in range(round(len(dataset)/BATCH_SIZE)):
                sample_batch = dataset[i*BATCH_SIZE:min(len(dataset), (i+1)*BATCH_SIZE)]
                labels_batch = labels[i*BATCH_SIZE:min(len(dataset), (i+1)*BATCH_SIZE)]
                train_step(sample_batch, labels_batch)
    
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
       


    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.fit(X_train, Y_train, validation_split=0, epochs=25, batch_size = 256)
    del X_train, Y_train
    
    tmp_weight = model.get_weights()
    
    attack_count = 0
    
    # Federated learning

    for r in range(Round):
        print('round:'+str(r+1))
        model_weights_sum = []


        for i in range(Clinets_per_round):  
            Models[i].set_weights(tmp_weight)
            
            train_ds = Client_data[i]
            train_l = Client_labels[i]

            if r != 0 and i == 0 and Test_accuracy[i-1] > 0.85:
                print("Attack round: {}".format(attack_count+1))
    
                malicious_discriminator.set_weights(Models[i].get_weights())

                train(attack_ds, attack_l, Gan_epoch)
                
    
                predictions = generator(seed_merge, training=False)
                malicious_samples = np.array(predictions)
                malicious_labels = np.array([1]*len(malicious_samples))
    

                if attack_count == 0:
                    Client_data[i] = np.vstack((Client_data[i], malicious_samples))
                    Client_labels[i] = np.append(Client_labels[i], malicious_labels)  
                else:
                    Client_data[i][len(Client_data[i])-len(malicious_samples):len(Client_data[i])] = malicious_samples
    
                attack_count += 1 
            
            if len(train_ds)!=0:
                Models[i].fit(train_ds, train_l, validation_split=0, epochs=1, batch_size = Batch_size)     
            
            if i == 0:
                model_weights_sum = np.array(Models[i].get_weights())
            else:
                model_weights_sum += np.array(Models[i].get_weights())
    
        mean_weight = np.true_divide(model_weights_sum,Clinets_per_round)
        tmp_weight = mean_weight.tolist()
        del model_weights_sum

        model.set_weights(tmp_weight)
        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
        Test_accuracy.append(test_acc)
        print('\nTest accuracy:', test_acc, 'Test loss:', test_loss)

    #**************************************************************************
    #**************************************************************************
    #**************************************************************************
    print("********************************************************************")
#******************************************************************************
if __name__ == "__main__":
    main()
#******************************************************************************
