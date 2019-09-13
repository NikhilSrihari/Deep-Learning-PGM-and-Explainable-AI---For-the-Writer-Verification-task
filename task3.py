import sys
import numpy as np
import csv
import pandas as pd
from cv2 import imread, imwrite, imshow, destroyAllWindows, waitKey
import time
import shutil
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, UpSampling2D
from keras.models import Model, model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


fileloc_features = "./15FeatureDataset/15features.csv"
fileloc_seen_training = "./15FeatureDataset/seen-dataset/seen-dataset/dataset_seen_training_siamese.csv"
fileloc_seen_validation = "./15FeatureDataset/seen-dataset/seen-dataset/dataset_seen_validation_siamese.csv"
fileloc_shuffled_training = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/dataset_seen_training_siamese.csv"
fileloc_shuffled_validation = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/dataset_seen_validation_siamese.csv"
fileloc_unseen_training = "./15FeatureDataset/unseen-dataset/unseen-dataset/dataset_seen_training_siamese.csv"
fileloc_unseen_validation = "./15FeatureDataset/unseen-dataset/unseen-dataset/dataset_seen_validation_siamese.csv"
featuresDS = None

def excelDataRead(filePath, featuresFileFlag = False):
    if (featuresFileFlag == False):
        dataMatrix = [] 
        with open(filePath, 'rU') as f:
            reader = csv.reader(f)
            for ind1,row in enumerate(reader):
                if (ind1!=0):
                    dataRow = []
                    for ind2,column in enumerate(row):
                        if (ind2!=0):
                            if (ind2==3):
                                dataRow.append(float(column))
                            else:
                                dataRow.append(column)
                    dataMatrix.append(dataRow)   
    else:
        dataMatrix = pd.read_csv(filePath)
    return dataMatrix


def siamese_net_getDataForTesting(dataList, img_baseloc):
    x = [[],[]]
    y = []
    for data in dataList:
        left_img, right_img = None, None
        left_img_loc = img_baseloc+(data[0])
        right_img_loc = img_baseloc+(data[1])
        left_img = np.array(imread(left_img_loc,0))
        right_img = np.array(imread(right_img_loc,0))
        try:
            left_img=(255.0-left_img)/255.0
            right_img=(255.0-right_img)/255.0
            left_img=np.expand_dims(left_img, axis=2)
            right_img=np.expand_dims(right_img, axis=2)
            (x[0]).append(left_img)
            (x[1]).append(right_img)
            y.append(data[2])
        except:
            print("    Exception while creating testing data.")
            continue
    return x,y
    
    
def siamese_net_predictAndCalculateAccuracy(model, X, Y, threshold=0.5):
    Y_predicted_prob = model.predict(X)
    Y_predicted = []
    for y_predicted_prob in Y_predicted_prob:
        if (y_predicted_prob>threshold):
            Y_predicted.append(1.0)
        else:
            Y_predicted.append(0.0)
    precision = precision_score(Y, Y_predicted)
    recall = recall_score(Y, Y_predicted)
    accuracy = accuracy_score(Y, Y_predicted) * 100
    return precision, recall, accuracy
    

def siamese_net_initialize_bias(shape, name=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def siamese_net_initialize_weights(shape, name=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)


"""
    Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
"""
def siamese_net_getmodel(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    model = Sequential()
    model.add(Conv2D(64, (5,5), activation='relu', input_shape=input_shape, kernel_initializer=siamese_net_initialize_weights))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer=siamese_net_initialize_weights, bias_initializer=siamese_net_initialize_bias))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer=siamese_net_initialize_weights, bias_initializer=siamese_net_initialize_bias))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3,3), activation='relu', kernel_initializer=siamese_net_initialize_weights, bias_initializer=siamese_net_initialize_bias))
    model.add(Flatten())
    #model.add(Dense(1024, activation='sigmoid', kernel_initializer=siamese_net_initialize_weights, bias_initializer=siamese_net_initialize_bias, kernel_regularizer=l2(1e-3)))
    model.add(Dense(1024, activation='sigmoid', kernel_initializer=siamese_net_initialize_weights, bias_initializer=siamese_net_initialize_bias, name='encoded_feature_vec'))
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1,activation='sigmoid',bias_initializer=siamese_net_initialize_bias)(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    optimizer = Adam(lr = 0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    #siamese_net.summary()
    return siamese_net


def siamese_net_batchGen(batch_size, excel_data, img_baseloc):
    excel_data_same = []
    excel_data_diff = []
    for excel_data0 in excel_data:
        if (excel_data0[2]==1.0):
            excel_data_same.append(excel_data0)
        else:
            excel_data_diff.append(excel_data0)
    while True:
        counter = 0
        samewriter_indices = np.random.randint(0,len(excel_data_same),int(batch_size/2))
        diffwriter_indices = np.random.randint(0,len(excel_data_diff),int(batch_size/2))
        x,y = [[],[]],[]
        for samewriter_index in samewriter_indices:
            left_img_loc = img_baseloc+(excel_data_same[samewriter_index][0])
            right_img_loc = img_baseloc+(excel_data_same[samewriter_index][1])
            left_img, right_img = None, None
            while(True):
                left_img = np.array(imread(left_img_loc,0))
                right_img = np.array(imread(right_img_loc,0))
                try:
                    #left_img=np.roll(axis=0,a=left_img,shift=(np.random.randint(-64,64)))
                    left_img=(255.0-left_img)/255.0
                    right_img=(255.0-right_img)/255.0
                    left_img=np.expand_dims(left_img, axis=2)
                    right_img=np.expand_dims(right_img, axis=2)
                    break
                except:
                    ind = np.random.randint(0,len(excel_data_same))
                    left_img_loc = img_baseloc+(excel_data_same[ind][0])
                    right_img_loc = img_baseloc+(excel_data_same[ind][1])
            (x[0]).append(left_img)
            (x[1]).append(right_img)
            y.append(1.0)
            counter+=1
        for diffwriter_index in diffwriter_indices:
            left_img_loc = img_baseloc+(excel_data_diff[diffwriter_index][0])
            right_img_loc = img_baseloc+(excel_data_diff[diffwriter_index][1])
            left_img, right_img = None, None
            while(True):
                left_img = np.array(imread(left_img_loc,0))
                right_img = np.array(imread(right_img_loc,0))
                try:
                    #left_img=np.roll(axis=0,a=left_img,shift=(np.random.randint(-64,64)))
                    left_img=(255.0-left_img)/255.0
                    right_img=(255.0-right_img)/255.0
                    left_img=np.expand_dims(left_img, axis=2)
                    right_img=np.expand_dims(right_img, axis=2)
                    break
                except:
                    ind = np.random.randint(0,len(excel_data_diff))
                    left_img_loc = img_baseloc+(excel_data_diff[ind][0])
                    right_img_loc = img_baseloc+(excel_data_diff[ind][1])
            (x[0]).append(left_img)
            (x[1]).append(right_img)
            y.append(0.0)
            counter+=1
        if counter == batch_size:
            yield x,y


def siamese_net_train(model, seen_training, seen_validation, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, modelfilename_prefix):
    whole_train_x, whole_train_y = siamese_net_getDataForTesting(seen_training, training_img_baseloc)
    whole_val_x, whole_val_y = siamese_net_getDataForTesting(seen_validation, validation_img_baseloc)
    train_time_list = []
    val_loss_list = []
    val_acc_list = []
    val_precision_list = []
    val_recall_list = []
    train_loss_list = []
    train_acc_list = []
    train_precision_list = []
    train_recall_list = []
    best_val_acc = None
    best_val_loss = None
    best_val_loss_iter = None
    t_start = time.time()
    for iterNum,batchData in enumerate(siamese_net_batchGen(batch_size, seen_training, training_img_baseloc)):
        if (iterNum==num_of_iters+1):
            break
        train_loss = model.train_on_batch(batchData[0], batchData[1])
        if iterNum % evaluate_for_every == 0:
            train_precision, train_recall, train_acc = siamese_net_predictAndCalculateAccuracy(model, whole_train_x, whole_train_y)
            val_loss = model.test_on_batch(whole_val_x, whole_val_y)
            val_precision, val_recall, val_acc = siamese_net_predictAndCalculateAccuracy(model, whole_val_x, whole_val_y)
            train_time = (time.time()-t_start)/60.0
            #Saving the model
            with open(modelfilename_prefix+"_model_arch_"+str(iterNum)+".json", "w") as json_file:
                json_file.write(model.to_json())
            model.save_weights(modelfilename_prefix+"_model_weights_"+str(iterNum)+".h5")
            train_time_list.append(train_time)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            val_precision_list.append(val_precision)
            val_recall_list.append(val_recall)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            train_precision_list.append(train_precision)
            train_recall_list.append(train_recall)
            print("------------------------------------------------")
            print("Time for {0} iterations: {1} mins".format(iterNum, train_time))
            print("Train Loss: {0}".format(train_loss))
            print("Train Acc in %: {0}".format(train_acc))
            print("Val Loss: {0}".format(val_loss))
            print("Val Acc in %: {0}".format(val_acc))
            if (best_val_loss == None or val_loss < best_val_loss):
                print("Current validation loss best: {0}, previous validation loss best: {1}".format(val_loss, best_val_loss))
                print("Current validation accuracy best: {0}, previous validation accuracy best: {1}".format(val_acc, best_val_acc))
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_loss_iter = iterNum
            print("------------------------------------------------")
    shutil.copyfile(modelfilename_prefix+"_model_arch_"+str(best_val_loss_iter)+".json", modelfilename_prefix+"_model_arch_final.json")
    shutil.copyfile(modelfilename_prefix+"_model_weights_"+str(best_val_loss_iter)+".h5", modelfilename_prefix+"_model_weights_final.h5")
    json_file2 = open(modelfilename_prefix+"_model_arch_final.json", 'r')
    loaded_model_json = json_file2.read()
    json_file2.close()
    final_model = model_from_json(loaded_model_json)
    final_model.load_weights(modelfilename_prefix+"_model_weights_final.h5")
    optimizer = Adam(lr = 0.00006)
    final_model.compile(loss="binary_crossentropy", optimizer=optimizer)
    thresholds_list = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    thresholds_val_acc_list = []
    thresholds_val_precision_list = []
    thresholds_val_recall_list = []
    for threshold in thresholds_list:
        val_precision, val_recall, val_accuracy = siamese_net_predictAndCalculateAccuracy(final_model, whole_val_x, whole_val_y, threshold)
        thresholds_val_acc_list.append(val_accuracy)
        thresholds_val_precision_list.append(val_precision)
        thresholds_val_recall_list.append(val_recall)
    best_val_acc = max(thresholds_val_acc_list)
    best_val_acc_index = thresholds_val_acc_list.index(best_val_acc)
    best_val_precision = thresholds_val_precision_list[best_val_acc_index]
    best_val_recall = thresholds_val_recall_list[best_val_acc_index]
    best_threshold = thresholds_list[best_val_acc_index]
    best_train_precision, best_train_recall, best_train_acc = siamese_net_predictAndCalculateAccuracy(final_model, whole_train_x, whole_train_y, best_threshold)     
    return { "train_loss_list": train_loss_list, "train_acc_list": train_acc_list, "train_precision_list":train_precision_list, "train_recall_list":train_recall_list, 
    "val_loss_list": val_loss_list, "val_acc_list":val_acc_list, "val_precision_list": val_precision_list, "val_recall_list": val_recall_list, 
    "best_train_acc": best_train_acc, "best_train_precision": best_train_precision, "best_train_recall": best_train_recall,  
    "best_val_loss": best_val_loss, "best_val_acc": best_val_acc, "best_val_precision": best_val_precision, "best_val_recall": best_val_recall, 
    "best_val_loss_iter": best_val_loss_iter, "train_time_list": train_time_list }
    
    
def siamese_net_main():
    print("SIAMESE NETWORKS:")
    #Seen Data Set
    print("Seen Data Set:")
    batch_size = 32
    input_shape = (64,64,1)
    num_of_iters = 500
    evaluate_for_every = 25
    iterationList = np.arange(0, num_of_iters+1, evaluate_for_every)
    training_img_baseloc = "./15FeatureDataset/seen-dataset/seen-dataset/TrainingSet/"
    validation_img_baseloc = "./15FeatureDataset/seen-dataset/seen-dataset/ValidationSet/"
    trainingDS = excelDataRead(fileloc_seen_training)
    validationDS = excelDataRead(fileloc_seen_validation)
    model = siamese_net_getmodel(input_shape)
    train_history = siamese_net_train(model, trainingDS, validationDS, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, "DL Models/SN_seen_dataset")
    print()
    print("------------------------------------------------")
    print("Model Training Complete. Model Saved.")
    print("Best validation loss: {0}".format(train_history["best_val_loss"]))
    print("Best validation accuracy: {0}".format(train_history["best_val_acc"]))
    print("Best validation precision: {0}".format(train_history["best_val_precision"]))
    print("Best validation recall: {0}".format(train_history["best_val_recall"]))
    print("Best validation loss iter number: {0}".format(train_history["best_val_loss_iter"]))
    print("Best training accuracy: {0}".format(train_history["best_train_acc"]))
    print("Best training precision: {0}".format(train_history["best_train_precision"]))
    print("Best training recall: {0}".format(train_history["best_train_recall"]))
    print("------------------------------------------------")
    print()
    plt.plot(iterationList, train_history["train_time_list"])
    plt.savefig("Siamese Networks - Seen Data Set - Training Time.png")
    plt.plot(iterationList, train_history["train_loss_list"])
    plt.savefig("Siamese Networks - Seen Data Set - Training Loss.png")
    plt.plot(iterationList, train_history["train_acc_list"])
    plt.savefig("Siamese Networks - Seen Data Set - Training Accuracy.png")
    plt.plot(iterationList, train_history["train_precision_list"])
    plt.savefig("Siamese Networks - Seen Data Set - Training Precision.png")
    plt.plot(iterationList, train_history["train_recall_list"])
    plt.savefig("Siamese Networks - Seen Data Set - Training Recall.png")
    plt.plot(iterationList, train_history["val_loss_list"])
    plt.savefig("Siamese Networks - Seen Data Set - Validation Loss.png")
    plt.plot(iterationList, train_history["val_acc_list"])
    plt.savefig("Siamese Networks - Seen Data Set - Validation Accuracy.png")
    plt.plot(iterationList, train_history["val_precision_list"])
    plt.savefig("Siamese Networks - Seen Data Set - Validation Precision.png")
    plt.plot(iterationList, train_history["val_recall_list"])
    plt.savefig("Siamese Networks - Seen Data Set - Validation Recall.png")
    print()
    print()
    #Shuffled Data Set
    print("Shuffled Data Set:")
    batch_size = 32
    input_shape = (64,64,1)
    num_of_iters = 500
    evaluate_for_every = 25
    iterationList = np.arange(0, num_of_iters+1, evaluate_for_every)
    training_img_baseloc = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/TrainingSet/"
    validation_img_baseloc = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/ValidationSet/"
    trainingDS = excelDataRead(fileloc_shuffled_training)
    validationDS = excelDataRead(fileloc_shuffled_validation)
    model = siamese_net_getmodel(input_shape)
    train_history = siamese_net_train(model, trainingDS, validationDS, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, "DL Models/SN_shuffled_dataset")
    print()
    print("------------------------------------------------")
    print("Model Training Complete. Model Saved.")
    print("Best validation loss: {0}".format(train_history["best_val_loss"]))
    print("Best validation accuracy: {0}".format(train_history["best_val_acc"]))
    print("Best validation precision: {0}".format(train_history["best_val_precision"]))
    print("Best validation recall: {0}".format(train_history["best_val_recall"]))
    print("Best validation loss iter number: {0}".format(train_history["best_val_loss_iter"]))
    print("Best training accuracy: {0}".format(train_history["best_train_acc"]))
    print("Best training precision: {0}".format(train_history["best_train_precision"]))
    print("Best training recall: {0}".format(train_history["best_train_recall"]))
    print("------------------------------------------------")
    print()
    plt.plot(iterationList, train_history["train_time_list"])
    plt.savefig("Siamese Networks - Shuffled Data Set - Training Time.png")
    plt.plot(iterationList, train_history["train_loss_list"])
    plt.savefig("Siamese Networks - Shuffled Data Set - Training Loss.png")
    plt.plot(iterationList, train_history["train_acc_list"])
    plt.savefig("Siamese Networks - Shuffled Data Set - Training Accuracy.png")
    plt.plot(iterationList, train_history["train_precision_list"])
    plt.savefig("Siamese Networks - Shuffled Data Set - Training Precision.png")
    plt.plot(iterationList, train_history["train_recall_list"])
    plt.savefig("Siamese Networks - Shuffled Data Set - Training Recall.png")
    plt.plot(iterationList, train_history["val_loss_list"])
    plt.savefig("Siamese Networks - Shuffled Data Set - Validation Loss.png")
    plt.plot(iterationList, train_history["val_acc_list"])
    plt.savefig("Siamese Networks - Shuffled Data Set - Validation Accuracy.png")
    plt.plot(iterationList, train_history["val_precision_list"])
    plt.savefig("Siamese Networks - Shuffled Data Set - Validation Precision.png")
    plt.plot(iterationList, train_history["val_recall_list"])
    plt.savefig("Siamese Networks - Shuffled Data Set - Validation Recall.png")
    print()
    print()
    #Unseen Data Set
    print("Unseen Data Set:")
    batch_size = 32
    input_shape = (64,64,1)
    num_of_iters = 500
    evaluate_for_every = 25
    iterationList = np.arange(0, num_of_iters+1, evaluate_for_every)
    training_img_baseloc = "./15FeatureDataset/unseen-dataset/unseen-dataset/TrainingSet/"
    validation_img_baseloc = "./15FeatureDataset/unseen-dataset/unseen-dataset/ValidationSet/"
    trainingDS = excelDataRead(fileloc_unseen_training)
    validationDS = excelDataRead(fileloc_unseen_validation)
    model = siamese_net_getmodel(input_shape)
    train_history = siamese_net_train(model, trainingDS, validationDS, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, "DL Models/SN_unseen_dataset")
    print()
    print("------------------------------------------------")
    print("Model Training Complete. Model Saved.")
    print("Best validation loss: {0}".format(train_history["best_val_loss"]))
    print("Best validation accuracy: {0}".format(train_history["best_val_acc"]))
    print("Best validation precision: {0}".format(train_history["best_val_precision"]))
    print("Best validation recall: {0}".format(train_history["best_val_recall"]))
    print("Best validation loss iter number: {0}".format(train_history["best_val_loss_iter"]))
    print("Best training accuracy: {0}".format(train_history["best_train_acc"]))
    print("Best training precision: {0}".format(train_history["best_train_precision"]))
    print("Best training recall: {0}".format(train_history["best_train_recall"]))
    print("------------------------------------------------")
    print()
    plt.plot(iterationList, train_history["train_time_list"])
    plt.savefig("Siamese Networks - Unseen Data Set - Training Time.png")
    plt.plot(iterationList, train_history["train_loss_list"])
    plt.savefig("Siamese Networks - Unseen Data Set - Training Loss.png")
    plt.plot(iterationList, train_history["train_acc_list"])
    plt.savefig("Siamese Networks - Unseen Data Set - Training Accuracy.png")
    plt.plot(iterationList, train_history["train_precision_list"])
    plt.savefig("Siamese Networks - Unseen Data Set - Training Precision.png")
    plt.plot(iterationList, train_history["train_recall_list"])
    plt.savefig("Siamese Networks - Unseen Data Set - Training Recall.png")
    plt.plot(iterationList, train_history["val_loss_list"])
    plt.savefig("Siamese Networks - Unseen Data Set - Validation Loss.png")
    plt.plot(iterationList, train_history["val_acc_list"])
    plt.savefig("Siamese Networks - Unseen Data Set - Validation Accuracy.png")
    plt.plot(iterationList, train_history["val_precision_list"])
    plt.savefig("Siamese Networks - Unseen Data Set - Validation Precision.png")
    plt.plot(iterationList, train_history["val_recall_list"])
    plt.savefig("Siamese Networks - Unseen Data Set - Validation Recall.png")
    print()
    print()


def autoencoder_net_getDataForTesting(img_baseloc, size=2000):
    global featuresDS
    img_indices = np.random.randint(0, featuresDS.shape[0], size)
    x,y = [],[]
    for img_index in img_indices:
        img_loc = img_baseloc+(featuresDS["imagename"][img_index])
        while(True):
            img = np.array(imread(img_loc,0))
            try:
                img=(255.0-img)/255.0
                img=np.expand_dims(img, axis=2)
                break
            except:
                ind = np.random.randint(0, featuresDS.shape[0])
                img_loc = img_baseloc+(featuresDS["imagename"][ind])
        x.append(img)
        y.append(img)
    return np.array(x), np.array(y)
    

def autoencoder_net_initialize_bias(shape, name=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def autoencoder_net_initialize_weights(shape, name=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)


"""
    Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
"""
def autoencoder_net_getmodel(input_shape):
    #Encoding
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(padding='same', name='encoded_feature_vec'))
    #Decoding
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(1, (3,3), activation='sigmoid', padding='same', name='output'))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    #model.compile(loss="binary_crossentropy", optimizer=Adam(lr = 0.00006))
    model.summary()
    return model


def autoencoder_net_batchGen(batch_size, img_baseloc):
    global featuresDS
    while True:
        counter = 0
        img_indices = np.random.randint(0, featuresDS.shape[0], batch_size)
        x,y = [],[]
        for img_index in img_indices:
            img_loc = img_baseloc+(featuresDS["imagename"][img_index])
            while(True):
                img = np.array(imread(img_loc,0))
                try:
                    img=np.roll(axis=0,a=img,shift=(np.random.randint(-16,16)))
                    img=(255.0-img)/255.0
                    img=np.expand_dims(img, axis=2)
                    break
                except:
                    ind = np.random.randint(0, featuresDS.shape[0])
                    img_loc = img_baseloc+(featuresDS["imagename"][ind])
            x.append(img)
            y.append(img)
            counter+=1
        if counter == batch_size:
            yield np.array(x),np.array(y)


def autoencoder_net_batchGen2(batch_size, excel_data, img_baseloc):
    counter = 0
    while True:
        x1, x2, y = [], [], []
        for index in range(0,batch_size):
            left_img_loc = img_baseloc+(excel_data[counter][0])
            right_img_loc = img_baseloc+(excel_data[counter][1])
            left_img, right_img = None, None
            try:
                left_img = np.array(imread(left_img_loc,0))
                right_img = np.array(imread(right_img_loc,0))
                left_img=(255.0-left_img)/255.0
                right_img=(255.0-right_img)/255.0
                left_img=np.expand_dims(left_img, axis=2)
                right_img=np.expand_dims(right_img, axis=2)
                x1.append(left_img)
                x2.append(right_img)
                y.append(excel_data[counter][2])
            except:
                pass
            counter+=1
            if (counter==len(excel_data)):
                break
        if (counter==len(excel_data)):
            return np.array(x1),np.array(x2),np.array(y)
        else:
            yield np.array(x1),np.array(x2),np.array(y)


def autoencoder_net_train(model, trainingDS, validationDS, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, modelfilename_prefix):
    whole_val_x, whole_val_y = autoencoder_net_getDataForTesting(validation_img_baseloc)
    train_time_list = []
    train_loss_list = []
    val_loss_list = []
    best_val_loss = None
    best_val_loss_iter = None
    t_start = time.time()
    for iterNum, batchData in enumerate(autoencoder_net_batchGen(batch_size, training_img_baseloc)):
        if (iterNum==num_of_iters+1):
            break
        train_loss = model.train_on_batch(batchData[0], batchData[1])
        if iterNum % evaluate_for_every == 0:
            val_loss = model.test_on_batch(whole_val_x, whole_val_y)
            train_time = (time.time()-t_start)/60.0
            #Saving the model
            with open(modelfilename_prefix+"_model_arch_"+str(iterNum)+".json", "w") as json_file:
                json_file.write(model.to_json())
            model.save_weights(modelfilename_prefix+"_model_weights_"+str(iterNum)+".h5")
            train_time_list.append(train_time)
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            print("------------------------------------------------")
            print("Time for {0} iterations: {1} mins".format(iterNum, train_time))
            print("Train Loss: {0} ;     Val Loss: {1}".format(train_loss, val_loss))
            if (best_val_loss == None or val_loss < best_val_loss):
                print("Current validation loss best: {0}, previous validation loss best: {1}".format(val_loss, best_val_loss))
                best_val_loss = val_loss
                best_val_loss_iter = iterNum
            print("------------------------------------------------")
    shutil.copyfile(modelfilename_prefix+"_model_arch_"+str(best_val_loss_iter)+".json", modelfilename_prefix+"_model_arch_final.json")
    shutil.copyfile(modelfilename_prefix+"_model_weights_"+str(best_val_loss_iter)+".h5", modelfilename_prefix+"_model_weights_final.h5")
    print()
    print("------------------------------------------------")
    print("Model Training Complete. Model Saved.")
    print("Best validation loss: {0}".format(best_val_loss))
    print("Best validation loss iter number: {0}".format(best_val_loss_iter))
    print("------------------------------------------------")
    print()
    json_file2 = open(modelfilename_prefix+"_model_arch_final.json", 'r')
    loaded_model_json = json_file2.read()
    json_file2.close()
    final_model = model_from_json(loaded_model_json)
    final_model.load_weights(modelfilename_prefix+"_model_weights_final.h5")
    final_model.compile(optimizer='adadelta', loss='binary_crossentropy')
    encoder_model = Model(inputs=final_model.inputs,outputs=final_model.get_layer('encoded_feature_vec').output)
    Cos_similarity_score = []
    Y = []
    for batchData in autoencoder_net_batchGen2(250, trainingDS, training_img_baseloc):
        left_img_features = encoder_model.predict(batchData[0])
        left_img_features = left_img_features.reshape((250, 512))
        right_img_features = encoder_model.predict(batchData[1])
        right_img_features = right_img_features.reshape((250, 512))
        cos_similarity_score = cosine_similarity(left_img_features, right_img_features)
        cos_similarity_score = cos_similarity_score[0]
        Cos_similarity_score.extend(cos_similarity_score)
        Y.extend(batchData[2])
    thresholds_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    thresholds_train_acc_list = []
    thresholds_train_precision_list = []
    thresholds_train_recall_list = []
    thresholds_train_pr_diff_list = []
    for threshold in thresholds_list:
        Y_predicted = []
        for cos_similarity_score in Cos_similarity_score:
            if (cos_similarity_score<threshold):
                y_predicted = 0.0
            else:
                y_predicted = 1.0
            Y_predicted.append(y_predicted)
        precision = precision_score(Y, Y_predicted) * 100
        recall = recall_score(Y, Y_predicted) * 100
        pr_diff = np.absolute(precision - recall)
        accuracy = accuracy_score(Y, Y_predicted) * 100
        thresholds_train_acc_list.append(accuracy)
        thresholds_train_precision_list.append(precision)
        thresholds_train_recall_list.append(recall)
        thresholds_train_pr_diff_list.append(pr_diff)
    print(thresholds_train_acc_list)
    print(thresholds_train_precision_list)
    print(thresholds_train_recall_list)
    best_train_pr_diff = max(thresholds_train_acc_list)
    best_train_pr_diff_index = thresholds_train_acc_list.index(best_train_pr_diff)
    best_train_precision = thresholds_train_precision_list[best_train_pr_diff_index]
    best_train_recall = thresholds_train_recall_list[best_train_pr_diff_index]
    best_train_acc = thresholds_train_acc_list[best_train_pr_diff_index]
    best_threshold = thresholds_list[best_train_pr_diff_index]
    Cos_similarity_score = []
    Y = []
    for batchData in autoencoder_net_batchGen2(250, validationDS, validation_img_baseloc):
        left_img_features = encoder_model.predict(batchData[0])
        left_img_features = left_img_features.reshape((250, 512))
        right_img_features = encoder_model.predict(batchData[1])
        right_img_features = right_img_features.reshape((250, 512))
        cos_similarity_score = cosine_similarity(left_img_features, right_img_features)
        cos_similarity_score = cos_similarity_score[0]
        Cos_similarity_score.extend(cos_similarity_score)
        Y.extend(batchData[2])
    Y_predicted = []
    for cos_similarity_score in Cos_similarity_score:
        if (cos_similarity_score<best_threshold):
            y_predicted = 0.0
        else:
            y_predicted = 1.0
        Y_predicted.append(y_predicted)
    best_val_precision = precision_score(Y, Y_predicted) * 100
    best_val_recall = recall_score(Y, Y_predicted) * 100
    best_val_acc = accuracy_score(Y, Y_predicted) * 100
    return { "train_loss_list": train_loss_list, "val_loss_list": val_loss_list, "train_time_list": train_time_list, "best_threshold": best_threshold, "thresholds_list": thresholds_list,
    "thresholds_train_acc_list": thresholds_train_acc_list,  "thresholds_train_precision_list": thresholds_train_precision_list, "thresholds_train_recall_list": thresholds_train_recall_list,
    "best_train_precision": best_train_precision, "best_train_recall": best_train_recall, "best_train_acc": best_train_acc,
    "best_val_precision": best_val_precision, "best_val_recall": best_val_recall, "best_val_acc": best_val_acc }
    
    
def autoencoder_net_main():
    global featuresDS
    print("AUTOENCODER NETWORKS:")
    #Fetch features excel
    featuresDS = excelDataRead(fileloc_features, True)
    #Seen Data Set
    print("Seen Data Set:")
    batch_size = 64
    input_shape = (64,64,1)
    num_of_iters = 7500
    evaluate_for_every = 250
    iterationList = np.arange(0, num_of_iters+1, evaluate_for_every)
    training_img_baseloc = "./15FeatureDataset/seen-dataset/seen-dataset/TrainingSet/"
    validation_img_baseloc = "./15FeatureDataset/seen-dataset/seen-dataset/ValidationSet/"
    trainingDS = excelDataRead(fileloc_seen_training)
    validationDS = excelDataRead(fileloc_seen_validation)
    model = autoencoder_net_getmodel(input_shape)
    train_history = autoencoder_net_train(model, trainingDS, validationDS, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, "DL Models/AN_seen_dataset")
    
    plt.plot(iterationList, train_history["train_time_list"])
    plt.xlabel('Num of Iterations')
    plt.ylabel('Training Time (in min)')
    plt.title('Autoencoder Networks - Seen Data Set - Training Time')
    plt.xlim(min(iterationList), max(iterationList))
    plt.ylim(min(train_history["train_time_list"])-1, max(train_history["train_time_list"])+1)
    plt.savefig("Autoencoder Networks - Seen Data Set - Training Time.png")
    plt.clf()

    plt.plot(iterationList, train_history["train_loss_list"])
    plt.xlabel('Num of Iterations')
    plt.ylabel('Training Loss')
    plt.title('Autoencoder Networks - Seen Data Set - Training Loss')
    plt.xlim(min(iterationList), max(iterationList))
    plt.ylim(min(train_history["train_loss_list"])-0.1, max(train_history["train_loss_list"])+0.1)
    plt.savefig("Autoencoder Networks - Seen Data Set - Training Loss.png")
    plt.clf()
    
    plt.plot(iterationList, train_history["val_loss_list"])
    plt.xlabel('Num of Iterations')
    plt.ylabel('Validation Loss')
    plt.title('Autoencoder Networks - Seen Data Set - Validation Loss')
    plt.xlim(min(iterationList), max(iterationList))
    plt.ylim(min(train_history["val_loss_list"])-0.1, max(train_history["val_loss_list"])+0.1)
    plt.savefig("Autoencoder Networks - Seen Data Set - Validation Loss.png")
    plt.clf()
    
    plt.plot(train_history["thresholds_list"], train_history["thresholds_train_acc_list"])
    plt.xlabel('Thresholds')
    plt.ylabel('Training Accuracy')
    plt.title('Autoencoder Networks - Seen Data Set - Threshold vs Training Acc')
    plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
    plt.ylim(min(train_history["thresholds_train_acc_list"])-5, max(train_history["thresholds_train_acc_list"])+5)
    plt.savefig("Autoencoder Networks - Seen Data Set - Threshold vs Training Acc.png")
    plt.clf()
    
    plt.plot(train_history["thresholds_list"], train_history["thresholds_train_precision_list"])
    plt.xlabel('Thresholds')
    plt.ylabel('Training Precision')
    plt.title('Autoencoder Networks - Seen Data Set - Threshold vs Training Precision')
    plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
    plt.ylim(min(train_history["thresholds_train_precision_list"])-5, max(train_history["thresholds_train_precision_list"])+5)
    plt.savefig("Autoencoder Networks - Seen Data Set - Threshold vs Training Precision.png")
    plt.clf()
    
    plt.plot(train_history["thresholds_list"], train_history["thresholds_train_recall_list"])
    plt.xlabel('Thresholds')
    plt.ylabel('Training Recall')
    plt.title('Autoencoder Networks - Seen Data Set - Threshold vs Training Recall')
    plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
    plt.ylim(min(train_history["thresholds_train_recall_list"])-5, max(train_history["thresholds_train_recall_list"])+5)
    plt.savefig("Autoencoder Networks - Seen Data Set - Threshold vs Training Recall.png")
    plt.clf()
    
    print()
    print("------------------------------------------------")
    print("For similarity detection:")
    print("Best training accuracy: {0}".format(train_history["best_train_acc"]))
    print("Best training precision: {0}".format(train_history["best_train_precision"]))
    print("Best training recall: {0}".format(train_history["best_train_recall"]))
    print("Best validation accuracy: {0}".format(train_history["best_val_acc"]))
    print("Best validation precision: {0}".format(train_history["best_val_precision"]))
    print("Best validation recall: {0}".format(train_history["best_val_recall"]))
    print("Best cosine threshold: {0}".format(train_history["best_threshold"]))
    print("------------------------------------------------")
    print()
    print()
    print()
    #Shuffled Data Set
    print("Shuffled Data Set:")
    batch_size = 64
    input_shape = (64,64,1)
    num_of_iters = 5000
    evaluate_for_every = 250
    iterationList = np.arange(0, num_of_iters+1, evaluate_for_every)
    training_img_baseloc = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/TrainingSet/"
    validation_img_baseloc = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/ValidationSet/"
    trainingDS = excelDataRead(fileloc_shuffled_training)
    validationDS = excelDataRead(fileloc_shuffled_validation)
    model = autoencoder_net_getmodel(input_shape)
    train_history = autoencoder_net_train(model, trainingDS, validationDS, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, "DL Models/AN_shuffled_dataset")
    
    plt.plot(iterationList, train_history["train_time_list"])
    plt.xlabel('Num of Iterations')
    plt.ylabel('Training Time (in min)')
    plt.title('Autoencoder Networks - Shuffled Data Set - Training Time')
    plt.xlim(min(iterationList), max(iterationList))
    plt.ylim(min(train_history["train_time_list"])-1, max(train_history["train_time_list"])+1)
    plt.savefig("Autoencoder Networks - Shuffled Data Set - Training Time.png")
    plt.clf()

    plt.plot(iterationList, train_history["train_loss_list"])
    plt.xlabel('Num of Iterations')
    plt.ylabel('Training Loss')
    plt.title('Autoencoder Networks - Shuffled Data Set - Training Loss')
    plt.xlim(min(iterationList), max(iterationList))
    plt.ylim(min(train_history["train_loss_list"])-0.1, max(train_history["train_loss_list"])+0.1)
    plt.savefig("Autoencoder Networks - Shuffled Data Set - Training Loss.png")
    plt.clf()

    plt.plot(iterationList, train_history["val_loss_list"])
    plt.xlabel('Num of Iterations')
    plt.ylabel('Validation Loss')
    plt.title('Autoencoder Networks - Shuffled Data Set - Validation Loss')
    plt.xlim(min(iterationList), max(iterationList))
    plt.ylim(min(train_history["val_loss_list"])-0.1, max(train_history["val_loss_list"])+0.1)
    plt.savefig("Autoencoder Networks - Shuffled Data Set - Validation Loss.png")
    plt.clf()

    plt.plot(train_history["thresholds_list"], train_history["thresholds_train_acc_list"])
    plt.xlabel('Thresholds')
    plt.ylabel('Training Accuracy')
    plt.title('Autoencoder Networks - Shuffled Data Set - Threshold vs Training Acc')
    plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
    plt.ylim(min(train_history["thresholds_train_acc_list"])-5, max(train_history["thresholds_train_acc_list"])+5)
    plt.savefig("Autoencoder Networks - Shuffled Data Set - Threshold vs Training Acc.png")
    plt.clf()

    plt.plot(train_history["thresholds_list"], train_history["thresholds_train_precision_list"])
    plt.xlabel('Thresholds')
    plt.ylabel('Training Precision')
    plt.title('Autoencoder Networks - Shuffled Data Set - Threshold vs Training Precision')
    plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
    plt.ylim(min(train_history["thresholds_train_precision_list"])-5, max(train_history["thresholds_train_precision_list"])+5)
    plt.savefig("Autoencoder Networks - Shuffled Data Set - Threshold vs Training Precision.png")
    plt.clf()

    plt.plot(train_history["thresholds_list"], train_history["thresholds_train_recall_list"])
    plt.xlabel('Thresholds')
    plt.ylabel('Training Recall')
    plt.title('Autoencoder Networks - Shuffled Data Set - Threshold vs Training Recall')
    plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
    plt.ylim(min(train_history["thresholds_train_recall_list"])-5, max(train_history["thresholds_train_recall_list"])+5)
    plt.savefig("Autoencoder Networks - Shuffled Data Set - Threshold vs Training Recall.png")
    plt.clf()

    print()
    print("------------------------------------------------")
    print("For similarity detection:")
    print("Best training accuracy: {0}".format(train_history["best_train_acc"]))
    print("Best training precision: {0}".format(train_history["best_train_precision"]))
    print("Best training recall: {0}".format(train_history["best_train_recall"]))
    print("Best validation accuracy: {0}".format(train_history["best_val_acc"]))
    print("Best validation precision: {0}".format(train_history["best_val_precision"]))
    print("Best validation recall: {0}".format(train_history["best_val_recall"]))
    print("Best cosine threshold: {0}".format(train_history["best_threshold"]))
    print("------------------------------------------------")
    print()
    print()
    print()
    #Unseen Data Set
    print("Unseen Data Set:")
    batch_size = 64
    input_shape = (64,64,1)
    num_of_iters = 5000
    evaluate_for_every = 250
    iterationList = np.arange(0, num_of_iters+1, evaluate_for_every)
    training_img_baseloc = "./15FeatureDataset/unseen-dataset/unseen-dataset/TrainingSet/"
    validation_img_baseloc = "./15FeatureDataset/unseen-dataset/unseen-dataset/ValidationSet/"
    trainingDS = excelDataRead(fileloc_unseen_training)
    validationDS = excelDataRead(fileloc_unseen_validation)
    model = autoencoder_net_getmodel(input_shape)
    train_history = autoencoder_net_train(model, trainingDS, validationDS, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, "DL Models/AN_unseen_dataset")
    
    plt.plot(iterationList, train_history["train_time_list"])
    plt.xlabel('Num of Iterations')
    plt.ylabel('Training Time (in min)')
    plt.title('Autoencoder Networks - Unseen Data Set - Training Time')
    plt.xlim(min(iterationList), max(iterationList))
    plt.ylim(min(train_history["train_time_list"])-1, max(train_history["train_time_list"])+1)
    plt.savefig("Autoencoder Networks - Unseen Data Set - Training Time.png")
    plt.clf()

    plt.plot(iterationList, train_history["train_loss_list"])
    plt.xlabel('Num of Iterations')
    plt.ylabel('Training Loss')
    plt.title('Autoencoder Networks - Unseen Data Set - Training Loss')
    plt.xlim(min(iterationList), max(iterationList))
    plt.ylim(min(train_history["train_loss_list"])-0.1, max(train_history["train_loss_list"])+0.1)
    plt.savefig("Autoencoder Networks - Unseen Data Set - Training Loss.png")
    plt.clf()

    plt.plot(iterationList, train_history["val_loss_list"])
    plt.xlabel('Num of Iterations')
    plt.ylabel('Validation Loss')
    plt.title('Autoencoder Networks - Unseen Data Set - Validation Loss')
    plt.xlim(min(iterationList), max(iterationList))
    plt.ylim(min(train_history["val_loss_list"])-0.1, max(train_history["val_loss_list"])+0.1)
    plt.savefig("Autoencoder Networks - Unseen Data Set - Validation Loss.png")
    plt.clf()

    plt.plot(train_history["thresholds_list"], train_history["thresholds_train_acc_list"])
    plt.xlabel('Thresholds')
    plt.ylabel('Training Accuracy')
    plt.title('Autoencoder Networks - Unseen Data Set - Threshold vs Training Acc')
    plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
    plt.ylim(min(train_history["thresholds_train_acc_list"])-5, max(train_history["thresholds_train_acc_list"])+5)
    plt.savefig("Autoencoder Networks - Unseen Data Set - Threshold vs Training Acc.png")
    plt.clf()

    plt.plot(train_history["thresholds_list"], train_history["thresholds_train_precision_list"])
    plt.xlabel('Thresholds')
    plt.ylabel('Training Precision')
    plt.title('Autoencoder Networks - Unseen Data Set - Threshold vs Training Precision')
    plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
    plt.ylim(min(train_history["thresholds_train_precision_list"])-5, max(train_history["thresholds_train_precision_list"])+5)
    plt.savefig("Autoencoder Networks - Unseen Data Set - Threshold vs Training Precision.png")
    plt.clf()

    plt.plot(train_history["thresholds_list"], train_history["thresholds_train_recall_list"])
    plt.xlabel('Thresholds')
    plt.ylabel('Training Recall')
    plt.title('Autoencoder Networks - Unseen Data Set - Threshold vs Training Recall')
    plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
    plt.ylim(min(train_history["thresholds_train_recall_list"])-5, max(train_history["thresholds_train_recall_list"])+5)
    plt.savefig("Autoencoder Networks - Unseen Data Set - Threshold vs Training Recall.png")
    plt.clf()

    print()
    print("------------------------------------------------")
    print("For similarity detection:")
    print("Best training accuracy: {0}".format(train_history["best_train_acc"]))
    print("Best training precision: {0}".format(train_history["best_train_precision"]))
    print("Best training recall: {0}".format(train_history["best_train_recall"]))
    print("Best validation accuracy: {0}".format(train_history["best_val_acc"]))
    print("Best validation precision: {0}".format(train_history["best_val_precision"]))
    print("Best validation recall: {0}".format(train_history["best_val_recall"]))
    print("Best cosine threshold: {0}".format(train_history["best_threshold"]))
    print("------------------------------------------------")
    print()
    print()
    print()


def main():
    #siamese_net_main()
    print()
    print()
    print()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print()
    print()
    print()
    autoencoder_net_main()


main()


'''
References:
    1. https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
    2. https://github.com/hlamba28/One-Shot-Learning-with-Siamese-Networks/blob/master/Siamese%20on%20Omniglot%20Dataset.ipynb
'''