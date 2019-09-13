import pandas as pd
import numpy as np
import csv
import time
import shutil
import matplotlib.pyplot as plt
from cv2 import imread, imwrite, imshow, destroyAllWindows, waitKey
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, UpSampling2D, Flatten, Dense
from keras.models import Model, model_from_json
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils import to_categorical


fileloc_features = "./15FeatureDataset/15features.csv"
fileloc_seen_training = "./15FeatureDataset/seen-dataset/seen-dataset/dataset_seen_training_siamese.csv"
fileloc_seen_validation = "./15FeatureDataset/seen-dataset/seen-dataset/dataset_seen_validation_siamese.csv"
fileloc_shuffled_training = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/dataset_seen_training_siamese.csv"
fileloc_shuffled_validation = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/dataset_seen_validation_siamese.csv"
fileloc_unseen_training = "./15FeatureDataset/unseen-dataset/unseen-dataset/dataset_seen_training_siamese.csv"
fileloc_unseen_validation = "./15FeatureDataset/unseen-dataset/unseen-dataset/dataset_seen_validation_siamese.csv"
features_names_list = ["pen_pressure", "letter_spacing", "size", "dimension", "is_lowercase", "is_continuous", "slantness", "tilt",
	"entry_stroke_a", "staff_of_a", "formation_n", "staff_of_d", "exit_stroke_d", "word_formation", "constancy"]
features_values_list = [2, 3, 3, 3, 2, 2, 4, 2, 2, 4, 2, 3, 4, 2, 2]


def getDataForTesting(img_baseloc, featuresDS, size=2000):
	global features_names_list
	ind = None
	img_indices = np.random.randint(0, featuresDS.shape[0], size)
	x,y = [],[]
	for img_index in img_indices:
		ind = img_index
		img_loc = img_baseloc+(featuresDS["imagename"][ind])
		temp = []
		while(True):
			img = np.array(imread(img_loc,0))
			try:
				img=(255.0-img)/255.0
				img=np.expand_dims(img, axis=2)
				for features_names in features_names_list:
					temp.append(featuresDS[features_names][ind]-1)
				break
			except:
				ind = np.random.randint(0, featuresDS.shape[0])
				img_loc = img_baseloc+(featuresDS["imagename"][ind])
		x.append(img)
		y.append(temp)
	x=np.array(x)
	y=np.array(y)
	y = y.transpose()
	y_ = []
	for i in range(len(y)):
		y_.append(to_categorical(y[i],num_classes=features_values_list[i]))
	return x,y_


def batchGen(batch_size, img_baseloc, featuresDS):
    while True:
        ind = None
        img_indices = np.random.randint(0, featuresDS.shape[0], batch_size)
        x,y = [],[]
        for img_index in img_indices:
            ind = img_index
            img_loc = img_baseloc+(featuresDS["imagename"][ind])
            temp = []
            while(True):
                img = np.array(imread(img_loc,0))
                try:
                    img=np.roll(axis=0,a=img,shift=(np.random.randint(-16,16)))
                    img=(255.0-img)/255.0
                    img=np.expand_dims(img, axis=2)
                    for features_names in features_names_list:
                    	temp.append(featuresDS[features_names][ind]-1)
                    break
                except:
                    ind = np.random.randint(0, featuresDS.shape[0])
                    img_loc = img_baseloc+(featuresDS["imagename"][ind])
            x.append(img)
            y.append(temp)
        x=np.array(x)
        y=np.array(y)
        yy = y
        y = y.transpose()
        y_ = []
        for i in range(len(y)):
            y_.append(to_categorical(y[i],num_classes=features_values_list[i]))
        yield x, y_, yy


def batchGen2(batch_size, excel_data, img_baseloc):
    counter = 0
    while True:
        x1, x2, y = [], [], []
        for index in range(0,batch_size):
            left_img_loc = img_baseloc+(excel_data.at[counter, 'left'])
            right_img_loc = img_baseloc+(excel_data.at[counter, 'right'])
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
                y.append(excel_data.at[counter, 'label'])
            except:
                pass
            counter+=1
            if (counter==(excel_data.shape[0])):
                break
        if (counter==(excel_data.shape[0])):
            return np.array(x1),np.array(x2),np.array(y)
        else:
            yield np.array(x1),np.array(x2),np.array(y)


def train(model, featuresDS, seen_training, seen_validation, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, modelfilename_prefix):
	whole_train_x, whole_train_y = getDataForTesting(training_img_baseloc, featuresDS)
	whole_val_x, whole_val_y = getDataForTesting(validation_img_baseloc, featuresDS)
	train_time_list = []
	val_loss_list = []
	train_loss_list = []
	best_val_loss = None
	best_val_loss_iter = None
	t_start = time.time()
	for iterNum,batchData in enumerate(batchGen(batch_size, training_img_baseloc, featuresDS)):
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
	final_model = model
	final_model.load_weights(modelfilename_prefix+"_model_weights_final.h5")
	Y = []
	Y_predicted = []
	acc = [] 
	max_iters_for_testing = 100
	for iterNum,batchData in enumerate(batchGen(batch_size, training_img_baseloc, featuresDS)):
		if (iterNum==max_iters_for_testing):
			break
		for i in range(batch_size):
			features = batchData[2][i]
			features_probs = final_model.predict(np.expand_dims(batchData[0][i], axis=0))
			features_pred=[]
			for prob in features_probs:
				features_pred.append(np.argmax(prob)+1)
			Y.append(features)
			Y_predicted.append(features_pred)
			cnt = 0
			for a,b in zip(features, features_pred):
				if (a==b):
					cnt+=1
			acc.append(cnt/15*100)
	acc = np.array(acc)
	dl_model_train_acc = np.mean(acc)
	Y = []
	Y_predicted = []
	acc = [] 
	max_iters_for_testing = 100
	for iterNum,batchData in enumerate(batchGen(batch_size, validation_img_baseloc, featuresDS)):
		if (iterNum==max_iters_for_testing):
			break
		for i in range(batch_size):
			features = batchData[2][i]
			features_probs = final_model.predict(np.expand_dims(batchData[0][i], axis=0))
			features_pred=[]
			for prob in features_probs:
				features_pred.append(np.argmax(prob)+1)
			Y.append(features)
			Y_predicted.append(features_pred)
			cnt = 0
			for a,b in zip(features, features_pred):
				if (a==b):
					cnt+=1
			acc.append(cnt/15*100)
	acc = np.array(acc)
	dl_model_val_acc = np.mean(acc)
	print()
	print("------------------------------------------------")
	print("Model Training Complete. Model Saved.")
	print("Best validation loss: {0}".format(best_val_loss))
	print("Best training accuracy: {0}".format(dl_model_train_acc))
	print("Best validation accuracy: {0}".format(dl_model_val_acc))
	print("Best validation loss iter number: {0}".format(best_val_loss_iter))
	print("------------------------------------------------")
	print()
	Cos_similarity_score = []
	Y = []
	for batchData in batchGen2(250, seen_training, training_img_baseloc):
		for i in range(250):
			left_img_features_probs = final_model.predict(np.expand_dims(batchData[0][i], axis=0))
			right_img_features_probs = final_model.predict(np.expand_dims(batchData[1][i], axis=0))
			left_img_features_pred=[]
			for left_img_prob in left_img_features_probs:
				left_img_features_pred.append(np.argmax(left_img_prob)+1)
			right_img_features_pred=[]
			for right_img_prob in right_img_features_probs:
				right_img_features_pred.append(np.argmax(right_img_prob)+1)
			cos_similarity_score = cosine_similarity((np.array(left_img_features_pred)).reshape(1, -1), (np.array(right_img_features_pred)).reshape(1, -1))
			cos_similarity_score = cos_similarity_score[0]
			Cos_similarity_score.append(cos_similarity_score)
			Y.append(batchData[2][i])
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
	for batchData in batchGen2(250, seen_validation, validation_img_baseloc):
		for i in range(250):
			left_img_features_probs = final_model.predict(np.expand_dims(batchData[0][i], axis=0))
			right_img_features_probs = final_model.predict(np.expand_dims(batchData[1][i], axis=0))
			left_img_features_pred=[]
			for left_img_prob in left_img_features_probs:
				left_img_features_pred.append(np.argmax(left_img_prob)+1)
			right_img_features_pred=[]
			for right_img_prob in right_img_features_probs:
				right_img_features_pred.append(np.argmax(right_img_prob)+1)
			cos_similarity_score = cosine_similarity((np.array(left_img_features_pred)).reshape(1, -1), (np.array(right_img_features_pred)).reshape(1, -1))
			cos_similarity_score = cos_similarity_score[0]
			Cos_similarity_score.append(cos_similarity_score)
			Y.append(batchData[2][i])
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


def get_model(modelfilename_prefix):
	json_file2 = open(modelfilename_prefix+"_model_arch_final.json", 'r')
	loaded_model_json = json_file2.read()
	json_file2.close()
	autoencoder_model = model_from_json(loaded_model_json)
	autoencoder_model.load_weights(modelfilename_prefix+"_model_weights_final.h5")
	#autoencoder_model.compile(optimizer='adadelta', loss='binary_crossentropy')
	encoder_model = Model(autoencoder_model.input, autoencoder_model.get_layer('encoded_feature_vec').output)
	encoder_model_output_layer = encoder_model.get_layer('encoded_feature_vec').output
	features_model_end_layers=[]
	for i in range(1,len(features_values_list)+1):
		x1 = (Flatten(name='flatten_'+str(i)))(encoder_model_output_layer)
		x2 = (Dense(256 , activation='relu', name = 'dense_layer_'+str(i)))(x1)
		x3 = (Dense(features_values_list[i-1] , activation='softmax', name = 'out_feature_'+str(i)))(x2)
		features_model_end_layers.append(x3)
	features_model = Model(inputs=encoder_model.input, outputs=features_model_end_layers)
	losses = {}
	lossWeights = {}
	for i in range(1,16):
	    losses["out_feature_"+str(i)] = "categorical_crossentropy"
	    lossWeights["out_feature_"+str(i)] = 1.0
	opt = SGD(lr=0.005, decay=1e-6, momentum=0.95, nesterov=True)
	features_model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
	features_model.summary()
	return features_model


def main():
	global features_values_list
	colors = [(1.0,0.0,0.0),(1.0,1.0,0.0),(0.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),
	(0.0,1.0,1.0),(1.0,0.0,1.0),(0.0,0.3,0.7),(0.7,0.3,0.2),(0.6,0.3,0.0),
	(0.1,0.2,0.3),(0.4,0.8,0.2),(0.4,0.2,0.7),(0.0,0.4,0.1),(0.8,0.2,0.4)]
	features_data = pd.read_csv(fileloc_features)
	print("Seen Data Set:")
	batch_size = 64
	input_shape = (64,64,1)
	num_of_iters = 12000
	evaluate_for_every = 125
	iterationList = np.arange(0, num_of_iters+1, evaluate_for_every)
	training_img_baseloc = "./15FeatureDataset/seen-dataset/seen-dataset/TrainingSet/"
	validation_img_baseloc = "./15FeatureDataset/seen-dataset/seen-dataset/ValidationSet/"
	trainingDS = pd.read_csv(fileloc_seen_training, usecols = ['left','right','label'])
	validationDS = pd.read_csv(fileloc_seen_validation, usecols = ['left','right','label'])
	model = get_model("DL Models/AN_seen_dataset")
	train_history = train(model, features_data, trainingDS, validationDS, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, "DL Models/MT_seen_dataset")

	plt.plot(iterationList, train_history["train_time_list"])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Training Time (in min)')
	plt.title('MT Networks - Seen Data Set - Training Time')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(min(train_history["train_time_list"])-1, max(train_history["train_time_list"])+1)
	plt.savefig("MT Networks - Seen Data Set - Training Time.png")
	plt.clf()

	A = np.array(train_history["train_loss_list"]) 
	A = np.transpose(A)
	for a_i in range(16,31):
		plt.plot(iterationList, A[a_i], color=colors[a_i-16])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Training Loss')
	plt.title('MT Networks - Seen Data Set - Training Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(np.array(train_history["train_loss_list"]))-0.1, np.amax(np.array(train_history["train_loss_list"]))+0.1)
	plt.savefig("MT Networks - Seen Data Set - Training Loss.png")
	plt.clf()

	plt.plot(iterationList, A[0])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Total Training Loss')
	plt.title('MT Networks - Seen Data Set - Total Training Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(A[0])-1, np.amax(A[0])+1)
	plt.savefig("MT Networks - Seen Data Set - Total Training Loss.png")
	plt.clf()

	A = np.array(train_history["val_loss_list"]) 
	A = np.transpose(A)
	for a_i in range(16,31):
		plt.plot(iterationList, A[a_i], color=colors[a_i-16])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Validation Loss')
	plt.title('MT Networks - Seen Data Set - Validation Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(np.array(train_history["val_loss_list"]))-0.1, np.amax(np.array(train_history["val_loss_list"]))+0.1)
	plt.savefig("MT Networks - Seen Data Set - Validation Loss.png")
	plt.clf()

	plt.plot(iterationList, A[0])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Total Validation Loss')
	plt.title('MT Networks - Seen Data Set - Total Validation Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(A[0])-1, np.amax(A[0])+1)
	plt.savefig("MT Networks - Seen Data Set - Total Validation Loss.png")
	plt.clf()

	plt.plot(train_history["thresholds_list"], train_history["thresholds_train_acc_list"])
	plt.xlabel('Thresholds')
	plt.ylabel('Training Accuracy')
	plt.title('MT Networks - Seen Data Set - Threshold vs Training Acc')
	plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
	plt.ylim(min(train_history["thresholds_train_acc_list"])-5, max(train_history["thresholds_train_acc_list"])+5)
	plt.savefig("MT Networks - Seen Data Set - Threshold vs Training Acc.png")
	plt.clf()

	plt.plot(train_history["thresholds_list"], train_history["thresholds_train_precision_list"])
	plt.xlabel('Thresholds')
	plt.ylabel('Training Precision')
	plt.title('MT Networks - Seen Data Set - Threshold vs Training Precision')
	plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
	plt.ylim(min(train_history["thresholds_train_precision_list"])-5, max(train_history["thresholds_train_precision_list"])+5)
	plt.savefig("MT Networks - Seen Data Set - Threshold vs Training Precision.png")
	plt.clf()

	plt.plot(train_history["thresholds_list"], train_history["thresholds_train_recall_list"])
	plt.xlabel('Thresholds')
	plt.ylabel('Training Recall')
	plt.title('MT Networks - Seen Data Set - Threshold vs Training Recall')
	plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
	plt.ylim(min(train_history["thresholds_train_recall_list"])-5, max(train_history["thresholds_train_recall_list"])+5)
	plt.savefig("MT Networks - Seen Data Set - Threshold vs Training Recall.png")
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
	num_of_iters = 2000
	evaluate_for_every = 125
	iterationList = np.arange(0, num_of_iters+1, evaluate_for_every)
	training_img_baseloc = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/TrainingSet/"
	validation_img_baseloc = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/ValidationSet/"
	trainingDS = pd.read_csv(fileloc_shuffled_training, usecols = ['left','right','label'])
	validationDS = pd.read_csv(fileloc_shuffled_validation, usecols = ['left','right','label'])
	model = get_model("DL Models/AN_shuffled_dataset")
	train_history = train(model, features_data, trainingDS, validationDS, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, "DL Models/MT_shuffled_dataset")

	plt.plot(iterationList, train_history["train_time_list"])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Training Time (in min)')
	plt.title('MT Networks - Shuffled Data Set - Training Time')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(min(train_history["train_time_list"])-1, max(train_history["train_time_list"])+1)
	plt.savefig("MT Networks - Shuffled Data Set - Training Time.png")
	plt.clf()

	A = np.array(train_history["train_loss_list"]) 
	A = np.transpose(A)
	for a_i in range(16,31):
		plt.plot(iterationList, A[a_i], color=colors[a_i-16])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Training Loss')
	plt.title('MT Networks - Shuffled Data Set - Training Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(np.array(train_history["train_loss_list"]))-0.1, np.amax(np.array(train_history["train_loss_list"]))+0.1)
	plt.savefig("MT Networks - Shuffled Data Set - Training Loss.png")
	plt.clf()

	plt.plot(iterationList, A[0])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Total Training Loss')
	plt.title('MT Networks - Shuffled Data Set - Total Training Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(A[0])-1, np.amax(A[0])+1)
	plt.savefig("MT Networks - Shuffled Data Set - Total Training Loss.png")
	plt.clf()

	A = np.array(train_history["val_loss_list"]) 
	A = np.transpose(A)
	for a_i in range(16,31):
		plt.plot(iterationList, A[a_i], color=colors[a_i-16])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Validation Loss')
	plt.title('MT Networks - Shuffled Data Set - Validation Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(np.array(train_history["val_loss_list"]))-0.1, np.amax(np.array(train_history["val_loss_list"]))+0.1)
	plt.savefig("MT Networks - Shuffled Data Set - Validation Loss.png")
	plt.clf()

	plt.plot(iterationList, A[0])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Total Validation Loss')
	plt.title('MT Networks - Shuffled Data Set - Total Validation Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(A[0])-1, np.amax(A[0])+1)
	plt.savefig("MT Networks - Shuffled Data Set - Total Validation Loss.png")
	plt.clf()

	plt.plot(train_history["thresholds_list"], train_history["thresholds_train_acc_list"])
	plt.xlabel('Thresholds')
	plt.ylabel('Training Accuracy')
	plt.title('MT Networks - Shuffled Data Set - Threshold vs Training Acc')
	plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
	plt.ylim(min(train_history["thresholds_train_acc_list"])-5, max(train_history["thresholds_train_acc_list"])+5)
	plt.savefig("MT Networks - Shuffled Data Set - Threshold vs Training Acc.png")
	plt.clf()

	plt.plot(train_history["thresholds_list"], train_history["thresholds_train_precision_list"])
	plt.xlabel('Thresholds')
	plt.ylabel('Training Precision')
	plt.title('MT Networks - Shuffled Data Set - Threshold vs Training Precision')
	plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
	plt.ylim(min(train_history["thresholds_train_precision_list"])-5, max(train_history["thresholds_train_precision_list"])+5)
	plt.savefig("MT Networks - Shuffled Data Set - Threshold vs Training Precision.png")
	plt.clf()

	plt.plot(train_history["thresholds_list"], train_history["thresholds_train_recall_list"])
	plt.xlabel('Thresholds')
	plt.ylabel('Training Recall')
	plt.title('MT Networks - Shuffled Data Set - Threshold vs Training Recall')
	plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
	plt.ylim(min(train_history["thresholds_train_recall_list"])-5, max(train_history["thresholds_train_recall_list"])+5)
	plt.savefig("MT Networks - Shuffled Data Set - Threshold vs Training Recall.png")
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
	num_of_iters = 2000
	evaluate_for_every = 125
	iterationList = np.arange(0, num_of_iters+1, evaluate_for_every)
	training_img_baseloc = "./15FeatureDataset/unseen-dataset/unseen-dataset/TrainingSet/"
	validation_img_baseloc = "./15FeatureDataset/unseen-dataset/unseen-dataset/ValidationSet/"
	trainingDS = pd.read_csv(fileloc_unseen_training, usecols = ['left','right','label'])
	validationDS = pd.read_csv(fileloc_unseen_validation, usecols = ['left','right','label'])
	model = get_model("DL Models/AN_unseen_dataset")
	train_history = train(model, features_data, trainingDS, validationDS, num_of_iters, batch_size, training_img_baseloc, validation_img_baseloc, evaluate_for_every, "DL Models/MT_unseen_dataset")

	plt.plot(iterationList, train_history["train_time_list"])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Training Time (in min)')
	plt.title('MT Networks - Unseen Data Set - Training Time')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(min(train_history["train_time_list"])-1, max(train_history["train_time_list"])+1)
	plt.savefig("MT Networks - Unseen Data Set - Training Time.png")
	plt.clf()

	A = np.array(train_history["train_loss_list"]) 
	A = np.transpose(A)
	for a_i in range(16,31):
		plt.plot(iterationList, A[a_i], color=colors[a_i-16])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Training Loss')
	plt.title('MT Networks - Unseen Data Set - Training Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(np.array(train_history["train_loss_list"]))-0.1, np.amax(np.array(train_history["train_loss_list"]))+0.1)
	plt.savefig("MT Networks - Unseen Data Set - Training Loss.png")
	plt.clf()

	plt.plot(iterationList, A[0])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Total Training Loss')
	plt.title('MT Networks - UnSeen Data Set - Total Training Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(A[0])-1, np.amax(A[0])+1)
	plt.savefig("MT Networks - UnSeen Data Set - Total Training Loss.png")
	plt.clf()

	A = np.array(train_history["val_loss_list"]) 
	A = np.transpose(A)
	for a_i in range(16,31):
		plt.plot(iterationList, A[a_i], color=colors[a_i-16])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Validation Loss')
	plt.title('MT Networks - Unseen Data Set - Validation Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(np.array(train_history["val_loss_list"]))-0.1, np.amax(np.array(train_history["val_loss_list"]))+0.1)
	plt.savefig("MT Networks - Unseen Data Set - Validation Loss.png")
	plt.clf()

	plt.plot(iterationList, A[0])
	plt.xlabel('Num of Iterations')
	plt.ylabel('Total Validation Loss')
	plt.title('MT Networks - Unseen Data Set - Total Validation Loss')
	plt.xlim(min(iterationList), max(iterationList))
	plt.ylim(np.amin(A[0])-1, np.amax(A[0])+1)
	plt.savefig("MT Networks - Unseen Data Set - Total Validation Loss.png")
	plt.clf()

	plt.plot(train_history["thresholds_list"], train_history["thresholds_train_acc_list"])
	plt.xlabel('Thresholds')
	plt.ylabel('Training Accuracy')
	plt.title('MT Networks - Unseen Data Set - Threshold vs Training Acc')
	plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
	plt.ylim(min(train_history["thresholds_train_acc_list"])-5, max(train_history["thresholds_train_acc_list"])+5)
	plt.savefig("MT Networks - Unseen Data Set - Threshold vs Training Acc.png")
	plt.clf()

	plt.plot(train_history["thresholds_list"], train_history["thresholds_train_precision_list"])
	plt.xlabel('Thresholds')
	plt.ylabel('Training Precision')
	plt.title('MT Networks - Unseen Data Set - Threshold vs Training Precision')
	plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
	plt.ylim(min(train_history["thresholds_train_precision_list"])-5, max(train_history["thresholds_train_precision_list"])+5)
	plt.savefig("MT Networks - Unseen Data Set - Threshold vs Training Precision.png")
	plt.clf()

	plt.plot(train_history["thresholds_list"], train_history["thresholds_train_recall_list"])
	plt.xlabel('Thresholds')
	plt.ylabel('Training Recall')
	plt.title('MT Networks - Unseen Data Set - Threshold vs Training Recall')
	plt.xlim(min(train_history["thresholds_list"]), max(train_history["thresholds_list"]))
	plt.ylim(min(train_history["thresholds_train_recall_list"])-5, max(train_history["thresholds_train_recall_list"])+5)
	plt.savefig("MT Networks - Unseen Data Set - Threshold vs Training Recall.png")
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
	


main()