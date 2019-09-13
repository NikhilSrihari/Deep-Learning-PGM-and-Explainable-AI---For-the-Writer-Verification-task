import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, K2Score
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination, BeliefPropagation


fileloc_features = "./15FeatureDataset/15features.csv"
fileloc_seen_training = "./15FeatureDataset/seen-dataset/seen-dataset/dataset_seen_training_siamese.csv"
fileloc_seen_validation = "./15FeatureDataset/seen-dataset/seen-dataset/dataset_seen_validation_siamese.csv"
fileloc_shuffled_training = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/dataset_seen_training_siamese.csv"
fileloc_shuffled_validation = "./15FeatureDataset/shuffled-dataset/shuffled-dataset/dataset_seen_validation_siamese.csv"
fileloc_unseen_training = "./15FeatureDataset/unseen-dataset/unseen-dataset/dataset_seen_training_siamese.csv"
fileloc_unseen_validation = "./15FeatureDataset/unseen-dataset/unseen-dataset/dataset_seen_validation_siamese.csv"


def main():
	#Fetching features data
	features_data = pd.read_csv(fileloc_features)
	features_data_f = features_data.add_prefix('f')
	features_data_g = features_data.add_prefix('g')
	#Seen Training Data
	seen_traindata = pd.read_csv(fileloc_seen_training, usecols = ['left','right','label'])
	#seen_traindata_f = pd.read_csv(fileloc_seen_training, usecols = ['left','label'])
	#seen_traindata_g = pd.read_csv(fileloc_seen_training, usecols = ['right','label'])
	seen_traindata_merged_f = seen_traindata.merge(features_data_f, left_on = 'left', right_on = 'fimagename')
	seen_traindata_merged_g = seen_traindata.merge(features_data_g, left_on = 'right', right_on = 'gimagename')
	seen_traindata_merged_f = seen_traindata_merged_f.drop(['left', 'right','fimagename','label'], axis = 1)
	seen_traindata_merged_g = seen_traindata_merged_g.drop(['left', 'right','gimagename','label'], axis = 1)
	seen_features_traindata_final = pd.concat([seen_traindata_merged_f, seen_traindata_merged_g], axis = 1)
	seen_label_traindata_final = seen_traindata.loc[:, 'label']
	seen_traindata_final = pd.concat([seen_features_traindata_final, seen_label_traindata_final], axis = 1)
	seen_traindata_final.replace([np.inf, -np.inf], np.nan)
	seen_traindata_final.dropna(inplace=True)  
	seen_traindata_final = seen_traindata_final.astype(int)
	seen_traindata_final_NDArray = seen_traindata_final.values 
	#Seen Validation Data
	seen_validationdata = pd.read_csv(fileloc_seen_validation, usecols = ['left','right','label'])
	#seen_validationdata_f = pd.read_csv(fileloc_seen_validation, usecols = ['left','label'])
	#seen_validationdata_g = pd.read_csv(fileloc_seen_validation, usecols = ['right','label'])
	seen_validationdata_merged_f = seen_validationdata.merge(features_data_f, left_on = 'left', right_on = 'fimagename')
	seen_validationdata_merged_g = seen_validationdata.merge(features_data_g, left_on = 'right', right_on = 'gimagename')
	seen_validationdata_merged_f = seen_validationdata_merged_f.drop(['left', 'right','fimagename','label'], axis = 1)
	seen_validationdata_merged_g = seen_validationdata_merged_g.drop(['left', 'right','gimagename','label'], axis = 1)
	seen_features_validationdata_final = pd.concat([seen_validationdata_merged_f, seen_validationdata_merged_g], axis = 1)
	seen_label_validationdata_final = seen_validationdata.loc[:, 'label']
	seen_validationdata_final = pd.concat([seen_features_validationdata_final, seen_label_validationdata_final], axis = 1)
	seen_validationdata_final.replace([np.inf, -np.inf], np.nan)
	seen_validationdata_final.dropna(inplace=True)
	seen_validationdata_final = seen_validationdata_final.astype(int)
	seen_validationdata_final_NDArray = seen_validationdata_final.values
	#Shuffled Training Data
	shuffled_traindata = pd.read_csv(fileloc_shuffled_training, usecols = ['left','right','label'])
	#shuffled_traindata_f = pd.read_csv(fileloc_shuffled_training, usecols = ['left','label'])
	#shuffled_traindata_g = pd.read_csv(fileloc_shuffled_training, usecols = ['right','label'])
	shuffled_traindata_merged_f = shuffled_traindata.merge(features_data_f, left_on = 'left', right_on = 'fimagename')
	shuffled_traindata_merged_g = shuffled_traindata.merge(features_data_g, left_on = 'right', right_on = 'gimagename')
	shuffled_traindata_merged_f = shuffled_traindata_merged_f.drop(['left', 'right','fimagename','label'], axis = 1)
	shuffled_traindata_merged_g = shuffled_traindata_merged_g.drop(['left', 'right','gimagename','label'], axis = 1)
	shuffled_features_traindata_final = pd.concat([shuffled_traindata_merged_f, shuffled_traindata_merged_g], axis = 1)
	shuffled_label_traindata_final = shuffled_traindata.loc[:, 'label']
	shuffled_traindata_final = pd.concat([shuffled_features_traindata_final, shuffled_label_traindata_final], axis = 1)
	shuffled_traindata_final.replace([np.inf, -np.inf], np.nan)
	shuffled_traindata_final.dropna(inplace=True)
	shuffled_traindata_final = shuffled_traindata_final.astype(int)
	shuffled_traindata_final_NDArray = shuffled_traindata_final.values
	#Shuffled Validation Data
	shuffled_validationdata = pd.read_csv(fileloc_shuffled_validation, usecols = ['left','right','label'])
	#shuffled_validationdata_f = pd.read_csv(fileloc_shuffled_validation, usecols = ['left','label'])
	#shuffled_validationdata_g = pd.read_csv(fileloc_shuffled_validation, usecols = ['right','label'])
	shuffled_validationdata_merged_f = shuffled_validationdata.merge(features_data_f, left_on = 'left', right_on = 'fimagename')
	shuffled_validationdata_merged_g = shuffled_validationdata.merge(features_data_g, left_on = 'right', right_on = 'gimagename')
	shuffled_validationdata_merged_f = shuffled_validationdata_merged_f.drop(['left', 'right','fimagename','label'], axis = 1)
	shuffled_validationdata_merged_g = shuffled_validationdata_merged_g.drop(['left', 'right','gimagename','label'], axis = 1)
	shuffled_features_validationdata_final = pd.concat([shuffled_validationdata_merged_f, shuffled_validationdata_merged_g], axis = 1)
	shuffled_label_validationdata_final = shuffled_validationdata.loc[:, 'label']
	shuffled_validationdata_final = pd.concat([shuffled_features_validationdata_final, shuffled_label_validationdata_final], axis = 1)
	shuffled_validationdata_final.replace([np.inf, -np.inf], np.nan)
	shuffled_validationdata_final.dropna(inplace=True)
	shuffled_validationdata_final = shuffled_validationdata_final.astype(int)
	shuffled_validationdata_final_NDArray = shuffled_validationdata_final.values
	#Unseen Training Data
	unseen_traindata = pd.read_csv(fileloc_unseen_training, usecols = ['left','right','label'])
	#unseen_traindata_f = pd.read_csv(fileloc_unseen_training, usecols = ['left','label'])
	#unseen_traindata_g = pd.read_csv(fileloc_unseen_training, usecols = ['right','label'])
	unseen_traindata_merged_f = unseen_traindata.merge(features_data_f, left_on = 'left', right_on = 'fimagename')
	unseen_traindata_merged_g = unseen_traindata.merge(features_data_g, left_on = 'right', right_on = 'gimagename')
	unseen_traindata_merged_f = unseen_traindata_merged_f.drop(['left', 'right','fimagename','label'], axis = 1)
	unseen_traindata_merged_g = unseen_traindata_merged_g.drop(['left', 'right','gimagename','label'], axis = 1)
	unseen_features_traindata_final = pd.concat([unseen_traindata_merged_f, unseen_traindata_merged_g], axis = 1)
	unseen_label_traindata_final = unseen_traindata.loc[:, 'label']
	unseen_traindata_final = pd.concat([unseen_features_traindata_final, unseen_label_traindata_final], axis = 1)
	unseen_traindata_final.replace([np.inf, -np.inf], np.nan)
	unseen_traindata_final.dropna(inplace=True)
	unseen_traindata_final = unseen_traindata_final.astype(int)
	unseen_traindata_final_NDArray = unseen_traindata_final.values
	#Unseen Validation Data
	unseen_validationdata = pd.read_csv(fileloc_unseen_validation, usecols = ['left','right','label'])
	#unseen_validationdata_f = pd.read_csv(fileloc_unseen_validation, usecols = ['left','label'])
	#unseen_validationdata_g = pd.read_csv(fileloc_unseen_validation, usecols = ['right','label'])
	unseen_validationdata_merged_f = unseen_validationdata.merge(features_data_f, left_on = 'left', right_on = 'fimagename')
	unseen_validationdata_merged_g = unseen_validationdata.merge(features_data_g, left_on = 'right', right_on = 'gimagename')
	unseen_validationdata_merged_f = unseen_validationdata_merged_f.drop(['left', 'right','fimagename','label'], axis = 1)
	unseen_validationdata_merged_g = unseen_validationdata_merged_g.drop(['left', 'right','gimagename','label'], axis = 1)
	unseen_features_validationdata_final = pd.concat([unseen_validationdata_merged_f, unseen_validationdata_merged_g], axis = 1)
	unseen_label_validationdata_final = unseen_validationdata.loc[:, 'label']
	unseen_validationdata_final = pd.concat([unseen_features_validationdata_final, unseen_label_validationdata_final], axis = 1)
	unseen_validationdata_final.replace([np.inf, -np.inf], np.nan)
	unseen_validationdata_final.dropna(inplace=True)
	unseen_validationdata_final = unseen_validationdata_final.astype(int)
	unseen_validationdata_final_NDArray = unseen_validationdata_final.values
	#Creating base models
	featureNamesList = ["pen_pressure","letter_spacing","size","dimension","is_lowercase","is_continuous","slantness","tilt","entry_stroke_a", "staff_of_a","formation_n","staff_of_d","exit_stroke_d","word_formation","constancy"]
	features_only_data = features_data[featureNamesList]
	initial_hcs = HillClimbSearch(features_only_data)
	initial_model = initial_hcs.estimate()
	#print(initial_model.edges())
	print("Hill Climb Done")
	basemodel = BayesianModel([('fpen_pressure', 'fis_lowercase'), ('fpen_pressure', 'fletter_spacing'), ('fsize', 'fslantness'), ('fsize', 'fpen_pressure'), 
								('fsize', 'fstaff_of_d'), ('fsize', 'fletter_spacing'), ('fsize', 'fexit_stroke_d'), ('fsize', 'fentry_stroke_a'), 
								('fdimension', 'fsize'), ('fdimension', 'fis_continuous'), ('fdimension', 'fslantness'), ('fdimension', 'fpen_pressure'), 
								('fis_lowercase', 'fstaff_of_a'), ('fis_lowercase', 'fexit_stroke_d'), ('fis_continuous', 'fexit_stroke_d'), ('fis_continuous', 'fletter_spacing'), 
								('fis_continuous', 'fentry_stroke_a'), ('fis_continuous', 'fstaff_of_a'), ('fis_continuous', 'fis_lowercase'), ('fslantness', 'fis_continuous'), 
								('fslantness', 'ftilt'), ('fentry_stroke_a', 'fpen_pressure'), ('fformation_n', 'fconstancy'), ('fformation_n', 'fword_formation'), ('fformation_n', 'fdimension'), 
								('fformation_n', 'fstaff_of_d'), ('fformation_n', 'fis_continuous'), ('fformation_n', 'fsize'), ('fformation_n', 'fstaff_of_a'), ('fstaff_of_d', 'fis_continuous'), 
								('fstaff_of_d', 'fexit_stroke_d'), ('fstaff_of_d', 'fis_lowercase'), ('fstaff_of_d', 'fslantness'), ('fstaff_of_d', 'fentry_stroke_a'), 
								('fword_formation', 'fdimension'), ('fword_formation', 'fstaff_of_a'), ('fword_formation', 'fsize'), ('fword_formation', 'fstaff_of_d'), 
								('fword_formation', 'fconstancy'), ('fconstancy', 'fstaff_of_a'), ('fconstancy', 'fletter_spacing'), ('fconstancy', 'fdimension'), 
								('gpen_pressure', 'gis_lowercase'), ('gpen_pressure', 'gletter_spacing'), ('gsize', 'gslantness'), ('gsize', 'gpen_pressure'), 
								('gsize', 'gstaff_of_d'), ('gsize', 'gletter_spacing'), ('gsize', 'gexit_stroke_d'), ('gsize', 'gentry_stroke_a'), ('gdimension', 'gsize'), 
								('gdimension', 'gis_continuous'), ('gdimension', 'gslantness'), ('gdimension', 'gpen_pressure'), ('gis_lowercase', 'gstaff_of_a'), 
								('gis_lowercase', 'gexit_stroke_d'), ('gis_continuous', 'gexit_stroke_d'), ('gis_continuous', 'gletter_spacing'), ('gis_continuous', 'gentry_stroke_a'), 
								('gis_continuous', 'gstaff_of_a'), ('gis_continuous', 'gis_lowercase'), ('gslantness', 'gis_continuous'), ('gslantness', 'gtilt'), 
								('gentry_stroke_a', 'gpen_pressure'), ('gformation_n', 'gconstancy'), ('gformation_n', 'gword_formation'), ('gformation_n', 'gdimension'), 
								('gformation_n', 'gstaff_of_d'), ('gformation_n', 'gis_continuous'), ('gformation_n', 'gsize'), ('gformation_n', 'gstaff_of_a'), ('gstaff_of_d', 'gis_continuous'), 
								('gstaff_of_d', 'gexit_stroke_d'), ('gstaff_of_d', 'gis_lowercase'), ('gstaff_of_d', 'gslantness'), ('gstaff_of_d', 'gentry_stroke_a'), 
								('gword_formation', 'gdimension'), ('gword_formation', 'gstaff_of_a'), ('gword_formation', 'gsize'), ('gword_formation', 'gstaff_of_d'), 
								('gword_formation', 'gconstancy'), ('gconstancy', 'gstaff_of_a'), ('gconstancy', 'gletter_spacing'), ('gconstancy', 'gdimension'),
								('fis_continuous', 'label'), ('fword_formation','label'),
								('gis_continuous', 'label'), ('gword_formation','label')])
	model_seen = basemodel.copy()
	model_shuffled = basemodel.copy()
	model_unseen = basemodel.copy()
	accuracies = {}
	#Training Seen Model
	model_seen.fit(seen_traindata_final)
	estimator_seen = BayesianEstimator(model_seen, seen_traindata_final)
	cpds=[]
	for featureName in featureNamesList :
		cpd = estimator_seen.estimate_cpd('f'+featureName)
		cpds.append(cpd)
		cpd = estimator_seen.estimate_cpd('g'+featureName)
		cpds.append(cpd)
	cpd = estimator_seen.estimate_cpd('label')
	cpds.append(cpd)
	model_seen.add_cpds(*cpds)
	print("CPDs Calculated")
	#Testing Seen Model - Training
	model_seen_ve = VariableElimination(model_seen)
	model_seen_traindata_predictions = []
	for i in range(seen_traindata_final_NDArray.shape[0]):
		evidenceDic = {}
		for index, featureName in enumerate(featureNamesList): 
			evidenceDic['f'+featureName]=(seen_traindata_final_NDArray[i,index]-1)
			evidenceDic['g'+featureName]=(seen_traindata_final_NDArray[i+15,index]-1)
		temp = model_seen_ve.map_query(variables=['label'],evidence=evidenceDic)
		model_seen_traindata_predictions.append(temp['label'])
	correctCnt = 0
	for i in range(len(model_seen_traindata_predictions)):
	    if(int(model_seen_traindata_predictions[i]) == int(seen_traindata_final_NDArray[i,30])):
	        correctCnt+=1
	accuracies["seen_train"]=correctCnt/len(model_seen_traindata_predictions)*100
	print("Bayesian Model Accuracy for Seen Training Data = "+str(accuracies["seen_train"]))
	#Testing Seen Model - Validation
	model_seen_ve = VariableElimination(model_seen)
	model_seen_validationdata_predictions = []
	for i in range(seen_validationdata_final_NDArray.shape[0]):
		evidenceDic = {}
		for index, featureName in enumerate(featureNamesList): 
			evidenceDic['f'+featureName]=seen_validationdata_final_NDArray[i,index]-1
			evidenceDic['g'+featureName]=seen_validationdata_final_NDArray[i+15,index]-1
		temp = model_seen_ve.map_query(variables=['label'],evidence=evidenceDic)
		model_seen_validationdata_predictions.append(temp['label'])
	correctCnt = 0
	for i in range(len(model_seen_validationdata_predictions)):
	    if(int(model_seen_validationdata_predictions[i]) == int(seen_validationdata_final_NDArray[i,30])):
	        correctCnt+=1
	accuracies["seen_validation"]=correctCnt/len(model_seen_validationdata_predictions)*100
	print("Bayesian Model Accuracy for Seen Validation Data = "+str(accuracies["seen_validation"]))
	#Training Shuffled Model
	model_shuffled.fit(shuffled_traindata_final)
	estimator_shuffled = BayesianEstimator(model_shuffled, shuffled_traindata_final)
	cpds=[]
	for featureName in featureNamesList :
		cpd = estimator_shuffled.estimate_cpd('f'+featureName)
		cpds.append(cpd)
		cpd = estimator_shuffled.estimate_cpd('g'+featureName)
		cpds.append(cpd)
	cpd = estimator_shuffled.estimate_cpd('label')
	cpds.append(cpd)
	model_shuffled.add_cpds(*cpds)
	#Testing Shuffled Model - Training
	model_shuffled_ve = VariableElimination(model_shuffled)
	model_shuffled_traindata_predictions = []
	for i in range(shuffled_traindata_final_NDArray.shape[0]):
		evidenceDic = {}
		for index, featureName in enumerate(featureNamesList): 
			evidenceDic['f'+featureName]=shuffled_traindata_final_NDArray[i,index]-1
			evidenceDic['g'+featureName]=shuffled_traindata_final_NDArray[i+15,index]-1
		temp = model_shuffled_ve.map_query(variables=['label'],evidence=evidenceDic)
		model_shuffled_traindata_predictions.append(temp['label'])
	correctCnt = 0
	for i in range(len(model_shuffled_traindata_predictions)):
	    if(int(model_shuffled_traindata_predictions[i]) == int(shuffled_traindata_final_NDArray[i,30])):
	        correctCnt+=1
	accuracies["shuffled_train"]=correctCnt/len(model_shuffled_traindata_predictions)*100
	print("Bayesian Model Accuracy for Shuffled Training Data = "+str(accuracies["shuffled_train"]))
	#Testing Shuffled Model - Validation
	model_shuffled_ve = VariableElimination(model_shuffled)
	model_shuffled_validationdata_predictions = []
	for i in range(shuffled_validationdata_final_NDArray.shape[0]):
		evidenceDic = {}
		for index, featureName in enumerate(featureNamesList): 
			evidenceDic['f'+featureName]=shuffled_validationdata_final_NDArray[i,index]-1
			evidenceDic['g'+featureName]=shuffled_validationdata_final_NDArray[i+15,index]-1
	temp = model_shuffled_ve.map_query(variables=['label'],evidence=evidenceDic)
	model_shuffled_validationdata_predictions.append(temp['label'])
	correctCnt = 0
	for i in range(len(model_shuffled_validationdata_predictions)):
	    if(int(model_shuffled_validationdata_predictions[i]) == int(shuffled_validationdata_final_NDArray[i,30])):
	        correctCnt+=1
	accuracies["shuffled_validation"]=correctCnt/len(model_shuffled_validationdata_predictions)*100
	print("Bayesian Model Accuracy for Shuffled Validation Data = "+str(accuracies["shuffled_validation"]))
	#Training Unseen Model
	model_unseen.fit(unseen_traindata_final)
	estimator_unseen = BayesianEstimator(model_unseen, unseen_traindata_final)
	cpds=[]
	for featureName in featureNamesList :
		cpd = estimator_unseen.estimate_cpd('f'+featureName)
		cpds.append(cpd)
		cpd = estimator_unseen.estimate_cpd('g'+featureName)
		cpds.append(cpd)
	cpd = estimator_unseen.estimate_cpd('label')
	cpds.append(cpd)
	model_unseen.add_cpds(*cpds)
	#Testing Unseen Model - Training
	model_unseen_ve = VariableElimination(model_unseen)
	model_unseen_traindata_predictions = []
	for i in range(unseen_traindata_final_NDArray.shape[0]):
		evidenceDic = {}
		for index, featureName in enumerate(featureNamesList): 
			evidenceDic['f'+featureName]=unseen_traindata_final_NDArray[i,index]-1
			evidenceDic['g'+featureName]=unseen_traindata_final_NDArray[i+15,index]-1
		temp = model_unseen_ve.map_query(variables=['label'],evidence=evidenceDic)
		model_unseen_traindata_predictions.append(temp['label'])
	correctCnt = 0
	for i in range(len(model_unseen_traindata_predictions)):
	    if(int(model_unseen_traindata_predictions[i]) == int(unseen_traindata_final_NDArray[i,30])):
	        correctCnt+=1
	accuracies["unseen_train"]=correctCnt/len(model_unseen_traindata_predictions)*100
	print("Bayesian Model Accuracy for Unseen Training Data = "+str(accuracies["unseen_train"]))
	#Testing Unseen Model - Validation
	model_unseen_ve = VariableElimination(model_unseen)
	model_unseen_validationdata_predictions = []
	for i in range(unseen_validationdata_final_NDArray.shape[0]):
		evidenceDic = {}
		for index, featureName in enumerate(featureNamesList): 
			evidenceDic['f'+featureName]=unseen_validationdata_final_NDArray[i,index]-1
			evidenceDic['g'+featureName]=unseen_validationdata_final_NDArray[i+15,index]-1
	temp = model_unseen_ve.map_query(variables=['label'],evidence=evidenceDic)
	model_unseen_validationdata_predictions.append(temp['label'])
	correctCnt = 0
	for i in range(len(model_unseen_validationdata_predictions)):
	    if(int(model_unseen_validationdata_predictions[i]) == int(unseen_validationdata_final_NDArray[i,30])):
	        correctCnt+=1
	accuracies["unseen_validation"]=correctCnt/len(model_unseen_validationdata_predictions)*100
	print("Bayesian Model Accuracy for Unseen Validation Data = "+str(accuracies["unseen_validation"]))


main()