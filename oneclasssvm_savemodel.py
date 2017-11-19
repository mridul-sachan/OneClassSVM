# coding: utf-8

import numpy as np  
import pandas as pd  
from sklearn import utils  
import matplotlib

#Read csv file using pandas 
read_data = pd.read_csv('kddcup_data_10_percent_corrected.csv', low_memory=False)
print("File read successfully.")

read_data = read_data[read_data['service'] == "http"]
read_data = read_data[read_data["logged_in"] == 1]

#Out of the 41 available features in the kdd data set, here I am taking only 3 relevant features.
print("Extracting the relevant features from the file.")

applicable_features = [  
    "duration",
    "src_bytes",
    "dst_bytes",
    "class_label" ]

# replace the read_data with a subset containing only the applicable features to train the model
read_data = read_data[applicable_features]

# normalise the read_data - which leads to better accuracy and reduces numerical instability.
read_data["duration"] = np.log((read_data["duration"] + 0.1).astype(float))  
read_data["src_bytes"] = np.log((read_data["src_bytes"] + 0.1).astype(float))  
read_data["dst_bytes"] = np.log((read_data["dst_bytes"] + 0.1).astype(float))  

#Here in this dataset, we have multiple classes as column contains different categories of attacks but I'm using 1-class svm,
# So, to make use of this read_data in a one-class system we need to convert the attacks into
# class 1 (normal) and class -1 (attacker)
read_data.loc[read_data['class_label'] == "normal.", "traffic_behaviour"] = 1  
read_data.loc[read_data['class_label'] != "normal.", "traffic_behaviour"] = -1

# Finding traffic_behaviour value as the target for training and testing.
target = read_data['traffic_behaviour']

# find the proportion of outliers we expect (aka where `traffic_behaviour == -1`). because 
# target is a series, we just compare against itself rather than a column.
outliers = target[target == -1]  
print("outliers.shape", outliers.shape)  
print("outlier fraction", outliers.shape[0]/target.shape[0])

read_data.drop(["class_label", "traffic_behaviour"], axis=1, inplace=True)

read_data.shape  


from sklearn.model_selection import train_test_split  
train_data, test_data, train_target, test_target = train_test_split(read_data, target, train_size = 0.8)  
train_data.shape  


from sklearn import svm

# set nu (which should be the proportion of outliers in our dataset)
nu = outliers.shape[0] / target.shape[0]  
print("The calculated values of nu is:", nu)

model = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=0.00005)  
model.fit(train_data)  


from sklearn import metrics  
values_preds = model.predict(train_data)  
values_targs = train_target

print("Training DataSET accuracy: ", 100 *  metrics.accuracy_score(values_targs, values_preds))
print("Training DataSET Precision: ",100 * metrics.precision_score(values_targs, values_preds))
print("Training DataSET Recall: ", 100 * metrics.recall_score(values_targs, values_preds))
print("Training DataSET f1: ", 100 * metrics.f1_score(values_targs, values_preds))

values_preds = model.predict(test_data)
values_targs = test_target

print("Test DataSet Accuracy: ", 100 * metrics.accuracy_score(values_targs, values_preds))
print("Test DataSet Precision: ", 100 * metrics.precision_score(values_targs, values_preds))
print("Test DataSet Recall: ", 100 * metrics.recall_score(values_targs, values_preds))
print("Test DataSet F1: ", 100 * metrics.f1_score(values_targs, values_preds))

outputfile = 'one_class_svm_2.model'
from sklearn.externals import joblib  
joblib.dump(model, outputfile, compress=9) 

