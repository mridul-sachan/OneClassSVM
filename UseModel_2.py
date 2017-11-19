# coding: utf-8

import numpy as np  
import pandas as pd  
from sklearn import utils  
import matplotlib


#Read csv file using pandas 
read_data = pd.read_csv('Test_file_without_class.csv', low_memory=False)
print("Data File read successfully.")

from sklearn.externals import joblib  
model = joblib.load('one_class_svm_2.model')
print("Loaded the One Class SVM Model successfully.")

applicable_features = [  
    "duration",
    "src_bytes",
    "dst_bytes" ]
	
read_data = read_data[applicable_features]

# normalise the read_data - which leads to better accuracy and reduces numerical instability.
read_data["duration"] = np.log((read_data["duration"] + 0.1).astype(float))  
read_data["src_bytes"] = np.log((read_data["src_bytes"] + 0.1).astype(float))  
read_data["dst_bytes"] = np.log((read_data["dst_bytes"] + 0.1).astype(float))


# then predict with
values =  (model.predict(read_data))

with open('predicted.txt','w+') as f:
    for value in values:
        f.write(str(value) + '\n')
