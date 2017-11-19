# coding: utf-8

import numpy as np  
import pandas as pd  
from sklearn import utils  
import matplotlib

#Read csv file using pandas 
read_data = pd.read_csv('kddcup_data_10_percent_corrected.csv', low_memory=False)
print("Data File read successfully.")

#Read txt files
#df = pd.read_fwf('input.txt')
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