#get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np  
import pandas as pd  
from sklearn import utils  
import matplotlib

#Read csv file
data = pd.read_csv('kddcup_data_10_percent_corrected.csv', low_memory=False)

data = data[data['service'] == "http"]  
data = data[data["logged_in"] == 1]

# let's take a look at the types of attack labels are present in the data.
data.label.value_counts().plot(kind='bar')

