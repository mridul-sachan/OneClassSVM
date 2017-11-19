#To use the model on new data (e.g. in JSON format) we could do something like this:

data = pd.read_json(some_json)  
model.predict(data)  

#If our output is -1 the model has predicted the data to be an outlier.
#(which means an attack in our case), a +1 means an inlier (not an attack).

#To use the model, just save it to disk.
outputfile = 'oneclass_1.model'  
from sklearn.externals import joblib  
joblib.dump(model, outputfile, compress=9) 

#Then in our deployed code we and load the model back in with:

from sklearn.externals import joblib  
model = joblib.load('oneclass_v1.model')

# then predict with
model.predict(..)  