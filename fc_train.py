from numpy import array
from keras.models import Model, Input
from keras.layers import Dense, Dropout,Flatten
import pandas as pd
import numpy as np
from keras import backend as K

# generate input and output pairs
# for the input sequence, a line is a record and a column is an attribute
def generate_examples(sequence,input_step):
    n_patterns=len(sequence)-input_step+1
    X, y = list(), list()
    for i in range(n_patterns-1):
        X.append(sequence[i:i+input_step])
        y.append(sequence[i+input_step,0:2])
    X = array(X)
    y = array(y)
    # X = array(X).reshape(n_patterns, input_step, feature_num)
    # y = array(y).reshape(n_patterns, feature_num)
    return X, y

def squeeze_dim_backend(inputs):
    return K.squeeze(inputs,axis=1)

# load training data
data=pd.read_csv('./data/p2_log_2019052505_EMD_DWT.txt',usecols=[0,1,7,8,9,10,11,12])
data=array(data).astype(np.float)
data =data[~np.isnan(data).any(axis=1)]
data_GPS= data[:,0:2]
data_INS= data[:,2:8]
# sequence = np.trunc(data/100) + np.trunc((data/100-np.trunc(data/100))*100)/60 + (data-np.trunc(data))*60/3600

baseline=np.mean(data_GPS,axis=0)
data_GPS=(data_GPS-baseline)*[1000,1000] #针对p2
sequence=np.hstack([data_GPS,data_INS])
input_step = 30
feature_num=sequence.shape[1]

# define model
inputs=Input(shape=(input_step,feature_num))
model=Flatten()(inputs)
model=Dense(64,activation='selu')(model)
model=Dense(64,activation='selu')(model)
model=Dropout(0.2)(model)
model=Dense(2)(model)

model=Model(inputs=inputs,outputs=model)
print(model.summary())
model.compile(loss='mae' , optimizer='adam')

# fit model
X, y = generate_examples(sequence,input_step)
history = model.fit(X, y, batch_size=300, epochs=200)
filename = 'GPS_FC_model_EMDDWT.h5'
model.save(filename)

# evaluate model
X, y = generate_examples(sequence, input_step)
loss = model.evaluate(X, y, verbose=0)
print('MAE: %f' % loss)