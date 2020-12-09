from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
import numpy as np
from keras import regularizers
# generate input and output pairs
# for the input sequence, a line is a record and a column is an attribute
def generate_examples(sequence,input_step):
    n_patterns=len(sequence)-input_step+1
    feature_num=sequence.shape[1]
    X, y = list(), list()
    for i in range(n_patterns-1):
        X.append(sequence[i:i+input_step])
        y.append(sequence[i+input_step,0:2])
    X = array(X)
    y = array(y)
    # X = array(X).reshape(n_patterns, input_step, feature_num)
    # y = array(y).reshape(n_patterns, feature_num)
    return X, y

# load training data
# data = pd.read_csv('./data/data5.txt',usecols=[0,1])  # 纬度，经度数据
# data= pd.read_csv('./data/p2_log_2019052505lonlat.txt',usecols=[0,1])
data=pd.read_csv('./data/p2_log_2019052505_EMD_DWT.txt',usecols=[0,1,7,8,9,10,11,12])
data=array(data).astype(np.float)
data =data[~np.isnan(data).any(axis=1)]
data_GPS= data[:,0:2]
data_INS= data[:,2:8]
# sequence = np.trunc(data/100) + np.trunc((data/100-np.trunc(data/100))*100)/60 + (data-np.trunc(data))*60/3600

baseline=np.mean(data_GPS,axis=0)
# sequence=(sequence-baseline)*[10000,10000] #针对p3
data_GPS=(data_GPS-baseline)*[1000,1000] #针对p2
sequence=np.hstack([data_GPS,data_INS])
input_step = 30
feature_num=sequence.shape[1]

# define model
model = Sequential()
model.add(LSTM(16, return_sequences=True, input_shape=(input_step, feature_num)))
model.add(LSTM(16,return_sequences=False,dropout=0.2))
# model.add(Dense(feature_num)) #输出特征数
model.add(Dense(2))
print(model.summary())
model.compile(loss='mae' , optimizer='adam')

# fit model
X, y = generate_examples(sequence,input_step)
history = model.fit(X, y, batch_size=300, epochs=200)
filename = 'GPS_LSTM_model_EMDDWT.h5'
model.save(filename)

# evaluate model
X, y = generate_examples(sequence, input_step)
loss = model.evaluate(X, y, verbose=0)
print('MAE: %f' % loss)