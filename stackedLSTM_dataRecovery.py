from numpy import array
from keras.models import load_model
from scipy.io import loadmat
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data_GPS=pd.read_csv('./data/data5_EMD.txt',usecols=[0,1])
data_GPS=array(data_GPS).astype(np.float)
data_GPS=data_GPS[~np.isnan(data_GPS).any(axis=1)]
baseline=np.mean(data_GPS,axis=0)

data= pd.read_csv('./data/data5_EMD.txt',usecols=[0,1,7,8,9,10,11,12])
data= array(data).astype(np.float)
data =data[~np.isnan(data).any(axis=1)]
data_GPS= data[:,0:2]
data_INS= data[:,2:8]

data_GPS=(data_GPS-baseline)*[1000,1000] #针对p2
sequence=np.hstack([data_GPS,data_INS])
feature_num=sequence.shape[1]
stackedLSTM = load_model('./GPS_LSTM_model.h5')
input_step = stackedLSTM.input_shape[1]

# prediction on new data
yhat=np.zeros(np.shape(data_GPS))
yhat[0:input_step,0:2]=sequence[0:input_step,0:2]
for i in range(input_step,len(sequence)):
    yhat[i,:]=stackedLSTM.predict(sequence[i-input_step:i,:].reshape(1,input_step,feature_num))
    if np.isnan(sequence[i,:]).any():
        sequence[i,0:2]=yhat[i,0:2]
y_predict=yhat*[1e-3,1e-3]+baseline
y_predict=np.c_[y_predict[:,1],y_predict[:,0],1*np.ones(len(y_predict))]
np.savetxt('LSTM_GPS.txt', y_predict, fmt="%.10f,%.10f,%.2f", delimiter="\n")