from numpy import array
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_examples(sequence,input_step):
    n_patterns=len(sequence)-input_step+1
    X, y = list(), list()
    for i in range(n_patterns-1):
        X.append(sequence[i:i+input_step,:])
        y.append(sequence[i+input_step,0:2])
    X = array(X)
    y = array(y)

    return X, y

# load test data and the trained model
data_GPS=pd.read_csv('./data/p2_log_2019052505_EMD_DWT.txt',usecols=[0,1])
data_GPS=array(data_GPS).astype(np.float)
data_GPS=data_GPS[~np.isnan(data_GPS).any(axis=1)]
baseline=np.mean(data_GPS,axis=0)

data= pd.read_csv('./data/p2_log_20190524_outlierExc_winLS.txt')
data= array(data).astype(np.float)
data =data[~np.isnan(data).any(axis=1)]
data_GPS= data[:,0:2]
data_INS= data[:,2:8]

data_GPS=(data_GPS-baseline)*[1000,1000]
sequence=np.hstack([data_GPS,data_INS])
feature_num=sequence.shape[1]
stackedLSTM = load_model('./model/GPS_FC_model_EMDDWT.h5')
input_step = stackedLSTM.input_shape[1]

X, y = generate_examples(sequence,input_step)
yhat = stackedLSTM.predict(X, verbose=0)
y_predict=yhat*[1e-3,1e-3]+baseline

# plt.plot(data[:,1])
# plt.plot(y_predict[:,1])
# plt.show()

err=np.sqrt(((y_predict-data[input_step:,0:2])**2))
max_err_t=err.max(0) #0表示所有列中的最小值
np.savetxt('fc_max_err_t.txt',max_err_t,fmt="%.16f",delimiter=",")