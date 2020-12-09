from numpy import array
from keras.models import load_model
from scipy.io import loadmat
import pandas as pd
import numpy as np
import time

# generate input and output pairs
# for the input sequence, a line is a record and a column is an attribute
def generate_examples(sequence,input_step):
    n_patterns=len(sequence)-input_step+1
    X, y = list(), list()
    for i in range(n_patterns-1):
        X.append(sequence[i:i+input_step,:])
        y.append(sequence[i+input_step,0:2])
    X = array(X)
    y = array(y)

    return X, y

# 加载Block Anomaly
faultData=loadmat('./data/BlockTest.mat')
test_input=faultData['TEST_DATA_ALL']
FLAG=faultData['FLAG_ALL']
FLAG=np.transpose(FLAG)
LABEL=faultData['LABEL_ALL']
LABEL=np.transpose(LABEL)
test_input_length=test_input.shape[1]

# 得到训练时的baseline
data_GPS=pd.read_csv('./data/p2_log_2019052505_EMD_DWT.txt',usecols=[0,1])
data_GPS=array(data_GPS).astype(np.float)
data_GPS=data_GPS[~np.isnan(data_GPS).any(axis=1)]
baseline=np.mean(data_GPS,axis=0)

#载入测试数据
data= pd.read_csv('./data/p2_log_20190524_outlierExc_winLS.txt',header=None)
data= array(data).astype(np.float)
data =data[~np.isnan(data).any(axis=1)]
data_GPS= data[:,0:2]
data_INS= data[:,2:8]

data_GPS = (data_GPS - baseline) * [1000, 1000]
sequence = np.hstack([data_GPS, data_INS])
sample_num = sequence.shape[0]
feature_num = sequence.shape[1]
stackedLSTM = load_model('./model/GPS_LSTM_model_EMDDWT_opt.h5')
input_step = stackedLSTM.input_shape[1]

decision_fc=np.zeros(test_input_length)
factors=0.1
stackedLSTM = load_model('./model/GPS_FC_model_EMDDWT.h5')
max_err_t_data = pd.read_table('./data/fc_max_err_t.txt', header=None)
max_err_t = array(max_err_t_data)

start=time.perf_counter()
for i in range (test_input_length):
    data_GPS = data[:, 0:2] #重新读取
    flag=int(FLAG[i])
    data_GPS[:, flag - 1] = test_input[:, i]
    data_GPS=(data_GPS-baseline)*[1000,1000] #针对p2
    sequence=np.hstack([data_GPS,data_INS])
    feature_num=sequence.shape[1]

    input_step = stackedLSTM.input_shape[1]

    X, y = generate_examples(sequence,input_step)
    yhat = stackedLSTM.predict(X, verbose=0)
    y_predict=yhat*[1e-3,1e-3]+baseline

    err=np.sqrt(((y_predict[:,flag-1] - test_input[input_step:,i]) ** 2)) #2-norm error

    # 计算每个点的决策函数

    err_max = max_err_t[flag - 1]
    beta = factors / (1 + np.exp(err)) + 0.9575
    delta = beta * err_max
    phi = np.sign(delta - err)
    if phi.sum() < sample_num-input_step:
        decision_fc[i]=1
interval=(time.perf_counter()-start)
print(interval)
np.savetxt('decision_fc.txt', decision_fc,fmt="%.0f", delimiter=",")