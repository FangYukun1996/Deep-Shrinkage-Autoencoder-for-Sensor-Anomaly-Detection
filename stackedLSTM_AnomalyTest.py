from numpy import array
from keras.models import load_model
from scipy.io import loadmat
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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
faultData=loadmat('./data/block.mat')
test_input=faultData['BlockAnomalyTestData']
FLAG=faultData['FLAG']
test_input_length=test_input.shape[1]
index = np.random.randint(0,test_input_length-1,size=1)
index=int(index)
flag= int(FLAG[index])

# 得到训练时的baseline
data_GPS=pd.read_csv('./data/p2_log_2019052505_EMD_DWT.txt',usecols=[0,1])
data_GPS=array(data_GPS).astype(np.float)
data_GPS=data_GPS[~np.isnan(data_GPS).any(axis=1)]
baseline=np.mean(data_GPS,axis=0)

# 载入测试数据进行测试
data=pd.read_csv('./data/p2_log_20190524_outlierExc_winLS.txt',header=None)
data= array(data).astype(np.float)
data =data[~np.isnan(data).any(axis=1)]
data_GPS= data[:,0:2]
data_GPS[:,flag-1]=test_input[:,index]
data_INS= data[:,2:8]

data_GPS=(data_GPS-baseline)*[1000,1000] #针对p2
sequence=np.hstack([data_GPS,data_INS])
feature_num=sequence.shape[1]
stackedLSTM = load_model('./model/GPS_LSTM_model_EMDDWT_opt.h5')
input_step = stackedLSTM.input_shape[1]

X, y = generate_examples(sequence,input_step)
yhat = stackedLSTM.predict(X, verbose=0)
y_predict=yhat*[1e-3,1e-3]+baseline
err=np.sqrt(((y_predict[:,flag-1]-test_input[input_step:,index])**2))

# 计算每个点的决策函数
max_err_t_data=pd.read_table('./data/max_err_t.txt',header=None)
max_err_t=array(max_err_t_data)
err_max=max_err_t[flag-1]
beta=1/(1+np.exp(err))+1
delta=beta*err_max
phi=np.sign(delta-err)

# 作图
data=pd.read_csv('./data/p2_log_20190524_outlierExc_winLS.txt',header=None)
data= array(data).astype(np.float)
data =data[~np.isnan(data).any(axis=1)]

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(data[:,flag-1])
plt.plot(test_input[:,index])
plt.plot(y_predict[:,flag-1])
plt.ylabel("Latitude or Longitude",fontsize=12)
plt.legend(["Original Data","Injected Anomaly","Predicted Values"],loc="upper left")
plt.subplot(3,1,2)
plt.plot(err)
plt.ylabel("err",fontsize=12)
plt.subplot(3,1,3)
plt.plot(phi)
plt.xlabel("Time Step",fontsize=12)
plt.ylabel("Decision Function",fontsize=12)
plt.savefig('blockAnomaly_err.png',dpi=600)
plt.show()