import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy import array
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARMA, ARIMA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ADF检验,检测信号是否平稳
def adf_test(ts):
    adftest = adfuller(ts)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    return adf_res

# 定阶
def get_pdq(ts):
    plot_acf(ts)
    plot_pacf(ts)
    plt.show()

    r,rac,Q = acf(ts, qstat=True)
    prac = pacf(ts,method='ywmle')
    table_data = np.c_[range(1,len(r)), r[1:],rac,prac[1:len(rac)+1],Q]
    table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])

    print(table)

def ts_arma(ts, p, q):
    arma = ARMA(ts, order=(p, q)).fit(disp = -1)
    ts_predict_arma = arma.predict()
    return ts_predict_arma

def ts_diff_rvs(ts):
    return np.cumsum(ts)

data=pd.read_csv('./data/p2_log_2019052505_EMD_DWT.txt',usecols=[0,1],header=None)
data=array(data).astype(np.float)
data =data[~np.isnan(data).any(axis=1)]
# 平稳化操作
data=np.diff(data,axis=0)
data_lat= data[:,0]
data_lat=data_lat.reshape(-1,1)
data_lon= data[:,1]
data_lon=data_lon.reshape(-1,1)
scaler_lat=StandardScaler()
data_lat=scaler_lat.fit_transform(data_lat)
scaler_lon=StandardScaler()
data_lon=scaler_lon.fit_transform(data_lon)
# test_lat = adf_test(data_lat)  # ADF检验
# test_lon=adf_test(data_lon)
# print(test_lat)
# print(test_lon)

#定阶
# get_pdq(data_lat)
# order_lat=sm.tsa.arma_order_select_ic(data_lat,max_ar=30,max_ma=30,ic='aic')['aic_min_order']  # AIC
# order_lon=sm.tsa.arma_order_select_ic(data_lon,max_ar=30,max_ma=30,ic='aic')['aic_min_order']  # AIC
# print(order_lat)
# print(order_lon)

# 预测
lat_arma=ts_arma(data_lat,6,0)
lat_arma=scaler_lat.inverse_transform(lat_arma)
lat_arma=ts_diff_rvs(lat_arma)

lon_arma=ts_arma(data_lon,6,0)
lon_arma=scaler_lon.inverse_transform(lon_arma)
lon_arma=ts_diff_rvs(lon_arma)

plt.figure(1)
plt.plot(lat_arma)
plt.figure(2)
plt.plot(lon_arma)
plt.show()

