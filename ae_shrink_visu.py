import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras import backend as K

INS_data=pd.read_csv('./data/data5_EMD_DWT.txt',usecols=[7,8,9,10,11,12])
INS_data=array(INS_data).astype(np.float)
INS_data =INS_data[~np.isnan(INS_data).any(axis=1)]
scaler=StandardScaler()
INS_data=scaler.fit_transform(INS_data)

# 加载模型
ae=load_model('./model/att_deep_ae2.h5')
print(ae.summary())

# shrink 前后对比
encoder_out=K.function([ae.layers[0].input,K.learning_phase()],[ae.layers[4].output])
encoder_output=encoder_out(INS_data)[0]
encoder_th_out=K.function([ae.layers[0].input,K.learning_phase()],[ae.layers[16].output])
encoder_th_output=encoder_th_out(INS_data)[0]
plt.scatter(encoder_output[:, 0], encoder_output[:, 1],c='blue')
plt.scatter(encoder_th_output[:, 0], encoder_th_output[:, 1],c='red')
# plt.title("High dimension features visualization",fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(["Before Shrinkage","After Shrinkage"],loc="upper right",fontsize='x-large')
plt.savefig('Shrinkage.png',dpi=600)
plt.show()