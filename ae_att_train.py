from numpy import array
from keras.models import Model,Input
from keras.layers import Dense,GlobalAveragePooling1D,BatchNormalization,Activation,subtract,maximum,multiply
from keras.layers.core import Lambda
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
from keras import backend as K
import matplotlib.pyplot as plt

def abs_backend(inputs):
    return K.abs(inputs)

def expand_dim_backend(inputs):
    # return K.expand_dims(K.expand_dims(inputs, 1), 1)
    return K.expand_dims(inputs,axis=1)

def squeeze_dim_backend(inputs):
    return K.squeeze(inputs,axis=1)

def sign_backend(inputs):
    return K.sign(inputs)

def shrinkage_block(net,out_channels):
    residual=net

    # Calculate global means
    residual_abs = Lambda(abs_backend)(residual)
    residual_abs = Lambda(expand_dim_backend)(residual_abs)
    abs_mean = GlobalAveragePooling1D(data_format='channels_last')(residual_abs)

    # Calculate scaling coefficients
    # scales = Dense(out_channels, activation=None)(abs_mean)
    scales = BatchNormalization()(abs_mean)
    # scales = Activation('relu')(scales)
    scales = Dense(out_channels, activation='sigmoid')(scales)

    # Calculate thresholds
    thres = multiply([abs_mean, scales])
    # Soft thresholding
    sub = subtract([residual_abs, thres])
    zeros = subtract([sub, sub])
    n_sub = maximum([sub, zeros])
    residual = multiply([Lambda(sign_backend)(residual), n_sub])
    residual=Lambda(squeeze_dim_backend)(residual)

    return residual

INS_data=pd.read_csv('./data/data5_EMD_DWT.txt',usecols=[7,8,9,10,11,12])
INS_data=array(INS_data).astype(np.float)
INS_data =INS_data[~np.isnan(INS_data).any(axis=1)]
scaler=StandardScaler()
INS_data=scaler.fit_transform(INS_data)
feature_num=INS_data.shape[1]
input_shape = (feature_num,)

inputs=Input(shape=input_shape)
# encoder
encoder=Dense(64,activation='selu')(inputs)
encoder=Dense(32,activation='selu')(encoder)
encoder=Dense(16,activation='selu')(encoder)
encoder=Dense(2,activation='selu')(encoder)

# # soft shrinkage
shrinkage=shrinkage_block(encoder,2)

# decoder
decoder=Dense(16,activation='selu')(shrinkage)
decoder=Dense(32,activation='selu')(decoder)
decoder=Dense(64,activation='selu')(decoder)
decoder=Dense(feature_num,activation='selu')(decoder)

ae=Model(inputs=inputs,outputs=decoder)
print(ae.summary())
ae.compile(optimizer='adam',loss='mean_squared_error')
start=time.perf_counter()
ae.fit(INS_data, INS_data, epochs=300, batch_size=200, shuffle=True, validation_split=0)
interval=(time.perf_counter()-start)
print("Time used: ",interval)
filename = 'att_deep_ae.h5'
ae.save(filename)

encoder_out=K.function([ae.layers[0].input,K.learning_phase()],[ae.layers[4].output])
encoder_output=encoder_out(INS_data)[0]
encoder_th_out=K.function([ae.layers[0].input,K.learning_phase()],[ae.layers[16].output])
encoder_th_output=encoder_th_out(INS_data)[0]
plt.scatter(encoder_output[:, 0], encoder_output[:, 1],c='blue')
plt.scatter(encoder_th_output[:, 0], encoder_th_output[:, 1],c='red')
plt.show()