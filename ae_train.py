from numpy import array
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from keras import backend as K

INS_data=pd.read_csv('./data/data5.txt',usecols=[7,8,9,10,11,12])
INS_data=array(INS_data).astype(np.float)
INS_data =INS_data[~np.isnan(INS_data).any(axis=1)]
feature_num=INS_data.shape[1]

ae=Sequential()
ae.add(Dense(64,activation='relu',input_shape=(feature_num,)))
ae.add(Dense(32,activation='relu'))
ae.add(Dense(16,activation='relu'))
ae.add(Dense(2,activation='relu'))
ae.add(Dense(16,activation='relu'))
ae.add(Dense(32,activation='relu'))
ae.add(Dense(64,activation='relu'))
ae.add(Dense(feature_num))
print(ae.summary())
ae.compile(optimizer='adam',loss='mean_squared_error')

start=time.perf_counter()
ae.fit(INS_data, INS_data, epochs=50, batch_size=300, shuffle=True, validation_split=0)
interval=(time.perf_counter()-start)
print("Time used: ",interval)

# filename = 'deep_ae.h5'
# ae.save(filename)

encoder_out=K.function([ae.layers[0].input,K.learning_phase()],[ae.layers[4].output])
encoder_output=encoder_out(INS_data)[0]
plt.scatter(encoder_output[:, 0], encoder_output[:, 1])
plt.show()