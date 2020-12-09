import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.io import loadmat
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

test=loadmat('./data/abrupt.mat')
test_input=test['X_abrupt']
scaler=StandardScaler()
test_input=scaler.fit_transform(test_input)
test_label=test['X_abrupt_label']

ae=load_model('./att_deep_ae2.h5')
test_result=ae.predict(test_input)
error=np.sqrt(((test_input - test_result) ** 2))#误差平方
abnormal_score= error.sum(1) #以误差平方和作为异常的度量

fpr, tpr, th = roc_curve(test_label, abnormal_score)
ROC_AUC_ae = 1-roc_auc_score(test_label, abnormal_score)
precision,recall,_=precision_recall_curve(test_label,abnormal_score)
f1_ae=2*(np.mean(precision)*np.mean(recall))/(np.mean(precision)+np.mean(recall))

print('ROC_AUC:')
print(ROC_AUC_ae)
print('f1_score:')
print(f1_ae)


plt.figure(1)
plt.plot(tpr,fpr,c='blue') #s控制点的大小
plt.title("ROC curve",fontsize=18)
plt.xlabel("False Positive Rate",fontsize=16)
plt.ylabel("True Positive Rate",fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.savefig('ROC.png',dpi=600)
plt.show()