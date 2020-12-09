import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from keras.models import load_model

#训练数据。均为搜集的健康的数据
INS_data=pd.read_csv('./data/data5_EMD_DWT.txt',usecols=[7,8,9,10,11,12])
INS_data=np.array(INS_data).astype(np.float)
X_train =INS_data[~np.isnan(INS_data).any(axis=1)]

#人工标记好的有错误的数据
faultData=loadmat('./data/abrupt.mat')
test_input=faultData['X_abrupt']
scaler=StandardScaler()
test_input=scaler.fit_transform(test_input)
test_label=faultData['X_abrupt_label']

# fit the model
clf_ocsvm = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
clf_ocsvm.fit(X_train)

clf_isoForest = IsolationForest(contamination=0.01, behaviour='new')
clf_isoForest.fit(X_train)

clf_lof = LocalOutlierFactor(n_neighbors=20,contamination=0.1,novelty=True)
clf_lof.fit(X_train)

clf_ae=load_model('./model/deep_ae4.h5')

clf_ae_shrink=load_model('./model/att_deep_ae2.h5')

# predict
y_pred_abrupt_ocsvm = clf_ocsvm.predict(test_input)#分类器的预测值
Z_ocsvm = clf_ocsvm.decision_function(test_input) #分类器给出的线性分类分值

y_pred_abrupt_isoForest=clf_isoForest.predict(test_input)
Z_isoForest=clf_isoForest.decision_function(test_input)

y_pred_abrupt_lof=clf_lof.predict(test_input)
Z_lof=clf_lof.decision_function(test_input)

y_pred_abrupt_ae=clf_ae.predict(test_input)
error=np.sqrt(((y_pred_abrupt_ae - test_input) ** 2))#误差平方
abnormal_score= error.sum(1) #以误差平方和作为异常的度量

y_pred_abrupt_ae_shrink=clf_ae_shrink.predict(test_input)
error_shrink=np.sqrt(((y_pred_abrupt_ae_shrink - test_input) ** 2))#误差平方
abnormal_score_shrink= error_shrink.sum(1) #以误差平方和作为异常的度量

# 性能度量
fpr1,tpr1,th1=roc_curve(test_label,Z_ocsvm)
ROC_AUC_ocsvm=roc_auc_score(test_label,Z_ocsvm)
precision_ocsvm,recall_ocsvm,_=precision_recall_curve(test_label,Z_ocsvm)
f1_ocsvm=2*(np.mean(precision_ocsvm)*np.mean(recall_ocsvm))/(np.mean(precision_ocsvm)+np.mean(recall_ocsvm))

fpr2,tpr2,th2=roc_curve(test_label,Z_isoForest)
ROC_AUC_isoForest=roc_auc_score(test_label,Z_isoForest)
precision_isoForest,recall_isoForest,_=precision_recall_curve(test_label,Z_isoForest)
f1_isoForest=2*(np.mean(precision_isoForest)*np.mean(recall_isoForest))/(np.mean(precision_isoForest)+np.mean(recall_isoForest))

fpr3,tpr3,th3=roc_curve(test_label,Z_lof)
ROC_AUC_lof=roc_auc_score(test_label,Z_lof)
precision_lof,recall_lof,_=precision_recall_curve(test_label,Z_lof)
f1_lof=2*(np.mean(precision_lof)*np.mean(recall_lof))/(np.mean(precision_lof)+np.mean(recall_lof))

fpr4, tpr4, th4 = roc_curve(test_label, abnormal_score)
ROC_AUC_ae = 1-roc_auc_score(test_label, abnormal_score)
precision_ae,recall_ae,_=precision_recall_curve(test_label,abnormal_score)
f1_ae=2*(np.mean(precision_ae)*np.mean(recall_ae))/(np.mean(precision_ae)+np.mean(recall_ae))

fpr5, tpr5, th5 = roc_curve(test_label, abnormal_score_shrink)
ROC_AUC_ae_shrink = 1-roc_auc_score(test_label, abnormal_score_shrink)
precision_ae_shrink,recall_ae_shrink,_=precision_recall_curve(test_label,abnormal_score_shrink)
f1_ae_shrink=2*(np.mean(precision_ae_shrink)*np.mean(recall_ae_shrink))/(np.mean(precision_ae_shrink)+np.mean(recall_ae_shrink))

print('ROC_AUC:')
print(ROC_AUC_ocsvm)
print(ROC_AUC_isoForest)
print(ROC_AUC_lof)
print(ROC_AUC_ae)
print(ROC_AUC_ae_shrink)
print('f1_score:')
print(f1_ocsvm)
print(f1_isoForest)
print(f1_lof)
print(f1_ae)
print(f1_ae_shrink)

lg1=plt.plot(fpr1,tpr1)
lg2=plt.plot(fpr2,tpr2,c='red')
lg3=plt.plot(fpr3,tpr3,c='gold')
lg4=plt.plot(tpr4,fpr4,c='green')
lg5=plt.plot(tpr5,fpr5,c='purple')

plt.title("ROC curve",fontsize=18)
plt.xlabel("False Positive Rate",fontsize=16)
plt.ylabel("True Positive Rate",fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(["OCSVM","iForest","LOF","DAE","DSAE"],loc="best",fontsize='x-large')
plt.savefig('ROC.png',dpi=600)
plt.show()


