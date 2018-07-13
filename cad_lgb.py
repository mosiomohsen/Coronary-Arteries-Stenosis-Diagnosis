import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

seed = 7
np.random.seed(seed)


df = pd.read_excel('C:/Users/Internet/Downloads/CAD.xlsx')
df.head()

X = df.drop(['Cath'], axis=1)
Y = df['Cath']
X = pd.get_dummies(X)
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=7)
sm = SMOTE(random_state=7, ratio = 1.0)
Xtrain, Ytrain = sm.fit_sample(Xtrain, Ytrain)



lgb_train = lgb.Dataset(Xtrain, Ytrain)
lgb_eval = lgb.Dataset(Xtest, Ytest, reference=lgb_train)

gbm = lgb.LGBMClassifier(n_estimators=10000, silent=True, subsample=0.85, colsample_bytree=0.85, learning_rate=0.02)
gbm.fit(Xtrain, Ytrain, eval_set=[(Xtest, Ytest)], early_stopping_rounds=50, verbose=False)

eval_res = gbm.evals_result_
loss = eval_res['valid_0']['binary_logloss']
# summarize history for loss
plt.figure(2)
plt.plot(loss)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('Start predicting...')
# predict
pred = gbm.predict(Xtest, num_iteration=gbm.best_iteration_)

acc = accuracy_score(pred, Ytest)
fmeasure = f1_score(pred, Ytest)
recall = recall_score(pred, Ytest)
precision = precision_score(pred, Ytest)
conf = confusion_matrix(pred, Ytest)

print(acc)
print(fmeasure)
print(recall)
print(precision)
print(conf)


###  predict_probability and drawing roc curve  ###
pred_proba = gbm.predict_proba(Xtest)
fpr, tpr, _ = roc_curve(Ytest, pred_proba[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of LightGBM')
plt.legend(loc="lower right")
plt.show()

