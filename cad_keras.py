import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
import keras.backend as K
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

seed = 7
np.random.seed(seed)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


df = pd.read_excel('CAD.xlsx')
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


def create_baseline():
  model = Sequential()
  model.add(Dense(10, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
  model.add(Dropout(0)) 
  model.add(Dense(15, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
  model.add(Dropout(0)) 
  model.add(Dense(20, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
  model.add(Dropout(0))
  model.add(Dense(15, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
  model.add(Dropout(0)) 
  model.add(Dense(10, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
  model.add(Dropout(0)) 
  model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
  return model


model = KerasClassifier(build_fn=create_baseline, epochs=400, batch_size=8, verbose=0, validation_data=(Xtest, Ytest))
history = model.fit(Xtrain,Ytrain)

# summarize history for Fmeasure
plt.figure(1)
plt.plot(history.history['f1'])
plt.plot(history.history['val_f1'])
plt.title('Model FMeasure')
plt.ylabel('FMeasure')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



pred = model.predict(Xtest)

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
pred_proba = model.predict_proba(Xtest)
fpr, tpr, _ = roc_curve(Ytest, pred_proba[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(3)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of Deep Neural Network')
plt.legend(loc="lower right")
plt.show()

