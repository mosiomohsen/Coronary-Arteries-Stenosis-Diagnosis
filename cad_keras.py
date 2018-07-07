import pandas as pd
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
import keras.backend as K
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

seed = 7
np.random.seed(seed)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
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
model.fit(Xtrain,Ytrain)
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

