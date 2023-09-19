import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Input, Dense, LSTM, concatenate, Flatten
from keras.models import Model
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC, Precision, Recall, BinaryAccuracy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time
import random
from keras.callbacks import EarlyStopping

accident_5min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=5 1to5 unbalanced/accident_5_min_before.csv", encoding='cp949', index_col=0)
accident_10min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=5 1to5 unbalanced/accident_10_min_before.csv", encoding='cp949', index_col=0)
accident_15min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=5 1to5 unbalanced/accident_15_min_before.csv", encoding='cp949', index_col=0)
accident_20min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=5 1to5 unbalanced/accident_20_min_before.csv", encoding='cp949', index_col=0)

non_accident_5min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=5 1to5 unbalanced/non_accident_5_min_before.csv", encoding='cp949', index_col=0)
non_accident_10min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=5 1to5 unbalanced/non_accident_10_min_before.csv", encoding='cp949', index_col=0)
non_accident_15min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=5 1to5 unbalanced/non_accident_15_min_before.csv", encoding='cp949', index_col=0)
non_accident_20min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=5 1to5 unbalanced/non_accident_20_min_before.csv", encoding='cp949', index_col=0)

# Each accident, non_accident data (to numpy shape)
accident_5min = accident_5min_before.drop(['datetime', 'link', 'label'], axis=1).to_numpy()
accident_10min = accident_10min_before.drop(['datetime', 'link', 'label'], axis=1).to_numpy()
accident_15min = accident_15min_before.drop(['datetime', 'link', 'label'], axis=1).to_numpy()
accident_20min = accident_20min_before.drop(['datetime', 'link', 'label'], axis=1).to_numpy()

non_accident_5min = non_accident_5min_before.drop(['datetime', 'link', 'label'], axis=1).to_numpy()
non_accident_10min = non_accident_10min_before.drop(['datetime', 'link', 'label'], axis=1).to_numpy()
non_accident_15min = non_accident_15min_before.drop(['datetime', 'link', 'label'], axis=1).to_numpy()
non_accident_20min = non_accident_20min_before.drop(['datetime', 'link', 'label'], axis=1).to_numpy()

accident_label = accident_5min_before['label'].to_numpy()
non_accident_label = non_accident_5min_before['label'].to_numpy()

# Construct dataset - train, validation, test
# For LSTM model, data sequence is 4 (20, 15, 10, 5 mintues before)
accident_X, accident_Y = [], []
for accident in range(len(accident_5min)):
    accident_X.append([accident_20min[accident], accident_15min[accident], accident_10min[accident], accident_5min[accident]])
    accident_Y.append(accident_label[accident])

non_accident_X, non_accident_Y = [], []
for non_accident in range(len(non_accident_5min)):
    non_accident_X.append([non_accident_20min[non_accident], non_accident_15min[non_accident], non_accident_10min[non_accident], non_accident_5min[non_accident]])
    non_accident_Y.append(non_accident_label[non_accident])

dataset_size = int(len(non_accident_X)) # 데이터셋 개수
data_indexes = np.arange(dataset_size)
np.random.shuffle(data_indexes)  # index 섞기
data_indexes = data_indexes[:len(accident_X) * 2]

non_accident_X = [non_accident_X[i] for i in data_indexes]
non_accident_Y = [non_accident_Y[i] for i in data_indexes]

pos_X_train, pos_X_test, pos_Y_train, pos_Y_test = train_test_split(accident_X, accident_Y, test_size=0.2, random_state=1234)
pos_X_val, pos_X_test, pos_Y_val, pos_Y_test = train_test_split(pos_X_test, pos_Y_test, test_size=0.2, random_state=1234)

neg_X_train, neg_X_test, neg_Y_train, neg_Y_test = train_test_split(non_accident_X, non_accident_Y, test_size=0.2, random_state=1234)
neg_X_val, neg_X_test, neg_Y_val, neg_Y_test = train_test_split(neg_X_test, neg_Y_test, test_size=0.2, random_state=1234)

# --------------------------Total Dataset construction--------------------------#

random.seed(42)

X_train = np.concatenate((pos_X_train, neg_X_train), axis=0)
#random.shuffle(X_train)
X_val = np.concatenate((pos_X_val, neg_X_val), axis=0)
#random.shuffle(X_val)
X_test = np.concatenate((pos_X_test, neg_X_test), axis=0)
#random.shuffle(X_test)

Y_train = np.concatenate((pos_Y_train, neg_Y_train), axis=0)
#random.shuffle(Y_train)
Y_val = np.concatenate((pos_Y_val, neg_Y_val), axis=0)
#random.shuffle(Y_val)
Y_test = np.concatenate((pos_Y_test, neg_Y_test), axis=0)
#random.shuffle(Y_test)

# ---------------------------Ratio Check point------------------------------#

# Train #
Train_unique, Train_counts = np.unique(Y_train, return_counts = True)
Train_uniq_cnt_dict = dict(zip(Train_unique, Train_counts))
print(Train_uniq_cnt_dict)

# Validation #
Val_unique, Val_counts = np.unique(Y_val, return_counts = True)
Val_uniq_cnt_dict = dict(zip(Val_unique, Val_counts))
print(Val_uniq_cnt_dict)

# Test #
Test_unique, Test_counts = np.unique(Y_test, return_counts = True)
Test_uniq_cnt_dict = dict(zip(Test_unique, Test_counts))
print(Test_uniq_cnt_dict)

# Pause - Data Check point #
#print(len(static_accident_X[0]))
#print(Y_train[0:100])
time.sleep(20)

# =================================================================================================#
# =================================== Model Part ===================================================#
# =================================================================================================#

# MLP model (for proposed our model)
input_data = Input(shape=(4, 16))

mlp_1 = Dense(units=100, activation='relu')(input_data)
mlp_2 = Dense(units=100, activation='relu')(mlp_1)
x = Flatten()(mlp_2)
result = Dense(units=1, activation='sigmoid')(x)

LSTM_model = Model(inputs=input_data, outputs=result)
LSTM_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
LSTM_model.summary()

early_stopping = EarlyStopping(monitor = 'val_auc', min_delta = 0, patience = 20, mode = 'max')

hist = LSTM_model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_data=(X_val, Y_val), callbacks = [early_stopping], verbose=1)

loss, acc, auc, precision, recall, binary_accuracy = LSTM_model.evaluate(X_test, Y_test, verbose=1)
print("m parameter", LSTM_model.count_params())
print('loss : ', loss)
print('acc : ', acc)
print('auc : ', auc)
print('precision : ', precision)
print('recall : ', recall)
print('binary_accuracy : ', binary_accuracy)
#print f1-score with calculate recall precision

f1_score = (2 * precision * recall) / (precision + recall)
print('f1_score : ', f1_score)

plt.plot(hist.history['auc'])
plt.plot(hist.history['val_auc'])
plt.title('model auc')

plt.ylabel('auc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()