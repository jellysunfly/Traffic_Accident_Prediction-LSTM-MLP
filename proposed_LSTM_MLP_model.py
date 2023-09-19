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

accident_5min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=10 1to5 unbalanced/accident_10_min_before.csv", encoding='cp949', index_col=0)
accident_10min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=10 1to5 unbalanced/accident_20_min_before.csv", encoding='cp949', index_col=0)
accident_15min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=10 1to5 unbalanced/accident_30_min_before.csv", encoding='cp949', index_col=0)
accident_20min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=10 1to5 unbalanced/accident_40_min_before.csv", encoding='cp949', index_col=0)

non_accident_5min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=10 1to5 unbalanced/non_accident_10_min_before.csv", encoding='cp949', index_col=0)
non_accident_10min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=10 1to5 unbalanced/non_accident_20_min_before.csv", encoding='cp949', index_col=0)
non_accident_15min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=10 1to5 unbalanced/non_accident_30_min_before.csv", encoding='cp949', index_col=0)
non_accident_20min_before = pd.read_csv("TRAFNET SEQ=4 SHIFT=10 1to5 unbalanced/non_accident_40_min_before.csv", encoding='cp949', index_col=0)

# Each accident, non_accident data (to numpy shape)
static_accident_X = accident_5min_before[['lanes', 'speed_limit', 'length', 'bump', 'camera']].to_numpy()
static_non_accident_X = non_accident_5min_before[['lanes', 'speed_limit', 'length', 'bump', 'camera']].to_numpy()

dynamic_accident_5min = accident_5min_before.drop(['datetime', 'link', 'lanes','speed_limit','length', 'bump', 'camera' ,'label'], axis=1).to_numpy()
dynamic_accident_10min = accident_10min_before.drop(['datetime', 'link', 'lanes','speed_limit','length', 'bump', 'camera' ,'label'], axis=1).to_numpy()
dynamic_accident_15min = accident_15min_before.drop(['datetime', 'link', 'lanes','speed_limit','length', 'bump', 'camera' ,'label'], axis=1).to_numpy()
dynamic_accident_20min = accident_20min_before.drop(['datetime', 'link', 'lanes','speed_limit','length', 'bump', 'camera' ,'label'], axis=1).to_numpy()

dynamic_non_accident_5min = non_accident_5min_before.drop(['datetime', 'link', 'lanes','speed_limit','length', 'bump', 'camera' ,'label'], axis=1).to_numpy()
dynamic_non_accident_10min = non_accident_10min_before.drop(['datetime', 'link', 'lanes','speed_limit','length', 'bump', 'camera' ,'label'], axis=1).to_numpy()
dynamic_non_accident_15min = non_accident_15min_before.drop(['datetime', 'link', 'lanes','speed_limit','length', 'bump', 'camera' ,'label'], axis=1).to_numpy()
dynamic_non_accident_20min = non_accident_20min_before.drop(['datetime', 'link', 'lanes','speed_limit','length', 'bump', 'camera' ,'label'], axis=1).to_numpy()

accident_label = accident_5min_before['label'].to_numpy()
non_accident_label = non_accident_5min_before['label'].to_numpy()

# Construct dataset - train, validation, test
# For LSTM model, data sequence is 4 (20, 15, 10, 5 mintues before)
dynamic_accident_X, dynamic_accident_Y = [], []
for accident in range(len(accident_label)):
    dynamic_accident_X.append([dynamic_accident_20min[accident], dynamic_accident_15min[accident], dynamic_accident_10min[accident], dynamic_accident_5min[accident]])
    dynamic_accident_Y.append(accident_label[accident])

dynamic_non_accident_X, dynamic_non_accident_Y = [], []
for non_accident in range(len(non_accident_label)):
    dynamic_non_accident_X.append([dynamic_non_accident_20min[non_accident], dynamic_non_accident_15min[non_accident], dynamic_non_accident_10min[non_accident], dynamic_non_accident_5min[non_accident]])
    dynamic_non_accident_Y.append(non_accident_label[non_accident])

# ----------------Negative Data ratio 맞추기 ---------------------------# (현재 1 : 2)#
dataset_size = int(len(dynamic_non_accident_X)) # 데이터셋 개수
data_indexes = np.arange(dataset_size)
np.random.shuffle(data_indexes)  # index 섞기
data_indexes = data_indexes[:round(len(dynamic_accident_X) * 0.5)]

# 선택된 index의 Negative 데이터를 각 list에 저장
dynamic_non_accident_X = [dynamic_non_accident_X[i] for i in data_indexes]
dynamic_non_accident_Y = [dynamic_non_accident_Y[i] for i in data_indexes]
static_non_accident_X = [static_non_accident_X[i] for i in data_indexes]

# ---------------------------Positive Dataset----------------------------------#
# Dynamic train data : validation data : test data = 8 : 1 : 1
dy_pos_X_train, dy_pos_X_test, dy_pos_Y_train, dy_pos_Y_test = train_test_split(dynamic_accident_X, dynamic_accident_Y, test_size=0.2, random_state=42, shuffle=True)
dy_pos_X_val, dy_pos_X_test, dy_pos_Y_val, dy_pos_Y_test = train_test_split(dy_pos_X_test, dy_pos_Y_test, test_size=0.5, random_state=42, shuffle=True)

# Static train data : validation data : test data = 8 : 1 : 1
st_pos_X_train, st_pos_X_test, st_pos_Y_train, st_pos_Y_test = train_test_split(static_accident_X, dynamic_accident_Y, test_size=0.2, random_state=42, shuffle=True)
st_pos_X_val, st_pos_X_test, st_pos_Y_val, st_pos_Y_test = train_test_split(st_pos_X_test, st_pos_Y_test, test_size=0.5, random_state=42, shuffle=True)

# ---------------------------Negative Dataset----------------------------------#
# Dynamic train data : validation data : test data = 8 : 1 : 1
dy_neg_X_train, dy_neg_X_test, dy_neg_Y_train, dy_neg_Y_test = train_test_split(dynamic_non_accident_X, dynamic_non_accident_Y, test_size=0.2, random_state=42, shuffle=True)
dy_neg_X_val, dy_neg_X_test, dy_neg_Y_val, dy_neg_Y_test = train_test_split(dy_neg_X_test, dy_neg_Y_test, test_size=0.5, random_state=42, shuffle=True)

# Static train data : validation data : test data = 8 : 1 : 1
st_neg_X_train, st_neg_X_test, st_neg_Y_train, st_neg_Y_test = train_test_split(static_non_accident_X, dynamic_non_accident_Y, test_size=0.2, random_state=42, shuffle=True)
st_neg_X_val, st_neg_X_test, st_neg_Y_val, st_neg_Y_test = train_test_split(st_neg_X_test, st_neg_Y_test, test_size=0.5, random_state=1234, shuffle=True)


# --------------------------Total Dataset construction--------------------------#
random.seed(42)

dy_X_train = np.concatenate((dy_pos_X_train, dy_neg_X_train), axis=0)
#random.shuffle(dy_X_train)
dy_X_val = np.concatenate((dy_pos_X_val, dy_neg_X_val), axis=0)
#random.shuffle(dy_X_val)
dy_X_test = np.concatenate((dy_pos_X_test, dy_neg_X_test), axis=0)
#random.shuffle(dy_X_test)

st_X_train = np.concatenate((st_pos_X_train, st_neg_X_train), axis=0)
#random.shuffle(st_X_train)
st_X_val = np.concatenate((st_pos_X_val, st_neg_X_val), axis=0)
#random.shuffle(st_X_val)
st_X_test = np.concatenate((st_pos_X_test, st_neg_X_test), axis=0)
#random.shuffle(st_X_test)

Y_train = np.concatenate((dy_pos_Y_train, dy_neg_Y_train), axis=0)
#random.shuffle(Y_train)
Y_val = np.concatenate((dy_pos_Y_val, dy_neg_Y_val), axis=0)
#random.shuffle(Y_val)
Y_test = np.concatenate((dy_pos_Y_test, dy_neg_Y_test), axis=0)
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

# LSTM-MLP model (proposed our model)
input_dy = Input(shape=(4, 11))
input_st = Input(shape=(5,))

# Dynamic Feature
lstm_1 = LSTM(64, return_sequences=True)(input_dy)
lstm_2 = LSTM(32, return_sequences=False)(lstm_1)
lstm_3 = Dense(16, activation='relu')(lstm_2)

# Static Feature
mlp_1 = Dense(100, activation='relu')(input_st)
mlp_2 = Dense(100, activation='relu')(mlp_1)
    
# Concatenate
concat = concatenate([lstm_3, mlp_2], axis=1)
dense_4 = Dense(100, activation='relu')(concat)

# Output
out = Dense(1, activation='sigmoid')(dense_4)

lstm_mlp = Model ([input_dy, input_st], out)
lstm_mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
lstm_mlp.summary()

early_stopping = EarlyStopping(monitor = 'val_auc', min_delta = 0, patience = 20, mode = 'max')

hist = lstm_mlp.fit([dy_X_train, st_X_train], Y_train, epochs=50, batch_size=32, validation_data=([dy_X_val, st_X_val], Y_val), callbacks = [early_stopping], verbose=1)

loss, acc, auc, precision, recall, binary_accuracy = lstm_mlp.evaluate([dy_X_test, st_X_test], Y_test, verbose=1)
print("m parameter", lstm_mlp.count_params())
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