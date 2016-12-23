# coding: utf-8

from keras.layers import merge, Dense, LSTM, Input, Embedding, Merge
from keras.models import Model

# 超參數
nb_dataset = 1010  # 序列样本总数
nb_simple_seq = 100  # 单个序列的单词数量
time_step = 20  # 样本时间片数量
dict_kinds = 1e4  # 字典种类
news_len = 128
# 建模型
main_input = Input(shape=(nb_simple_seq,), name='main_input')
# Embedding的作用是将字符/单词转换成向量，作用类似word2vec
x = Embedding(output_dim=512, input_dim=dict_kinds, input_length=nb_simple_seq)(main_input)
lstm_out = LSTM(32)(x)
aux_output = Dense(1, activation='sigmod', name='aux_output')(lstm_out)
aux_input = Input(shape=(news_len,), name='aux_input')
x = merge(inputs=[lstm_out, aux_input], mode='concat')

x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='sigmod', name='main_output')(x)


import numpy as np

# data = np.random.random((nb_dataset, time_step, data_dim))
