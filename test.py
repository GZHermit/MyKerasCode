# coding: utf-8
from keras.layers import merge, Dense, LSTM, Input, Embedding
from keras.models import Model

# 超參數
nb_dataset = 1010  # 序列样本总数
nb_simple_seq = 100  # 单个序列的单词数量
time_step = 20  # 样本时间片数量
dict_kinds = 1e4  # 字典种类
news_len = 128  # 额外输入信息长度

# 建模型
main_input = Input(shape=(nb_simple_seq,), name='main_input')
# Embedding的作用是将字符/单词转换成向量，作用类似word2vec
x = Embedding(output_dim=512, input_dim=dict_kinds, input_length=nb_simple_seq)(main_input)
lstm_out = LSTM(32)(x)
aux_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
aux_input = Input(shape=(news_len,), name='aux_input')
x = merge(inputs=[lstm_out, aux_input], mode='concat')
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

# 定义模型Tensor -> model
model = Model(input=[main_input, aux_input], output=[main_output, aux_output])

# 生成输入数据
import numpy as np

headline_data = np.random.random(size=(nb_dataset,time_step,))
additional_data = np.random.random()
labels = np.random.randint()

# 编译模型
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# 传输数据运行模型
# model.fit(x=[headline_data, additional_data], y=[labels, labels], nb_epoch=20, batch_size=32)
model.fit({'main_input': headline_data, 'aux_input': aux_input},
          {'main_output': labels, 'aux_output': labels},
          nb_epoch=20, batch_size=32)
