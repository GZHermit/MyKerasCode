# coding: utf-8

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, TimeDistributed
from keras.optimizers import SGD

# model = Sequential()
# # model.add(Dense(output_dim=32,input_dim=128))
#
# # 这里input_shape是有10个vector，每个vector长度为12
# model.add(TimeDistributed(Dense(output_dim=32), input_shape=(10, 12)))
# model.add(LSTM(32, input_shape=(10, 11)))
# model.add(Activation('relu'))

# 建立模型
model = Sequential()
model.add(layer=Dense(output_dim=1, init='uniform', input_dim=512))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

# 设置优化器，即梯度下降函数
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# 对模型进行编译
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

import numpy as np

# 生成输入数据
data = np.random.random((1000, 512))
labels = np.random.randint(2, size=(1000, 1))
# 运行模型
model.fit(x=data, y=labels, batch_size=50, nb_epoch=10)
