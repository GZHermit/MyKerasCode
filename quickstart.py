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

model = Sequential()
model.add(Dense(64, init='uniform', input_dim=10))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
