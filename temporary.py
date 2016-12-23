# coding : utf-8

from keras.layers import Input, Dense

x = Input(shape=(100, 50), name='x')
y = Dense()(x)
