# coding: utf-8

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

inputs = Input(shape=(784,))

x = Dense(output_dim=64, activation='relu')(inputs)
x = Dense(output_dim=64, activation='relu')(x)
predictions = Dense(output_dim=10, activation='softmax')(x)

model = Model(input=inputs, output=predictions)
adam = Adam()
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

import numpy as np

data = np.random.random((100, 784))
# print data
labels = np.random.randint(10, size=(100, 10))
# print labels
model.fit(x=data, y=labels, batch_size=10, nb_epoch=20)
