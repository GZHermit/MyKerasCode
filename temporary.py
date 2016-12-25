# coding : utf-8
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.optimizers import SGD

left = Sequential()
left.add(Dense(output_dim=32, input_dim=64))

right = Sequential()
right.add(Dense(output_dim=16, input_dim=128))

merged = Merge(layers=[left, right], mode='concat')

end = Sequential()
end.add(merged)
end.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True, )
end.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])

import numpy as np

leftdata = np.random.random()
rightdata = np.random.random()
labels = np.random.randint()

end.fit(x=[leftdata, rightdata], y=labels)
