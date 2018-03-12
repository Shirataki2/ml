import numpy as np
import theano
from MnistUtil import load_mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


X_train, y_train = load_mnist('mnist')
X_test, y_test = load_mnist('mnist', kind='t10k')
np.random.seed(1)
theano.config.floatX = 'float32'
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)
y_train_ohe = np_utils.to_categorical(y_train)  # one-hot

model = Sequential()

model.add(Dense(input_dim=X_train.shape[1],
                output_dim=50,
                init='uniform',
                activation='tanh'))
model.add(Dense(input_dim=50,
                output_dim=50,
                init='uniform',
                activation='tanh'))
model.add(Dense(input_dim=50,
                output_dim=y_train_ohe.shape[1],
                init='uniform',
                activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, y_train_ohe, nb_epoch=50,
          batch_size=300, verbose=1, validation_split=0.1)

y_train_pred = model.predict_classes(X_train, verbose=0)
train_acc = np.sum(y_train == y_train_pred) / X_train.shape[0]
print('Training Accuracy: %.2f%%' % (train_acc * 100.0))

y_test_pred = model.predict_classes(X_test, verbose=0)
test_acc = np.sum(y_test == y_test_pred) / X_test.shape[0]
print('Test Accuracy: %.2f%%' % (test_acc * 100.0))
