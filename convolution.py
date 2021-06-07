
import numpy as np
from keras.utils import to_categorical
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.optimizers import SGD
x = np.arange(0, 5, 0.1);
y = np.sin(x)
plt.plot(x, y)
plt.savefig('line_plot.pdf')  
train_X = np.load('train.npy')
train_X = np.expand_dims(train_X, -1)
#train_X = np.array(train_X, dtype=float)
train_Y = np.load('trainLabel.npy')

train_X = train_X.astype('float32')
train_X = train_X / 255.

train_Y_one_hot = to_categorical(train_Y)
print(train_X.shape)
print(len(train_Y))


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


batch_size = 64
epochs = 20
num_classes = 2

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))
epochs = 20
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr=0.1, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(optimizer=sgd,
              loss="binary_crossentropy",
              metrics=['accuracy'])

from sklearn.model_selection import train_test_split
train_X,valid_X,train_Y,valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=13)

class_weight = {0: 50.,
                1: 2.}
history_cnn = model.fit(train_X, train_Y, epochs=20, validation_data=(valid_X, valid_Y),batch_size=32,class_weight=class_weight)

test_eval = model.evaluate(valid_X, valid_Y, verbose=1)



print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


fashion_train=history_cnn
print('uzunlul:',len(fashion_train.history))
accuracy = fashion_train.history['accuracy']
val_accuracy = fashion_train.history['val_accuracy']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
plt.savefig('line_plot2.pdf') 

predicted_classes = model.predict(valid_X)

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(valid_Y, predicted_classes, target_names=target_names))



