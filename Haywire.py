import tensorflow as tf
import sklearn
import keras
from keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Create a neural network model
model = Sequential()
model.add(Dense(64, input_dim=2, activation=LeakyReLU(alpha=0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation=LeakyReLU(alpha=0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation=LeakyReLU(alpha=0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation=LeakyReLU(alpha=0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation=LeakyReLU(alpha=0.01)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])

# Create dataset
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train the model
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, min_lr=1e-5)
history = model.fit(X_train, to_categorical(y_train), epochs=100, batch_size=32, validation_split=0.2, callbacks=[earlystopper, reduce_lr], verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(X_test, to_categorical(y_test), verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

# Plotting the Training Loss, Validation Loss and accuracy over the epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
plt.show()

# Visualize the model
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot()
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k')
plt.title("Decision boundary of the trained model")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()