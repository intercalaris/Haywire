import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions import single_obj as fx

# Create a neural network model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Create dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train the model
model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32, verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(X_test, to_categorical(y_test), verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

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
