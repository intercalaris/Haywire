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
import pydot
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

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
model_image = model_to_dot(model, show_shapes=True, show_layer_names=True)
SVG(model_image.create(prog='dot', format='svg'))