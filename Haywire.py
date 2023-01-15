# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions import single_obj as fx

# Define neural network architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile and train model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

# Define PSO optimizer
def optimize(w):
    return fx.sphere(w)

# Optimize neural network using PSO
optimizer = GlobalBestPSO(n_particles=10, dimensions=model.count_params(), options={'c1': 0.5, 'c2': 0.3}, bounds=(-100,100))
cost, pos = optimizer.optimize(optimize, iters=100)

# Update model with optimized weights
model.set_weights(pos.reshape(model.count_params()))

# Evaluate model on test dataset
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
