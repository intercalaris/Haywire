import tensorflow as tf
import sklearn
import keras
from keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap 


print("\nThis Python program is an educational tool for understanding the basics of neural network construction, training, and evaluation. It uses the TensorFlow, Keras, and Scikit-learn libraries.")

print("It prompts you to input the number of hidden layers, the optimizer, the learning rate, and whether to use EarlyStopping and ReduceLROnPlateau callbacks and creates a")
print("neural network model with the specified number of hidden layers and neurons using Dense, LeakyReLU, Dropout, and BatchNormalization layers.")
print("It then trains the model on a randomly generated classification dataset using the specified optimizer and callbacks.")
print("Finally, the program visualizes the training and validation loss and accuracy, as well as the decision boundary of the trained model using a matplotlib colormap.")
input("\nClick enter to continue!")
# Prompt the user for the number of hidden layers
print("\n\n\n\n\n\nThe hidden layers in a neural network are the layers between the input and output layers. The number of hidden layers affects the model's ability to learn and generalize from the training data.\nA deeper network (more layers) could learn more complex features, but might be more prone to overfitting data.\nOne or two hidden layers are often used initially, with more added gradually to see if the additional layers improve the model's performance.")
print("\nPlease enter the number of hidden layers you want to include in the model (minimum 1): ")
n_hidden_layers = int(input())

# Prompt the user for the optimizer
print("\nThe optimizer is an algorithm that is used to adjust the weights of the neural network during training to minimize the loss function. Choose the optimizer you want to use:")
print("1. Adam: Adaptive Moment Estimation (Adam) combines the best features of two other optimizers, Adagrad and RMSprop, to adjust learning rates on a per-parameter basis. Adam is particularly useful for deep learning models and for data with high dimensionality.")
print("2. RMSprop: Root Mean Square Propagation (RMSprop) is used to minimize the fluctuations in the learning rate during the training process. RMSprop works by dividing the learning rate by an exponentially decaying average of the squared gradients.")
print("3. SGD: Stochastic Gradient Descent (SGD) updates the model's parameters based on the gradient (direction and rate of change) of the loss function by sampling random subsets of the training data instead of the entire data set. This makes SGD useful for large datasets as it can converge faster.")

optimizer_choice = int(input("\nEnter your choice (1-3): "))

print("\nThe learning rate is a hyperparameter (a set value that affects model behavior) that controls the step size taken during training.\nA higher learning rate means bigger steps and faster convergence, but also risks overshooting the minimum (lowest point on a loss function, indicating best values for model's parameters).\nA lower learning rate means smaller steps and slower convergence, but it is less likely to overshoot the minimum.")
if optimizer_choice == 1:
    learning_rate = float(input("\nEnter the learning rate for Adam (default is 0.001): "))
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    optimizer_name = "Adam"

elif optimizer_choice == 2:
    learning_rate = float(input("\nEnter the learning rate for RMSprop (default is 0.001): "))
    optimizer = RMSprop(learning_rate=learning_rate)
    optimizer_name = "RMSprop"
else:
    learning_rate = float(input("\nEnter the learning rate for SGD (default is 0.01): "))
    optimizer = SGD(learning_rate=learning_rate)
    optimizer_name = "SGD" 
    
# Prompt the user for the callbacks
callbacks = []
patience_es = None
patience_lr = None
factor = None
print("Callbacks are functions in deep learning models that can be applied at certain stages of the training process, such as at the end of each epoch (a cycle through the entire training dataset during training).\nThey provide a way to customize the behavior of the model during training and to monitor its performance.\nExamples of commonly used callbacks include EarlyStopping and ReduceLROnPlateau, which are used to stop training early if the validation loss does not improve or to reduce the learning rate if the model stops improving during training.")
print("\nDo you want to use EarlyStopping? (y/n)")
if input().lower() == 'y':
    patience_es = int(input("Enter the value of patience, or # of epochs without reduction in loss, before early stopping is implemented (default is 6): "))
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_es, verbose=1, mode='auto')
    callbacks.append(earlystopper)

print("\nDo you want to use ReduceLROnPlateau? (y/n)")
if input().lower() == 'y':
    patience_lr = int(input("Enter the value of patience, or # of epochs without reduction in loss, before ReduceLROnPlateau is implemented (default is 6): "))
    factor = float(input("Enter the factor (by which the learning rate will be reduced) for ReduceLROnPlateau (default is 0.1): "))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience_lr, min_lr=1e-5)
    callbacks.append(reduce_lr)

# Create a neural network model
model = Sequential()
model.add(Dense(64, input_dim=2, activation=LeakyReLU(alpha=0.01)))
model.add(Dropout(0.5))

neurons_in_layers = []

for i in range(n_hidden_layers):
    print("\nEach hidden layer consists of a number of neurons. The number contributes to the model's complexity.\nA larger number of neurons may lead to overfitting, while a smaller number of neurons may lead to underfitting.")
    print(f"\nEnter the number of neurons in hidden layer {i+1}: (default is 64 neurons)")
    n_neurons = int(input())
    neurons_in_layers.append(n_neurons)
    model.add(Dense(n_neurons, activation=LeakyReLU(alpha=0.01)))
    model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Create dataset
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train the model
history = model.fit(X_train, to_categorical(y_train), epochs=50, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(X_test, to_categorical(y_test), verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

# Plotting the Training Loss, Validation Loss and accuracy over the epochs
def plot_customizations(n_hidden_layers, neurons_in_layers, optimizer_name, learning_rate, callbacks, patience_es, patience_lr, factor):
    customizations = f"Customizations:"
    customizations += f" | # of Hidden Layers: {n_hidden_layers}"
    for i in range(n_hidden_layers):
        customizations += f" | # of Neurons in H.L. {i+1}: {neurons_in_layers[i]}"
    customizations += f" | Optimizer: {optimizer_name}"  # Use the optimizer's name instead of the choice number
    customizations += f" | Learning Rate: {learning_rate}"
    
    if not callbacks:
        customizations += f" | Callbacks: None"
    else:
        customizations += f" | Callbacks:"
        if patience_es is not None:
            customizations += f" EarlyStopping (patience: {patience_es})"
        if patience_lr is not None:
            customizations += f", ReduceLROnPlateau (patience: {patience_lr}, factor: {factor})"
    
    plt.figtext(0.5, 0.05, customizations, wrap=True, horizontalalignment='center', fontsize=8)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
plot_customizations(n_hidden_layers, neurons_in_layers, optimizer_name, learning_rate, callbacks, patience_es, patience_lr, factor)
plt.gcf().set_size_inches(8, 6)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
plot_customizations(n_hidden_layers, neurons_in_layers, optimizer_name, learning_rate, callbacks, patience_es, patience_lr, factor)
plt.gcf().set_size_inches(8, 6)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
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
plot_customizations(n_hidden_layers, neurons_in_layers, optimizer_name, learning_rate, callbacks, patience_es, patience_lr, factor)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.gcf().set_size_inches(8, 6)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.show()