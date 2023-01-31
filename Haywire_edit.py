import os
import tensorflow as tf
from keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization, Flatten, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input

# Create a neural network model
base_model = VGG19(weights='imagenet', include_top=False)

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False

# Add a new layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['accuracy'])

# Create dataset
data_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
X_train = data_gen.flow_from_directory('path/to/training/data', target_size = (224, 224), batch_size = 32, class_mode = 'categorical')
X_test = data_gen.flow_from_directory('path/to/test/data', target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

# Train the model
history = model.fit(X_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(X_test, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))


# Plotting the Training Loss, Validation Loss and accuracy over the epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
