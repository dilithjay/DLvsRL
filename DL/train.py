#-------------------Set Directories---------------------#
import os

train_dir = os.path.join(os.getcwd(), 'train')
validation_dir = os.path.join(os.getcwd(), 'validation')

# Directory with our training pictures
train_data0_dir = os.path.join(train_dir, 'data0')
train_data1_dir = os.path.join(train_dir, 'data1')
train_data2_dir = os.path.join(train_dir, 'data1')


# Directory with our validation pictures
validation_data0_dir = os.path.join(validation_dir, 'data0')
validation_data1_dir = os.path.join(validation_dir, 'data1')
validation_data1_dir = os.path.join(validation_dir, 'data2')


import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


training_datagen = ImageDataGenerator(rescale = 1./255)

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	train_dir,
    color_mode = 'grayscale',
	target_size=(80, 96),
	class_mode='categorical',
    batch_size=10
)

validation_generator = validation_datagen.flow_from_directory(
	validation_dir,
    color_mode = 'grayscale',
	target_size=(80, 96),
	class_mode='categorical',
    batch_size=10
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(80, 96, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_generator, epochs=50, validation_data = validation_generator, verbose = 1)

# serialize weights to HDF5
model.save_weights("model_0.h5")
print("Saved model to disk")

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()
