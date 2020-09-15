#-------------------Set Directories---------------------#
import cv2
import tensorflow as tf
import numpy as np

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

model.load_weights('model_0.h5')

print("[INFO] starting video stream...")
vc = cv2.VideoCapture(0)

success,img = vc.read()
img = cv2.resize(img[240:, 120:520, :], (80, 48))
prev_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
count = 1
while success:
    if not success:
        print('count:', count)
        break
    if count % 2 == 0:
        img = cv2.resize(img[img.shape[0]//2:, :, :], (80, 48))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data = np.concatenate((prev_img, img)).reshape((1, 80, 96, 1))/255
        prev_img = img
        out = model.predict(data)
        for i in range(3):
            if out[0][i] > 0.9:
                print(i)
                break
        else:
            print('None')
    count += 1
    
    success,img = vc.read()
        
