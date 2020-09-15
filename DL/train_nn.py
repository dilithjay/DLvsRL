import tensorflow as tf
import numpy as np
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[3]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

with open("data.txt", "r") as file:
    rows = file.read().rstrip('\n').split('\n')

xs, ys = [], []
for i in rows:
    row = list(map(float, i.split()))
    print(row)
    xs.append(row[:3])
    ys.append(row[-1])

model.fit(xs, ys, epochs=100)

model.save_weights('model.h5')