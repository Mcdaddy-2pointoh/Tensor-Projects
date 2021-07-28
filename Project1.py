from tensorflow import keras
import numpy as np

# Defining and compiling a model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Defining the array
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

# Combining the two
model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))