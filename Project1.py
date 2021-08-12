from tensorflow import keras
import numpy as np


# Defining and compiling a model
model = keras.Sequential([keras.layers.Dense(units=2, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Defining the array
x = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9])
y = np.array([10, 20, 25, 30, 40, 45, 40, 50, 60, 55])

# Combining the two
model.fit(x, y, epochs=5000)

print(model.predict([0.0]))
print(model.predict([10.0]))