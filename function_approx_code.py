import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import time
from tensorflow.keras.callbacks import Callback

def u_0(x, y):
    sigma = 0.1
    return np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / (2 * sigma**2))  

def u_e(x, y, t):
    x_rot = x*np.cos(2*np.pi*t) + y*np.sin(2*np.pi*t)
    y_rot = -x*np.sin(2*np.pi*t) + y*np.cos(2*np.pi*t)
    return u_0(x_rot, y_rot)

X_min, X_max = -1, 1.0
Y_min, Y_max = -1, 1.0
T_min, T_max = 0.0, 1.0

# Generate collocation points
X = np.linspace(X_min, X_max, 50)
Y = np.linspace(Y_min, Y_max, 50)
T = np.linspace(T_min, T_max, 50)
X_grid, Y_grid, T_grid = np.meshgrid(X, Y, T, indexing='ij')
XYT = np.stack([X_grid.flatten(), Y_grid.flatten(), T_grid.flatten()], axis=1)

x, y, t = XYT[:, 0], XYT[:, 1], XYT[:, 2]
Y_true = u_e(x, y, t)

model = Sequential([
    Dense(100, activation=tf.math.sin, input_shape=(3,)),
    Dense(100, activation=tf.math.sin),
    Dense(100, activation=tf.math.sin),
    Dense(100, activation=tf.math.sin),
    Dense(100, activation=tf.math.sin),
    Dense(100, activation=tf.math.sin),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

history = model.fit(XYT, Y_true, epochs=35, batch_size=1024, verbose=1)

# Fix final time
t_final = 0.5

# Grid for (x, y) at t = 1.0
x_vals = np.linspace(X_min, X_max, 100)
y_vals = np.linspace(Y_min, Y_max, 100)
X_plot, Y_plot = np.meshgrid(x_vals, y_vals)
T_plot = np.full_like(X_plot, t_final)

# Prepare input for model
XYT_test = np.stack([X_plot.flatten(), Y_plot.flatten(), T_plot.flatten()], axis=1)

# Compute true values and predicted values
Z_true = u_e(XYT_test[:, 0], XYT_test[:, 1], XYT_test[:, 2]).reshape(100, 100)
Z_pred = model.predict(XYT_test, verbose=0).reshape(100, 100)

# Plot contours
plt.figure(figsize=(12, 5))

# True
plt.subplot(1, 2, 1)
cp1 = plt.contourf(X_plot, Y_plot, Z_true, levels=100, cmap='viridis')
plt.title('True $u_e(x, y, t=1.0)$')
plt.colorbar(cp1)

# Predicted
plt.subplot(1, 2, 2)
cp2 = plt.contourf(X_plot, Y_plot, Z_pred, levels=100, cmap='inferno')
plt.title('Predicted $\\hat{u}_e(x, y, t=1.0)$')
plt.colorbar(cp2)

plt.tight_layout()
plt.show()
