import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

# Exact solution
def u_e(x, y, t):
    return np.sin(np.pi*(x - t)) * np.cos(np.pi*(y - t))

# Domain parameters
X_min, X_max = 0.0, 1.0
Y_min, Y_max = 0.0, 1.0
T_min, T_max = 0.0, 1.0

# Generate collocation points
X = np.linspace(X_min, X_max, 50)
Y = np.linspace(Y_min, Y_max, 50)
T = np.linspace(T_min, T_max, 50)
X_grid, Y_grid, T_grid = np.meshgrid(X, Y, T, indexing='ij')
XYT = np.stack([X_grid.flatten(), Y_grid.flatten(), T_grid.flatten()], axis=1)

# Generate boundary points
num_boundary = 50

# x=0 boundary
x_b0 = np.zeros(num_boundary)
y_b0 = np.linspace(Y_min, Y_max, num_boundary)
t_b0 = np.linspace(T_min, T_max, num_boundary)
X_b0, Y_b0, T_b0 = np.meshgrid(x_b0, y_b0, t_b0, indexing='ij')
B_x0 = np.stack([X_b0.flatten(), Y_b0.flatten(), T_b0.flatten()], axis=1)
u_b_x0 = u_e(X_b0.flatten(), Y_b0.flatten(), T_b0.flatten()).reshape(-1, 1)

# x=1 boundary
x_b1 = np.ones(num_boundary)
y_b1 = np.linspace(Y_min, Y_max, num_boundary)
t_b1 = np.linspace(T_min, T_max, num_boundary)
X_b1, Y_b1, T_b1 = np.meshgrid(x_b1, y_b1, t_b1, indexing='ij')
B_x1 = np.stack([X_b1.flatten(), Y_b1.flatten(), T_b1.flatten()], axis=1)
u_b_x1 = u_e(X_b1.flatten(), Y_b1.flatten(), T_b1.flatten()).reshape(-1, 1)

# y=0 boundary
y_b0 = np.zeros(num_boundary)
x_b0 = np.linspace(X_min, X_max, num_boundary)
t_b0 = np.linspace(T_min, T_max, num_boundary)
X_b0, T_b0, Y_b0 = np.meshgrid(x_b0, t_b0, y_b0, indexing='ij')
B_y0 = np.stack([X_b0.flatten(), Y_b0.flatten(), T_b0.flatten()], axis=1)
u_b_y0 = u_e(X_b0.flatten(), Y_b0.flatten(), T_b0.flatten()).reshape(-1, 1)

# y=1 boundary
y_b1 = np.ones(num_boundary)
x_b1 = np.linspace(X_min, X_max, num_boundary)
t_b1 = np.linspace(T_min, T_max, num_boundary)
X_b1, T_b1, Y_b1 = np.meshgrid(x_b1, t_b1, y_b1, indexing='ij')
B_y1 = np.stack([X_b1.flatten(), Y_b1.flatten(), T_b1.flatten()], axis=1)
u_b_y1 = u_e(X_b1.flatten(), Y_b1.flatten(), T_b1.flatten()).reshape(-1, 1)

# Combine all boundary points and values
B_all = np.vstack([B_x0, B_x1, B_y0, B_y1])
u_b_all = np.vstack([u_b_x0, u_b_x1, u_b_y0, u_b_y1])

# Initial condition points
X_initial = np.linspace(X_min, X_max, 100)
Y_initial = np.linspace(Y_min, Y_max, 100)
X_initial, Y_initial = np.meshgrid(X_initial, Y_initial, indexing='ij')
I = np.stack([X_initial.flatten(), Y_initial.flatten(), np.zeros_like(X_initial.flatten())], axis=1)
u_i = u_e(I[:,0], I[:,1], I[:,2]).reshape(-1, 1)

# Convert to tensors
B_all_tensor = tf.constant(B_all, dtype=tf.float32)
u_b_all_tensor = tf.constant(u_b_all, dtype=tf.float32)
I_tensor = tf.constant(I, dtype=tf.float32)
u_i_tensor = tf.constant(u_i, dtype=tf.float32)

# Model architecture
model = Sequential([
    Dense(100, activation='tanh', input_shape=(3,)),
    Dense(100, activation='tanh'),
    Dense(100, activation='tanh'),
    Dense(1)
])

def pinn_loss(y_true, y_pred):
    # PDE loss
    XYT = y_true
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(XYT)
        u = model(XYT)
    u_x = tape.gradient(u, XYT)[:, 0:1]
    u_y = tape.gradient(u, XYT)[:, 1:2]
    u_t = tape.gradient(u, XYT)[:, 2:3]
    residual = u_t + u_x + u_y  # f=0 as derived
    loss_pde = tf.reduce_mean(tf.square(residual))
    
    # Boundary loss
    u_b_pred = model(B_all_tensor)
    loss_bc = tf.reduce_mean(tf.square(u_b_pred - u_b_all_tensor))
    
    # Initial condition loss
    u_i_pred = model(I_tensor)
    loss_ic = tf.reduce_mean(tf.square(u_i_pred - u_i_tensor))
    
    return loss_pde + loss_bc + loss_ic


optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=pinn_loss)

# Train the model
history = model.fit(XYT, XYT, epochs=20, batch_size=1024, verbose=1)

# Plotting results
x_plot = np.linspace(X_min, X_max, 100)
y_plot = np.linspace(Y_min, Y_max, 100)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot, indexing='ij')
T_plot = np.full_like(X_plot, T_max)
input_data = np.stack([X_plot.flatten(), Y_plot.flatten(), T_plot.flatten()], axis=1)

Z_exact = u_e(X_plot, Y_plot, T_max)
Z_pred = model.predict(input_data).reshape(X_plot.shape)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
contour = axs[0].contourf(X_plot, Y_plot, Z_exact, levels=100, cmap='viridis')
plt.colorbar(contour, ax=axs[0])
axs[0].set_title('Exact Solution')

contour = axs[1].contourf(X_plot, Y_plot, Z_pred, levels=100, cmap='viridis')
plt.colorbar(contour, ax=axs[1])
axs[1].set_title('PINN Solution')

error = np.abs(Z_exact - Z_pred)
contour = axs[2].contourf(X_plot, Y_plot, error, levels=100, cmap='magma')
plt.colorbar(contour, ax=axs[2])
axs[2].set_title('Absolute Error')

plt.tight_layout()
plt.show()