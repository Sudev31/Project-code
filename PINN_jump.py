"""
Trained a 1D PINN for the transport equation u_t + k u_x = 0 with a discontinuous
(square-wave) initial condition. Compared the PINN solution with a FTBS solver
and the analytical shifted initial-condition solution. Produces:
  - heatmap of PINN prediction
  - heatmap of analytical solution
  - heatmap of absolute error
  - time-slice comparison plots (PINN vs FTBS vs exact)
Usage:
  python pinn_1d_jump.py
Notes / Warnings:
  - PINNs often struggle to approximate discontinuities; results may show smoothing
    near the jump. The purpose is to demonstrate the same
"""

import numpy as np

# for building and training neural networks
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.regularizers import l2
def set_plot_style():
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
def u_initial(x):
    x_mod = np.mod(x, 1.0)
    return np.where((x_mod >= 0.5), 1.0, -1.0)

def u_e(x, t):
    return u_initial(x - t)

#generating the data points where y will be computed
# we are trying to solve the pde u_t + k*u_x = f(x)

T_min = 0
T_max = 1
X_min = 0
X_max = 1
X_1 = np.linspace(X_min, X_max/2, 100)
X_2 = np.linspace(X_max/2, X_max, 100)
X = np.append(X_1, X_2)
T = np.linspace(T_min, T_max, 200)

X_grid, T_grid = np.meshgrid(X, T)   # shape: (20, 20) each

X_flat = X_grid.flatten().reshape(-1, 1)  # shape: (400, 1)
T_flat = T_grid.flatten().reshape(-1, 1)

XT = np.hstack((X_flat, T_flat))   # shape: (400, 2)
X_B1 = X_min*np.ones_like(X)
X_B1 = X_B1.reshape(-1,1)
T_new = T.reshape(-1,1)
B1 = np.hstack((X_B1, T_new))
X_B2 = X_max*np.ones_like(X)
X_B2 = X_B2.reshape(-1,1)                # Shape (100, 1), filled with 1
B2 = np.hstack((X_B2, T_new))

T_i = 0*np.ones_like(X)
T_i = T_i.reshape(-1,1)
X_new = X.reshape(-1,1)
I = np.hstack((X_new, T_i))

X1_new = X_1.reshape(-1,1)
X2_new = X_2.reshape(-1,1)
T_i = 0*np.ones_like(X_1)
T_i = T_i.reshape(-1,1)
I_1 = np.hstack((X1_new, T_i))
I_2 = np.hstack((X2_new,T_i))

#model archutecture
model = Sequential([
    tf.keras.Input(shape=(2,)),  # Input size for stencil points
    Dense(40, activation='tanh'),
    Dense(40, activation='tanh'), # Added closing parenthesis here
    Dense(40, activation='tanh'), # Added closing parenthesis here
    Dense(1), # Added closing parenthesis here

])

def pinn_loss(y_true, y_pred):
    # y_true contains [X, T]
    XT = y_true
    X = XT[:, 0:1]
    T = XT[:, 1:2]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(XT)
        u = model(XT)

    grads = tape.gradient(u, XT)
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    # PDE: u_t + k * u_x = f(x)
    k = 1.0
    f = 0.0  # RHS forcing term

    # PDE residual
    residual = u_t + k * u_x - f
    loss_pde = tf.reduce_mean(tf.square(residual))

    # Boundary and initial conditions

    u_b1 = model(B1)
    u_b2 = model(B2)
    u_i_1  = model(I_1)
    u_i_2  = model(I_2)

    #loss_bc = tf.reduce_mean(tf.square(u_b1 - u_b2))
    x_i_1 = I_1[:, 0:1]
    x_i_2 = I_2[:, 0:1]
    t_i_1 = I_1[:, 1:2]
    t_i_2 = I_2[:, 1:2]
    
    x_i_1 = tf.cast(x_i_1, dtype=tf.float32)
    x_i_2 = tf.cast(x_i_2, dtype=tf.float32)    # extract x values from initial points
    t_i_1 = tf.cast(t_i_1, dtype=tf.float32) 
    t_i_2 = tf.cast(t_i_2, dtype=tf.float32)   # extract t values from initial points
    u_i_true_1 = -1
    u_i_true_2 = 1 #initial data
    #b1_data = tf.sin(np.pi * (-1 - t_i)) * tf.cos(np.pi * (-1 - t_i))

    loss_ic_1 = tf.reduce_mean(tf.square(u_i_1 - u_i_true_1))
    loss_ic_2 = tf.reduce_mean(tf.square(u_i_2 - u_i_true_2))
    loss_ic = loss_ic_1 + loss_ic_2
    #loss_bc = tf.reduce_mean(tf.square(u_b1 - b1_data))
    loss_bc = tf.reduce_mean(tf.square(u_b1 - u_b2))
    # Total loss
    total_loss = loss_pde + loss_ic  + loss_bc
    return total_loss
model.compile(
    loss=pinn_loss,  # Use sparse labels
    optimizer=tf.keras.optimizers.Adam(0.001))
# Train the model
y_dummy = XT.copy()  # needed for pinn_loss to pass as y_true
history = model.fit(XT, y_dummy, epochs=15)

X = np.linspace(X_min, X_max, 500)
T = np.linspace(T_min, T_max, 500)
X_grid, T_grid = np.meshgrid(X, T)

# 2. Flatten and stack into model input (shape: [10000, 2])
X_flat = X_grid.flatten().reshape(-1, 1)
T_flat = T_grid.flatten().reshape(-1, 1)
XT_input = np.hstack((X_flat, T_flat))  # shape: (10000, 2)

# 3. Predict
Z_pred = model.predict(XT_input, verbose=0)  # shape: (10000, 1)

# 4. Reshape predictions back to grid shape
Z = Z_pred.reshape(X_grid.shape)  # shape: (100, 100)
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, T, Z, 100, cmap='viridis')  # 100 contour levels
plt.colorbar(cp)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Heatmap of PINN approximation')
plt.tight_layout()
plt.savefig("PINN_solution_jump.pdf", format="pdf", bbox_inches="tight")
plt.show()

X_min, X_max = 0, 1
T_min, T_max = 0, 1


# Generate grid
X = np.linspace(X_min, X_max, 500)
T = np.linspace(T_min, T_max, 500)
X_grid, T_grid = np.meshgrid(X, T)

# Compute analytical solution
Z_analytical = np.sin(np.pi * (X_grid - T_grid))*np.cos(np.pi*(X_grid - T_grid))

# Plot surface
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, T, Z_analytical, 100, cmap='viridis')  # 100 contour levels
plt.colorbar(cp)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Heatmap of u(x, t)')

plt.show()

error = np.abs(Z_analytical - Z)

# Plot heatmap of the error
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, T, error, 100, cmap='viridis')  # or 'coolwarm', 'plasma', etc.
plt.colorbar(cp)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Heatmap of |u_analytical - u_PINN| (Absolute Error)')
plt.tight_layout()
plt.savefig("comparison_heatmap_jump.pdf", format="pdf", bbox_inches="tight")
plt.show()

def u_exact(x, t):
    return u_initial(x - t)

# Domain settings
X = 1.0  # Domain length [0, 1]
T = 1.0  # Final time

# Grid parameters
del_x = 0.005  # Spatial step
del_t = 0.9 * del_x  # CFL = 0.5 (stable for FTBS)
size_x = int(X / del_x) + 1  # Number of spatial points (0 to X inclusive)
size_t = int(T / del_t) + 1  # Number of time steps

# Initialize solution array
u = np.zeros((size_t, size_x))
x_vals = np.linspace(0, X, size_x)
u[0, :] = u_initial(x_vals)  # Set initial condition

# FTBS time-stepping
for n in range(size_t - 1):
    # Update interior points
    for j in range(1, size_x):
        u[n + 1, j] = u[n, j] - (del_t / del_x) * (u[n, j] - u[n, j - 1])
    # Periodic boundary (u[0] = u[-1])
    u[n + 1, 0] = u[n + 1, -1]

# Generate exact solution
X_grid, T_grid = np.meshgrid(x_vals, np.linspace(0, T, size_t))
u_exact_vals = u_exact(X_grid, T_grid)

# Plotting
fig, (ax1) = plt.subplots(1)

# Approximate solution
c1 = ax1.imshow(u, extent=[0, X, T,0], aspect='auto', cmap='viridis')
fig.colorbar(c1, ax=ax1, label='u(x,t)')
ax1.set_title('FTBS Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.invert_yaxis()


time_steps = [0, 0.2, 0.4, 0.6, 0.8, 1]
set_plot_style()

for i, t in enumerate(time_steps):
    fig, ax = plt.subplots(figsize=(8, 4))

    x_domain = np.arange(0, X_max + del_x, del_x)
    t_domain = np.arange(0, X_max + del_t, del_t)
    u_t_4 = u[int(t / del_t), :]

    X = np.linspace(X_min, X_max, int(X_max / del_x))
    T = np.ones(len(X)) * t
    exact = u_e(X, T)

    X_reshaped = X.reshape(-1, 1)
    T_reshaped = T.reshape(-1, 1)
    XT_input = np.hstack((X_reshaped, T_reshaped))

    Z_pred = model.predict(XT_input, verbose=0)
    Z = Z_pred.reshape(len(x_domain) - 1)

    ax.plot(x_domain, u_t_4, '-',label='FTBS solution', c = 'k', marker ='o',markersize = 5)
    ax.plot(X, Z, label='PINN solution', linestyle=':', color='red',marker ='^',markersize = 5)
    ax.plot(X, exact,'-', label='Exact solution',c = 'b')
    ax.set_title(f't = {t:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    filename = f"solution_jump_t{t:.2f}.pdf"
    plt.savefig(filename, format='pdf', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    filename = f"solution_t{t:.2f}.pdf"
    plt.savefig(filename, format='pdf', dpi=600, bbox_inches='tight')
    fname1 = f'exact_t{t:.2f}.txt'
    np.savetxt(fname1, np.column_stack([X, exact]))
    fname2 = f'ftbs_t{t:.2f}.txt'
    np.savetxt(fname2, np.column_stack([x_domain, u_t_4]))
    fname3 = f'pinn_t{t:.2f}.txt'
    np.savetxt(fname3, np.column_stack([X, Z.flatten()]))  # Ensure Z is flattened
    plt.close(fig)
