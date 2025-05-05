import numpy as np
import matplotlib.pyplot as plt

# Define the function
def  u_e(x,t):
    return np.sin(np.pi*(x-t))*np.cos(np.pi*(x-t))
def set_plot_style():
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
# Create grid
X_min, X_max = 0, 1
Y_min, Y_max = 0, 1
N = 50  # Grid resolution

x = np.linspace(X_min, X_max, N)
y = np.linspace(Y_min, Y_max, N)
X, Y = np.meshgrid(x, y, indexing='ij')

# Choose a time value
t = 0.0

# Evaluate function
Z_e = u_e(X, Y, t)

# Plotting
set_plot_style()
plt.figure(figsize=(12, 4))

# Contour plot
contour = plt.contourf(X, Y, Z_e, levels=50, cmap='viridis')
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.07, shrink=0.58, aspect=40)
plt.title("Contour Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(False)
 
"""
# Heatmap
plt.subplot(122)
plt.imshow(Z, cmap='hot', origin='lower', extent=[X_min, X_max, Y_min, Y_max])
plt.title("Heatmap")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
"""
plt.savefig(f"exact_2D_solution_t{t:.2f}.pdf")
plt.show()
