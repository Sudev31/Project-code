import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def r(x, y):
    return ((x - 0.3)**2 + (y - 0.3)**2)**0.5

def u_0(x, y):
    return 1 - np.clip(r(x, y)/0.2, 0, 1)  # Clipped for better visualization

def u_e(x, y, t):
    x_rot = x*np.cos(5*t) + y*np.sin(5*t)
    y_rot = -x*np.sin(5*t) + y*np.cos(5*t)
    return u_0(x_rot, y_rot)

# Create spatial grid
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(x, y)

from matplotlib.animation import FuncAnimation

# Set up figure and axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Initialize surface plot
Z = u_e(X, Y, 0)
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis)

def update(frame):
    ax.clear()
    Z = u_e(X, Y, frame)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis)
    ax.set_title(f'Time = {frame:.2f}')
    ax.set_zlim(0, 1)
    ax.view_init(elev=30, azim=-60)
    return surf,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, 50), blit=False)
plt.show()