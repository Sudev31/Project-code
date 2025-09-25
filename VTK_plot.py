import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib.pyplot as plt

def u_0(x, y):
    sigma = 0.1
    return np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / (2 * sigma**2))  

def u_e(x, y, t):
    x_rot = x*np.cos(2*np.pi*t) + y*np.sin(2*np.pi*t)
    y_rot = -x*np.sin(2*np.pi*t) + y*np.cos(2*np.pi*t)
    return u_0(x_rot, y_rot)

def set_plot_style():
    plt.rcParams['font.size'] = 24
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24

def plot_vtk(filename, t):
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    
    points = vtk_to_numpy(data.GetPoints().GetData())
    solution = vtk_to_numpy(data.GetPointData().GetScalars())
    
    x = points[:, 0].reshape(100, 100)
    y = points[:, 1].reshape(100, 100)
    z = solution.reshape(100, 100)
    
    Z_e = u_e(x, y, t)
    
    set_plot_style()
    # Plot solution
    plt.figure(figsize=(10, 8))
    contour_plot = plt.contourf(x, y, z, levels=20)
    plt.colorbar(contour_plot, orientation='vertical', pad=0.07, shrink=0.58, aspect=40)
    plt.title(f"Solution at t={t:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    plt.savefig(f"solution_PINN_var_2D_t{t:.2f}.pdf", bbox_inches='tight', dpi=300, transparent=True)
    plt.show()

    # Plot error
    plt.figure(figsize=(10, 8))
    error = np.abs(Z_e - z)
    contour_plot = plt.contourf(x, y, error, levels=20)
    plt.colorbar(contour_plot, orientation='vertical', pad=0.07, shrink=0.58, aspect=40)
    plt.title(f"Error at t={t:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    plt.savefig(f"Error_PINN_var_2D_t{t:.2f}.pdf", bbox_inches='tight', dpi=300, transparent=True)
    plt.show()

    plt.figure(figsize=(10, 8))
    contour_plot = plt.contourf(x, y, Z_e, levels=20)
    plt.colorbar(contour_plot, orientation='vertical', pad=0.07, shrink=0.58, aspect=40)
    plt.title(f"Exact at t={t:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    plt.savefig(f"Exact_PINN_var_2D_t{t:.2f}.pdf", bbox_inches='tight', dpi=300, transparent=True)
    plt.show()

t=0.00
plot_vtk(f"pinn_t{t:.2f}.vtk",t)