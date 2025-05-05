"""
number of cells =nx X ny
Solve scalar conservation law with periodic bc
To get help, type
    python lwfr.py -h
"""
import os
print("Working directory:", os.getcwd())
import time 

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys
# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-pde', choices=('linear', 'varadv', 'burger', 'bucklev'),
                    help='PDE', default='linear')
parser.add_argument('-scheme', choices=('uw','rk2' ), help='uw',
                    default='uw')
parser.add_argument('-ncellx', type=int, help='Number of x cells', default=50)
parser.add_argument('-ncelly', type=int, help='Number of y cells', default=50)
parser.add_argument('-cfl', type=float, help='CFL number', default=1.0)
parser.add_argument('-Tf', type=float, help='Final time', default=1.0)

parser.add_argument('-ic', choices=('sin2pi', 'expo','hat', 'solid'),
                    help='Initial condition', default='sin2pi')

args = parser.parse_args()

# Select PDE

# Select initial condition
if args.ic == 'sin2pi':
    from sin2pi import *
else:
    print('Unknown initial condition')
    exit()

def u_e(x, y, t):
    return np.sin(2*np.pi*(x-t)) * np.sin(2*np.pi*(y-t))
with open("convergence_table.txt", "w") as f:
    f.write("1/N\th\tTime\tL1_Error\n")  # Header
for i in range(1,6):
    # Select cfl
    cfl = args.cfl
    nx = args.ncellx*i       # number of cells in the x-direction
    ny = args.ncelly*i       # number of cells in the y-direction
    dx = (xmax - xmin)/nx
    dy = (ymax - ymin)/ny
    # Allocate solution variables
    v = np.zeros((nx+5, ny+5))  # 2 ghost points each side
    # Set initial condition by interpolation
    for i in range(nx+5):
        for j in range(ny+5):
            x = xmin + (i-2) * dx     
            y = ymin + (j-2) * dy
            val = initial_condition(x, y)
            v[i, j] = val
    # copy the initial condition
    v0 = v[2:nx+3, 2:ny+3].copy()
    # it stores the coordinates of real cell vertices 
    xgrid1 = np.linspace(xmin, xmax, nx + 1)
    ygrid1 = np.linspace(ymin, ymax, ny +1 )
    ygrid, xgrid = np.meshgrid(ygrid1, xgrid1)

    # Fill ghost cells using periodicity
    def update_ghost(v1):
        # left ghost cell
        v1[0,:] = v1[nx,:]
        v1[1,:] = v1[nx+1,:]
        # right ghost cell
        v1[nx+3,:] = v1[3,:]
        v1[nx+4,:] = v1[4,:]
        # bottom ghost cell
        v1[:,0]= v1[:,ny]
        v1[:,1]= v1[:,ny+1]
        # top ghost cell
        v1[:,ny+4] = v1[:,4]
        v1[:,ny+3] = v1[:,3]

    # Set time step from CFL condition
    start_time = time.time()
    dt = cfl/(1.0/dx + 1.0/dy + 1.0e-14)

    iter, t = 0, 0.0
    Tf = args.Tf   
    #save initial data

    error = 0
    while t < Tf:
        if t+dt > Tf:
            dt = Tf - t
        lamx, lamy = dt/dx,  dt/dy
        # Loop over real cells (no ghost cell) and compute cell integral
        update_ghost(v)
        v_old = v.copy()
        for i in range(2, nx+3):
            x = xmin + (i-2)*dx
            for j in range(2, ny+3):
                y = ymin + (j-2)*dy
                v[i,j] = v_old[i,j] - lamx *(v_old[i,j] - v_old[i-1,j]) - lamy *(v_old[i,j] - v_old[i,j-1])
                exact = u_e(x,y,t)
                error = error + abs((v[i,j] - exact))
        t, iter = t+dt, iter+1
        
        
    end_time = time.time()
    total_space_time_points = iter * nx * ny  # iter = number of time steps
    L1_error = error / total_space_time_points
    print(f"L1 Error (Entire Domain): {L1_error}")

    total_time = end_time - start_time
    inv_N = 1.0 / total_space_time_points # 1/N
    h = dx            # Grid spacing

    # Append results to file
    with open("convergence_table.txt", "a") as f:
        f.write(f"{inv_N}\t{h}\t{total_time}\t{L1_error}\n")


