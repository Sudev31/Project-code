import numpy as np
import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16

time_steps = [0, 0.2, 0.4, 0.6, 0.8, 1]
for i, t in enumerate(time_steps):
    uw = np.loadtxt(f'ftbs_t{t:.2f}.txt')
    ex = np.loadtxt(f'exact_t{t:.2f}.txt')
    pn = np.loadtxt(f'pinn_t{t:.2f}.txt')
    set_plot_style()
    plt.figure()
    plt.plot(uw[:,0],uw[:,1],'-',label='UW', c = 'k', marker ='o',markersize = 5)
    plt.plot(ex[:,0],ex[:,1],'-',label='Exact', c = 'b')
    plt.plot(pn[:,0], pn[:,1], label='PINN solution', linestyle=':', color='red',marker ='^',markersize = 5)
    plt.xlim(0,1)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(fontsize=12);
    plt.grid(True, linestyle = '--', linewidth = 0.5)

    plt.savefig(f"solution_t{t:.2f}.pdf")
