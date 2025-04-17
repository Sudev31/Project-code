import numpy as np
import matplotlib.pyplot as plt
import argparse
# this computes the solution of advection equation with speed 1
# ut+ ux = 0 with UW scheme

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-nc', type=int, help='Number of cells', default=100)
parser.add_argument('-cfl', type=float, help='CFL number', default=0.9)
parser.add_argument('-Tf', type=float, help='Final time', default=1e40)
parser.add_argument('-plot_freq', type=int, help='Frequency to plot solution',
                    default=1)
args = parser.parse_args()

def smooth(x):
    x_mod = np.mod(x, 1.0)
    return np.where((x_mod >= 0.5), 1.0, -1.0)

ng = 201;
x  = np.linspace(0,1,ng)
h = 1/(ng-1);
u  = smooth(x)  # initialize the solution vector
ue = smooth(x) # exact solution at t =0
plot_freq = args.plot_freq
cfl = 0.9
t = 0.0  # initial time
# plot initial condition
if plot_freq >0:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1,line2 = ax.plot(x, u, 'ro',x, ue, 'b')
    #line1, = ax.plot(x, u, 'o')
    ax.set_xlabel('x'); ax.set_ylabel('u')
    plt.title('ng='+str(ng)+', CFL='+str(cfl)+', time ='+str(np.round(t,3)))
    plt.legend(('Numerical','Exact'))
    #plt.ylim(0.3,0.7)
    plt.grid(True); plt.draw(); plt.pause(0.1)
    wait = input("Press enter to continue ")

Tf = 0.5; # final time
dt = cfl * h   # h is dx
iter  = 0
while t < Tf:
    lam = dt/h
    if t+dt > Tf:
        dt = Tf - t
        lam = dt/h
    # warning! uold = u is a wrong assignment
    uold = np.copy(u)
    uold[-1] = uold[ng-2] # periodic  boundary condition
    for i in range(0, ng):
        u[i] = uold[i] - lam * (uold[i] - uold[i-1])

    t +=dt
    iter +=1
    print('iter,t, solmin, solmax=',"%04d"% iter, "%.5f"% t, "%.6f"%round(u.min(),6), \
              "%.7f"%round(u.max(),7) )
    if plot_freq > 0:
        ue = smooth(x-t)
        line1.set_ydata(u)
        line2.set_ydata(ue)
        plt.title('ng='+str(ng)+', CFL='+str(cfl)+', time ='+str(np.round(t,3)))
        plt.draw(); plt.pause(0.1)
    
plt.show()

ue = smooth(x-t)
#save solution to a pdf file
plt.plot(x, u,'ko',fillstyle='none',label='UW')
plt.plot(x, ue,'-',fillstyle='none',label='Exact')
plt.xlabel('x')
plt.ylabel('u')
plt.legend();
plt.grid(True, linestyle = '--')
plt.savefig('solution.pdf')

# save solution to a  data file 
fname1  = 'exact.txt'
np.savetxt(fname1, np.column_stack([x, ue]))
fname2 = 'uw.txt'
np.savetxt(fname2, np.column_stack([x, u]) )
# u_t + c u_x = 0,  u(x,t ) = g(x-ct)
# u(x,0) = g(x)   