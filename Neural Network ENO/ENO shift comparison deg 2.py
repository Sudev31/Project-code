#as we wanna do eno for degree = 2, let us implement lagrange interpolation for degree 2 only

#given {x_0,x_1,x_2} we want to interpolate a deg 2 polynomial through it

import matplotlib.pyplot as plt
import numpy as np

s = 100
X_main = np.linspace(-1,1,s)
Y_main=np.zeros(s)
for i in range(len(X_main)):
  if (X_main[i]>0.5):
    Y_main[i]=1
  elif (X_main[i]<-0.5):
    Y_main[i]=-1
  elif(X_main[i] == 0):
    Y_main[i]=0
  else:
    Y_main[i] = np.sin(1/X_main[i])


p = 3 #deg+1 in here
n = p+(p-2)
#for i in range(3):
#    X.append(float(input()))
#    Y.append(float(input()))

r=np.zeros(len(X_main)-2*p+3, dtype=int)

for k in range(len(X_main)-2*p+3):


  X = X_main[k:k+(p+(p-2))]
  Y = Y_main[k:k+(p+(p-2))]
  D = np.zeros((p, n))
  D[0] = Y

  for j in range(1, p): # Changed loop range to start from 1
    D[j, :(n-j)] = D[j-1, 1:n-(j-1)] - D[j-1, :n-j] # Subtraction from the second element onward

  for l in range(2,p):
    if (abs(D[l][p-3-r[k]])<abs(D[l][p-2-r[k]])):
      r[k]+=1
print(D)
print(r)

n = 900
p=p-1
y = np.zeros(n)
t = 1
fig, axes = plt.subplots(1, 2, figsize=(10, 4)) 
for l in range(len(X_main)-2*(p-1)-1):
  x = np.linspace(X_main[(p-1)+l],X_main[p+l],n)
  X = X_main[l+(p-1)-r[l]:l+2*p-r[l]]  #[2:5]  [1:4]
  Y = Y_main[l+(p-1)-r[l]:l+2*p-r[l]]
  y = np.zeros(n)
  for i in range(n):
    for j in range(len(X)):
      t =1
      for k in range(1,len(X)):
        t = t* (x[i] - X[(j+k)%(p+1)])/(X[j] - X[(j+k)%(p+1)])
      y[i] = y[i] + t*Y[j]
      t = 1
  axes[0].plot(x, y, color="blue", label="Interpolation")
  axes[0].scatter(X, Y, color="red", label="Given Points")  
  axes[0].set_title("ENO Interpolation (Degree 2)")
  axes[0].set_xlabel("x")
  axes[0].set_ylabel("y")


axes[0].grid(True)


p = 3  #deg+1 in here
#for i in range(3):
#    X.append(float(input()))
#    Y.append(float(input()))
n = 900
p=p-1
y = np.zeros(n)
t = 1
for l in range(len(X_main)-2*(p-1)-1):
  x = np.linspace(X_main[(p-1)+l],X_main[p+l],n)
  X = X_main[l+(p-1):l+2*p]  #[2:5]  [1:4]
  Y = Y_main[l+(p-1):l+2*p]
  y = np.zeros(n)
  for i in range(n):
    for j in range(len(X)):
      t =1
      for k in range(1,len(X)):
        t = t* (x[i] - X[(j+k)%(p+1)])/(X[j] - X[(j+k)%(p+1)])
      y[i] = y[i] + t*Y[j]
      t = 1
  axes[1].plot(x, y, color="blue", label="Interpolation")
  axes[1].scatter(X, Y, color="red", label="Given Points")  
  axes[1].set_title("ENO w/o shift (Degree 2)")
  axes[1].set_xlabel("x")
  axes[1].set_ylabel("y")


axes[1].grid(True)


plt.show()




