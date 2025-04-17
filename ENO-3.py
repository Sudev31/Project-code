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
from tensorflow.keras.models import Model

s = 500
x = np.linspace(-1,1,s)
# Initialize X_train outside the loop
X_train = []
Y=np.zeros(4)
r=np.zeros(s-3, dtype=int)
for k in range(s-3):
  X = x[k:k+4] # X is now correctly defined within the loop
  for i in range(4):
    if (X[i]>=0.5):
      Y[i]=1
    elif (X[i]<=-0.5):
      Y[i]=-1
    elif(X[i] == 0):
      Y[i]=0
    else:
      Y[i] = np.sin(1/X[i])
  X_train.append(Y.copy())

  p = 3
  n = p+1
  D = np.zeros((p, n))
  D[0] = Y

  for j in range(1, p): # Changed loop range to start from 1
    D[j, :(p-(j-1))] = D[j-1, 1:n-(j-1)] - D[j-1, :n-j] # Subtraction from the second element onward


  if (abs(D[2][p-3-r[k]])<abs(D[2][p-2-r[k]])):
    r[k]+=1

X_train = np.array(X_train)

#particular matrix

W1 = np.array([
    [0,1,-2,1],
    [1, -2, 1, 0],
    [0,-1, 2, -1],
    [-1,2,-1,0],

])
W1 = W1.T  # Transpose to match TensorFlow's shape (6,10)

W2 = np.array([
    [-1, 1, -1,1],
    [1,-1,1,-1]
])
W2 = W2.T  # Transpose (10,6) → (6,10)

W3 = np.array([
    [1,0],
    [0,1],

])
W3 = W3.T  # Transpose (6,3)

from tensorflow.keras.layers import LayerNormalization

model = Sequential([
    tf.keras.Input(shape=(4,)),

    Dense(4, activation='relu',
          kernel_initializer=tf.keras.initializers.Constant(W1),
          bias_initializer=tf.keras.initializers.Zeros(),trainable=False),


    Dense(2, activation='softmax',
          kernel_initializer=tf.keras.initializers.Constant(W2),
          bias_initializer=tf.keras.initializers.Zeros(),trainable=False
          ),
])


# Compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Use sparse labels (integer form)
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=['sparse_categorical_accuracy']
)

# Check model summary
model.summary()
print(r)
# Train the model
history = model.fit(
    X_train, r,  # Sparse labels (integer form)
    epochs=50,
    verbose=1
)

predictions = np.zeros(s-3)
for k in range(s-3):
  X = x[k:k+4]
  for i in range(4):
    if (X[i]>0.5):
      Y[i]=1
    elif (X[i]<-0.5):
      Y[i]=-1
    elif(X[i] == 0):
      Y[i]=0
    else:
      Y[i] = np.sin(1/X[i])
    X_test = Y.reshape(1, -1)

  Y_test = np.array([r[k]])

  predictions[k] = np.argmax(model.predict(X_test)) # Get the class with highest probability
temp = 0
for i in range(len(r)):  # Use len(r) instead of hardcoding 193
    if r[i] != predictions[i]:
        temp += 1  # More concise way to increment
        print(i)
    
#as we wanna do eno for degree = 2, let us implement lagrange interpolation for degree 2 only

#given {x_0,x_1,x_2} we want to interpolate a deg 2 polynomial through it

#if deg is fixed, we can define th elagrange polynomial directly
import matplotlib.pyplot as plt
import numpy as np

s = 500
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

temp = 0
for i in range(len(r)):  # Use len(r) instead of hardcoding 193
    if r[i] != predictions[i]:
        temp += 1  # More concise way to increment
        print(i)

print(f"Total mismatches: {temp}")  # Print total mismatches at the end

r = predictions
r = np.round(predictions).astype(int)
n = 900
p=p-1
y = np.zeros(n)
t = 1
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
  plt.plot(x, y, color="blue")
  plt.scatter(X, Y, color="red")  # Mark the given points
  plt.title("ENO-3 Interpolation")
  plt.xlabel("x")
  plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("plot.pdf", format="pdf", bbox_inches="tight")
plt.show()



s = 600
X_main = np.linspace(-1, 1, s) 
Y = np.zeros(4)  
predictions = np.zeros(s - 3)

for k in range(s-3):  
    X = X_main[k:k+4]  
    for i in range(4):
        if X[i] > 0:
            Y[i] = 1
        elif X[i] < 0:
            Y[i] = -1
        else:
            Y[i] = 1
    X_test = Y.reshape(1, -1)
    predictions[k] = np.argmax(model.predict(X_test))  # Printing Y in both iterations
print(predictions)
X_main = np.linspace(-1,1,s)
Y_main = np.zeros(s)

for i in range(len(X_main)):
  if X_main[i]>0:
    Y_main[i]=1
  elif X_main[i]<0:
    Y_main[i]=-1
  else:
    Y_main[i] = 1  
  

p = 3
 #deg+1 in here
n = p+(p-2)
#for i in range(3):
#    X.append(float(input()))
#    Y.append(float(input()))

#update r formula in the other code previous one
r = predictions
r = np.round(predictions).astype(int)
n = 900
p=p-1
y = np.zeros(n)
t = 1
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
  plt.plot(x, y, color="blue")
  plt.scatter(X, Y, color="red")  # Mark the given points
  plt.title("ENO-3 Interpolation (Degree 2)")
  plt.xlabel("x")
  plt.ylabel("y")




plt.legend()
plt.grid(True)
plt.savefig("plot1.pdf", format="pdf", bbox_inches="tight")
plt.show()


