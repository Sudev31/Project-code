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

import numpy as np

#let's us try with the function f(x) = x^2+sin(x)
#also the case being implemented here is for p=3 only, for higher p , the last if statement will also havea for loop.
p = 4
s = 600

x = np.linspace(-1,1,s)
n = p+(p-2)
D = np.zeros((p, p+(p-2)))
Y=np.zeros(p+(p-2))
X_train = []
r=np.zeros(s-5, dtype=int)
for k in range(s-5):
  X = x[k:k+6]  #here we are taking 6 points one by one
  for i in range(6):
    if (X[i]>0.5):
      Y[i]=1
    elif (X[i]<-0.5):
      Y[i]=-1
    elif(X[i] == 0):
      Y[i]=0
    else:
      Y[i] = np.sin(1/X[i])
  X_train.append(Y.copy())
  D = np.zeros((p, n))
  D[0] = Y


  for j in range(1, p): # Changed loop range to start from 1
    D[j, :(n-j)] = D[j-1, 1:n-(j-1)] - D[j-1, :n-j] # Subtraction from the second element onward

  for l in range(2,p):
    if (abs(D[l][p-3-r[k]])<abs(D[l][p-2-r[k]])):
      r[k]+=1
print(D)
print(r)
X_train = np.array(X_train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

# Define weight matrices from the paper (ensuring correct shapes for TensorFlow)
W1 = np.array([
    [0, 0, 1, -2, 1, 0],
    [0, 1, -2, 1, 0, 0],
    [0, 0, -1, 3, -3, 1],
    [0, -1, 3, -3, 1, 0],
    [-1, 3, -3, 1, 0, 0],
    [0, 0, -1, 2, -1, 0],
    [0, -1, 2, -1, 0, 0],
    [0, 0, 1, -3, 3, -1],
    [0, 1, -3, 3, -1, 0],
    [1, -3, 3, -1, 0, 0],
])
W1 = W1.T  # Transpose to match TensorFlow's shape (6,10)

W2 = np.array([
    [-1, 1, 0, 0, 0, -1, 1, 0, 0, 0],
    [0, 0, -1, 1, 0, 0, 0, -1, 1, 0],
    [0, 0, 0, -1, 1, 0, 0, 0, -1, 1],
    [1, -1, 0, 0, 0, 1, -1, 0, 0, 0],
    [0, 0, 1, -1, 0, 0, 0, 1, -1, 0],
    [0, 0, 0, 1, -1, 0, 0, 0, 1, -1]
])
W2 = W2.T  # Transpose (10,6) → (6,10)

W3 = np.array([
    [1,1,1,0,0,1],
    [1,0,1,0,1,1],
    [-1,1,0,1,0,-1],
    [0,1,0,1,1,1]
])
W3 = W3.T  # Transpose (6,3)

W4 = np.array([
    [1, 0,0,0],
    [0, 1,1,0],
    [0, 0,0,1],
])
W4 = W4.T  # Transpose (3,3)

# Build model
model = Sequential([
    tf.keras.Input(shape=(6,)),  # Input size for stencil points
    Dense(10, activation='relu',
          kernel_initializer=tf.keras.initializers.Constant(W1),
          bias_initializer=tf.keras.initializers.Zeros(),trainable=False), # Added closing parenthesis here
    Dense(6, activation='relu',
          kernel_initializer=tf.keras.initializers.Constant(W2),
          bias_initializer=tf.keras.initializers.Zeros(),trainable=False), # Added closing parenthesis here
    Dense(4, activation='relu',
          kernel_initializer=tf.keras.initializers.Constant(W3),
          bias_initializer=tf.keras.initializers.Zeros(),trainable=False), # Added closing parenthesis here
    Dense(3, activation='softmax',
          kernel_initializer=tf.keras.initializers.Constant(W4),
          bias_initializer=tf.keras.initializers.Zeros(),trainable=False), # Added closing parenthesis here

])

# Compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Use sparse labels (integer form)
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=['sparse_categorical_accuracy']
)

# Check model summary
model.summary()

# Ensure X_train is correctly shaped

print(r)
# Train the model
history = model.fit(
    X_train, r,  # Sparse labels (integer form)
    epochs=50,
    verbose=1, shuffle=False
)
