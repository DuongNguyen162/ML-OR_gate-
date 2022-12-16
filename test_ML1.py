import numpy as np

# Active func
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# daoham of activation function
def sigmoid_deri(x):
  return sigmoid(x) * (1 - sigmoid(x))

# OR gate
def or_gate(a, b, w0, w1, w2):
  weighted_sum = w0 + w1 * a + w2 * b
  return sigmoid(weighted_sum)

# Training func
def train(a, b, c, w0, w1, w2, learning_rate):
  # tinh output
  output = or_gate(a, b, w0, w1, w2)

  # Cal error
  error = c - output # error ~~ 0 

  # Calculate gradient of error with respect to weights
  gradient_w0 = error * sigmoid_deri(output)
  gradient_w1 = error * sigmoid_deri(output) * a
  gradient_w2 = error * sigmoid_deri(output) * b

  # Update weights
  w0 += learning_rate * gradient_w0
  w1 += learning_rate * gradient_w1
  w2 += learning_rate * gradient_w2

  return w0, w1, w2

# Init weight
w0 = -0.2
w1 = 0.5
w2 = 0.5

# Set rate hoc
learning_rate = 0.15

# Set number train iterations
num_i = 1000

# Training OR_gate
for i in range(num_i):
  w0, w1, w2 = train(0, 0, 0, w0, w1, w2, learning_rate)
  w0, w1, w2 = train(0, 1, 1, w0, w1, w2, learning_rate)
  w0, w1, w2 = train(1, 0, 1, w0, w1, w2, learning_rate)
  w0, w1, w2 = train(1, 1, 1, w0, w1, w2, learning_rate)

# Test OR_gate
print(or_gate(0, 0, w0, w1, w2)) # Output: 0
print(or_gate(0, 1, w0, w1, w2)) # Output: 1
print(or_gate(1, 0, w0, w1, w2)) # Output: 1
print(or_gate(1, 1, w0, w1, w2)) # Output: 1