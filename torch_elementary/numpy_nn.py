import numpy as np

'''
Implementation of a very naive MLP with Relu and backprop
'''

N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

lr = 1e-6
epochs = 1000
for e in range(epochs):
    # Forward pass
    h = x.dot(w1) # NxH
    h_relu = np.maximum(h, 0) # NxH
    y_pred = h_relu.dot(w2) # NxD_out

    # Loss
    L = np.square(y_pred - y).sum()
    if e % 50 == 0:
        print(e, L)

    # Backprop
    grad_y_pred = 2*(y_pred - y) # Delta 3 Del_out
    grad_w2 = h_relu.T.dot(grad_y_pred) # HxD_out
    grad_h_relu = grad_y_pred.dot(w2.T) # NxH
    grad_h_relu[h<0] = 0 # multiply by g_prime Del_hidden

    grad_w1 = x.T.dot(grad_h_relu)

    # Update weights
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2


