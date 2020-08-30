import torch

'''
These are step by step implementations of how torch uses backprop
and fwdprop to compute losses and update parameters
'''

N, D_in, H, D_out = 64, 1000, 100, 10
device = torch.device('cpu')
dtype = torch.float

x = torch.randn(N, D_in, device=device, dtype = dtype)
y = torch.randn(N, D_out, device=device, dtype = dtype)

# Weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

lr = 1e-6
epochs = 1000
for e in range(epochs):
    # Forward pass
    h = x.mm(w1)
    h_relu = h.clamp(min=0) # NxH
    y_pred = h_relu.mm(w2)

    # Loss
    L = (y_pred - y).pow(2).sum().item()

    if e % 50 == 0:
        print(e, L)

    # Backprop
    grad_y_pred = 2*(y_pred - y) # NxD_out
    grad_w2 = h_relu.t().mm(grad_y_pred) # HxD_out
    grad_h_relu = grad_y_pred.mm(w2.t()) # NxH
    grad_h_relu[h<0] = 0

    grad_w1 = x.t().mm(grad_h_relu)

    # Update weights
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2



