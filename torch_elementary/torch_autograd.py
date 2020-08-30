import torch

'''
These are step by step implementations of how torch uses backprop
and fwdprop to compute losses and update parameters
'''

N, D_in, H, D_out = 64, 1000, 100, 10
device = torch.device('cpu')
dtype = torch.float

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

lr = 1e-6
epochs = 1000
for e in range(epochs):
    # Forward
    y_pred = x.mm(w1).clamp(min=0).mm(w2) # NxD_out

    # Loss
    L = (y_pred - y).pow(2).sum()
    
    if e % 50 == 0:
        print(e, L.item())
        
    L.backward()

    # Update weights
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()




