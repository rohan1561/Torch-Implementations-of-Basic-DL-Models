import torch

N, D_in, H, D_out = 64, 1000, 100, 10
device = torch.device('cpu')
dtype = torch.float

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Model
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),)

loss = torch.nn.MSELoss(reduction='sum')
lr = 1e-4
epochs = 1000
for e in range(epochs):
    # Forward
    y_pred = model(x)

    L = loss(y_pred, y)

    if e % 50 == 0:
        print(e, L.item())
    model.zero_grad()
    L.backward()

    with torch.no_grad():
        for p in model.parameters():
            p -= lr*p.grad
 

