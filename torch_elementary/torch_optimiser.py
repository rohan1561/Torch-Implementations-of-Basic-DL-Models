import torch

'''
These are step by step implementations of how torch uses backprop
and fwdprop to compute losses and update parameters
'''

class MyModel(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10
device = torch.device('cpu')
dtype = torch.float

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

model = MyModel(D_in, H, D_out)
loss = torch.nn.MSELoss(reduction='sum')

lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 600

for e in range(epochs):
    y_pred = model.forward(x)

    L = loss(y_pred, y)
    if e % 50 == 0:
        print(e, L.item())

    optimizer.zero_grad()
    L.backward()
    optimizer.step()

