import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import os

'''
A basic denoising autoencoder
'''
batch_size = 128
image_size = 28
num_epochs = 5
lr = 0.001
dataloader = torch.utils.data.DataLoader(
        MNIST('/files/', train=True, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
                             batch_size=batch_size, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def add_noise(images):
    noise = torch.randn(images.size())
    images = images + noise
    return images

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear((image_size**2), 512),
                nn.ReLU(True),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Linear(128, 50),
                )
        self.decoder = nn.Sequential(
                nn.Linear(50, 128),
                nn.ReLU(True),
                nn.Linear(128, 512),
                nn.ReLU(True),
                nn.Linear(512, (image_size**2)),
                )

    def forward(self, input):
        return (self.decoder(self.encoder(input.view(input.shape[0], -1))))

net = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
epochs = 5

for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        net.zero_grad()
        images = data[0]
        images = Variable(images)
        images = add_noise(images)
        images = images.cuda()
        output = net(images)
        loss = criterion(output, images.view(images.shape[0], -1))
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('loss = {}, epoch = {}'.format(loss.item(), epoch))
        
