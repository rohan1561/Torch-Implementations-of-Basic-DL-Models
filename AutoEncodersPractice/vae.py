import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import os

'''
A basic variational autoencoder with the KLD component of the loss
'''

dataroot = os.getcwd()
workers = 2
batch_size = 128
image_size = 64
nc = 3
hid_dim = 100
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,\
        num_workers=workers)

device = torch.device("cuda:0" if torch.cuda.is_available() and ngpu > 0 else "cpu")

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Define the encoder
        hidden_dim = [32, 64, 128, 256, 512]
        modules = []
        in_ch = nc
        for h in hidden_dim:
            modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, h, 3, 2, 1),
                        nn.BatchNorm2d(h),
                        nn.LeakyReLU(),
                        )
                    )
            in_ch = h
        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(hidden_dim[-1]*4, hid_dim)
        self.logvar = nn.Linear(hidden_dim[-1]*4, hid_dim)

        # Define the decoder
        self.pre_decode = nn.Linear(hid_dim, hidden_dim[-1]*4)
        hidden_dim.reverse()
        hidden_dim = hidden_dim[:-1]
        modules = []
        for h in hidden_dim:
            modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(h, int(h/2), 3, 2, 1, 1),
                        nn.BatchNorm2d(int(h/2)),
                        nn.LeakyReLU(),
                        )
                    )
        modules.append(
                nn.Sequential(
                nn.ConvTranspose2d(32, 32, 3, 2, 1, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, nc, kernel_size=3, padding=1),
                nn.Tanh(),
                )
            )
            
        self.decoder = nn.Sequential(*modules)


    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std)
        return std*epsilon + mu

    def forward(self, x):
        # Encode
        z = torch.flatten(self.encoder(x), start_dim=1)
        mu = self.mu(z)
        logvar = self.logvar(z)
        repar_vec = self.reparam(mu, logvar)

        # Decode
        decoder_inp = self.pre_decode(repar_vec).view(-1, 512, 2, 2)
        decoded = self.decoder(decoder_inp)
        return decoded, x, mu, logvar


model = VAE()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
loss = nn.MSELoss()
epochs = 10

def loss(*args):
    recon = args[0]
    orig = args[1]
    mu = args[2]
    logvar = args[3]

    mse = loss(recon, orig)
    kld = torch.mean(-0.5*torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1),\
            dim=0)
    print(kld)
    return mse + kld

for i in range(epochs):
    for i, data in enumerate(dataloader):
        model.zero_grad()
        data = data[0].to(device)
        forwards = model(data)
        print(forwards[2], 'ssssssssss')
        l = loss(*forwards).to(device)
        print(l)
        l.backward()
        optimizer.step()

        


