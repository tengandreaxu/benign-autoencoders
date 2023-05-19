import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(Generator, self).__init__()
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim) -> None:
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.encoder = Encoder(x_dim, h_dim1, h_dim2, z_dim)
        # decoder part
        self.generator = Generator(x_dim, h_dim1, h_dim2, z_dim)

    def encode(self, x):
        h = F.relu(self.encoder.fc1(x))
        h = F.relu(self.encoder.fc2(h))
        return self.encoder.fc31(h), self.encoder.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.generator.fc4(z))
        h = F.relu(self.generator.fc5(h))
        return F.sigmoid(self.generator.fc6(h))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def set_generator_not_trainable(self):
        for param in self.generator.parameters():
            param.requires_grad = False

    def set_generator_trainable(self):
        for param in self.generator.parameters():
            param.requires_grad = True

    def set_encoder_not_trainable(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def set_encoder_trainable(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
