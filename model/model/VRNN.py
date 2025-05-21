import torch
import random
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributions as dist
from torch.autograd import Variable


class VRNNSimple(nn.Module):
    def __init__(self, input_size, hidden_size, z_size, num_layers=1):
        super(VRNNSimple, self).__init__()

        self.hidden_size = hidden_size
        self.z_size = z_size
        self.num_layers = num_layers

        # GRU for hidden state transitions
        self.gru = nn.GRU(input_size + z_size, hidden_size, num_layers, batch_first=True)

        # Encoder: approximate posterior q     (z_t | x_t, h_{t-1})
        self.enc_mu = nn.Linear(hidden_size + input_size, z_size)  # mean of z_t
        self.enc_sigma = nn.Linear(hidden_size + input_size, z_size)  # variance of z_t

        # Prior: prior distribution p(z_t | h_{t-1})
        self.prior_mu = nn.Linear(hidden_size, z_size)  # mean of prior z_t
        self.prior_sigma = nn.Linear(hidden_size, z_size)  # variance of prior z_t

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z from q(z|x, h)."""
        std = torch.exp(0.5 * logvar)  # compute standard deviation from log-variance
        eps = torch.randn_like(std)  # random noise
        return mu + eps * std  # sample z_t

    def forward(self, x):
        batch_size, seq_length, input_size = x.size()

        # Initialize hidden state (h_0)
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Iterate through each time step
        for t in range(seq_length):
            x_t = x[:, t, :]

            # Prior: compute p(z_t | h_{t-1})
            prior_mu = self.prior_mu(h[-1])
            prior_logvar = self.prior_sigma(h[-1])

            # Encoder: compute q(z_t | x_t, h_{t-1})
            enc_input = torch.cat([x_t, h[-1]], dim=1)
            z_mu = self.enc_mu(enc_input)
            z_logvar = self.enc_sigma(enc_input)

            # Reparameterize to get z_t
            z_t = self.reparameterize(z_mu, z_logvar)

            # GRU: Update hidden state h_t using x_t and z_t
            gru_input = torch.cat([x_t, z_t], dim=1).unsqueeze(1)
            _, h = self.gru(gru_input, h) # h: tensor of shape (num_layers, batch_size, hidden_size)

        # Return the hidden state of the last time step
        out = h[-1]

        return out

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False, alpha=1.0, beta=1.0):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.alpha = alpha
        self.beta = beta

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
        nn.Linear(x_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.ReLU())
        self.phi_z = nn.Sequential(
        nn.Linear(z_dim, h_dim),
        nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
        nn.Linear(h_dim + h_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
        nn.Linear(h_dim, z_dim),
        nn.Softplus())

        #prior
        self.prior = nn.Sequential(
        nn.Linear(h_dim, h_dim),
        nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
        nn.Linear(h_dim, z_dim),
        nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
        nn.Linear(h_dim + h_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.ReLU())
        self.dec_std = nn.Sequential(
        nn.Linear(h_dim, x_dim),
        nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
        nn.Linear(h_dim, x_dim),
        nn.Sigmoid())

        #recurrence
        self.gru = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

    def forward(self, x):

        elbo_loss = 0

        x = x.reshape(x.size(1), x.size(0), -1)
        # after reshape, x shape (seq_length, batch_size, x_dim)

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(x.device)

        for t in range(x.size(0)):

            phi_x_t = self.phi_x(x[t]).to(x.device)

            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self.reparameterized_sample(enc_mean_t, enc_std_t).to(x.device)
            phi_z_t = self.phi_z(z_t) # z_t shape (batch_size, z_dim)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            #recurrence
            output, h = self.gru(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            # output shape (1 (seq_length), batch_size, hidden_size)
            # h shape (num_layers, batch_size, hidden_size)

            #computing losses
            elbo_loss += self.elbo_loss(x[t], dec_mean_t, enc_mean_t, enc_std_t, prior_mean_t, 
                                        prior_std_t, alpha=self.alpha, beta=self.beta)
        h = h.squeeze(0)

        return elbo_loss/(x.size(1)*x.size(0)), z_t, h

    # def _reparameterized_sample(self, mean, std):
    #     """using std to sample"""
    #     eps = torch.FloatTensor(std.size()).normal_().to(std.device) # eps means epsilon, a random noise
    #     return eps.mul(std).add_(mean)  # z = eps*std + mean

    # def kld_gauss(self, mean_1, std_1, mean_2, std_2):
    #     """Using std to compute KLD"""
    #     kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
    #         (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
    #         std_2.pow(2) - 1)
    #     return 0.5 * torch.sum(kld_element)
    
    def reparameterized_sample(self, mean, std):
        """Using torch.distributions.Normal for reparameterized sampling."""

        # Create a normal distribution with the given mean and std (standard deviation)
        normal_dist = dist.Normal(mean, std)
        # Sample from the distribution using rsample (reparameterization trick)
        z = normal_dist.rsample().to(std.device)
        return z

    def kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using torch.distributions.kl.kl_divergence to compute KLD."""
        dist_1 = dist.Normal(mean_1, std_1)
        dist_2 = dist.Normal(mean_2, std_2)

        return dist.kl_divergence(dist_1, dist_2).sum()
        
    def elbo_loss(self, x, dec_mean_t, enc_mean_t, enc_std_t, prior_mean_t, prior_std_t, alpha=1.0, beta=1.0):
        # Reconstruction loss: using MSE (Mean Squared Error) as a substitute for log likelihood
        recon_loss = alpha * torch.mean((x - dec_mean_t) ** 2)
    
        # KL divergence: calculating the difference between the posterior and prior distributions
        kl_loss = beta * self.kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

        # Final ELBO loss: combining reconstruction loss and KL divergence
        return recon_loss + kl_loss