# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import numpy as np
import itertools as it
import math
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from kaiwu.torch_plugin import RestrictedBoltzmannMachine, QVAE
from kaiwu.classical import SimulatedAnnealingOptimizer


EPS = 1e-10
ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}


class QBMEncoder(nn.Module):
    """Encoder for QBM-VAE"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        act_fn,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.weight_decay = weight_decay
        self.act_fn = act_fn

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x

    def get_weight_decay(self):
        return self.weight_decay * (
            torch.sum(self.fc1.weight**2) + torch.sum(self.fc2.weight**2)
        )


class QBMDecoder(nn.Module):
    """Decoder for QBM-VAE"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        act_fn,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.weight_decay = weight_decay
        self.act_fn = act_fn

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z):
        z = self.fc1(z)
        z = self.norm1(z)
        z = self.act_fn(z)
        z = self.fc2(z)
        return z

    def get_weight_decay(self):
        return self.weight_decay * (
            torch.sum(self.fc1.weight**2) + torch.sum(self.fc2.weight**2)
        )


class BiQBMVAE(nn.Module):
    """Bidirectional QBM-VAE for compound-protein interaction"""
    def __init__(self, k, compound_encoder_structure, protein_encoder_structure, likelihood, act_fn,
                 num_visible=None, num_hidden=None, dist_beta=10.0, device=torch.device("cpu")):
        super(BiQBMVAE, self).__init__()

        self.k = k
        self.likelihood = likelihood
        self.act_fn = ACT.get(act_fn, nn.ReLU())
        self.device = device
        self.dist_beta = dist_beta

        # Set default num_visible and num_hidden if not provided
        if num_visible is None:
            num_visible = k // 2
        if num_hidden is None:
            num_hidden = k - num_visible

        self.num_visible = num_visible
        self.num_hidden = num_hidden

        # Initialize latent representations
        self.mu_theta = torch.zeros((protein_encoder_structure[0], k))  # n_compound * k
        self.mu_beta = torch.zeros((compound_encoder_structure[0], k))  # n_protein * k
        self.theta = torch.randn(protein_encoder_structure[0], k) * 0.01
        self.beta = torch.randn(compound_encoder_structure[0], k) * 0.01
        torch.nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))

        # Compound QVAE components
        compound_hidden_dim = compound_encoder_structure[1] if len(compound_encoder_structure) > 1 else 40
        self.compound_encoder = QBMEncoder(
            input_dim=compound_encoder_structure[0],
            hidden_dim=compound_hidden_dim,
            latent_dim=k,
            act_fn=self.act_fn
        )
        self.compound_decoder = QBMDecoder(
            input_dim=k,
            hidden_dim=compound_hidden_dim,
            latent_dim=compound_encoder_structure[0],
            act_fn=self.act_fn
        )

        # Protein QVAE components
        protein_hidden_dim = protein_encoder_structure[1] if len(protein_encoder_structure) > 1 else 40
        self.protein_encoder = QBMEncoder(
            input_dim=protein_encoder_structure[0],
            hidden_dim=protein_hidden_dim,
            latent_dim=k,
            act_fn=self.act_fn
        )
        self.protein_decoder = QBMDecoder(
            input_dim=k,
            hidden_dim=protein_hidden_dim,
            latent_dim=protein_encoder_structure[0],
            act_fn=self.act_fn
        )

        # Shared Boltzmann Machine
        self.rbm = RestrictedBoltzmannMachine(
            num_visible=num_visible,
            num_hidden=num_hidden,
            h_range=[-1, 1],
            j_range=[-1, 1],
            device=device
        )

        # Sampler for BM
        self.sampler = SimulatedAnnealingOptimizer(alpha=0.95)

        # Mean values (will be set during training)
        self.register_buffer('compound_mean', torch.zeros(compound_encoder_structure[0]))
        self.register_buffer('protein_mean', torch.zeros(protein_encoder_structure[0]))

    def encode_compound(self, x):
        """Encode compound using QBM encoder"""
        return self.compound_encoder(x - self.compound_mean)

    def encode_protein(self, x):
        """Encode protein using QBM encoder"""
        return self.protein_encoder(x - self.protein_mean)

    def decode_compound(self, theta, beta):
        """Decode compound from latent representations"""
        h = theta.mm(beta.t())
        return torch.sigmoid(h)

    def decode_protein(self, theta, beta):
        """Decode protein from latent representations"""
        h = beta.mm(theta.t())
        return torch.sigmoid(h)

    def forward(self, x, compound=True, beta=None, theta=None):
        """Forward pass for BiQBMVAE"""
        if compound:
            q_logits = self.encode_compound(x)
            # For compatibility, return encoded representation and reconstruction
            theta_encoded = q_logits
            x_recon = self.decode_compound(theta_encoded, beta)
            return theta_encoded, x_recon, q_logits, q_logits
        else:
            q_logits = self.encode_protein(x)
            beta_encoded = q_logits
            x_recon = self.decode_protein(theta, beta_encoded)
            return beta_encoded, x_recon, q_logits, q_logits

    def loss(self, x, x_, mu, std, kl_beta):
        """Compute loss for BiQBMVAE"""
        # Likelihood
        ll_choices = {
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -(x - x_) ** 2,
            "pois": x * torch.log(x_ + EPS) - x_,
        }

        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))
        ll = torch.sum(ll, dim=1)

        # For QBM-VAE, we use a simplified KL term based on the encoded representation
        kld = 0.5 * torch.sum(mu.pow(2), dim=1)

        return torch.mean(kl_beta * kld - ll)


class BiVAE(nn.Module):
    def __init__(self, k, compound_encoder_structure, protein_encoder_structure, likelihood, act_fn):
        super(BiVAE, self).__init__()
        self.mu_theta = torch.zeros((protein_encoder_structure[0], k))  # n_compound * k
        self.mu_beta = torch.zeros((compound_encoder_structure[0], k))  # n_protein * k
        self.theta = torch.randn(protein_encoder_structure[0], k) * 0.01
        self.beta = torch.randn(compound_encoder_structure[0], k) * 0.01
        torch.nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))

        self.likelihood = likelihood
        self.act_fn = ACT.get(act_fn, None)

        # compound encoder
        self.compound_encoder = nn.Sequential()
        for i in range(len(compound_encoder_structure) - 1):
            self.compound_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(compound_encoder_structure[i], compound_encoder_structure[i + 1]),
            )
            self.compound_encoder.add_module("act{}".format(i), self.act_fn)
        self.compound_mu = nn.Linear(compound_encoder_structure[-1], k)  # mu
        self.compound_std = nn.Linear(compound_encoder_structure[-1], k)

        # protein Encoder
        self.protein_encoder = nn.Sequential()
        for i in range(len(protein_encoder_structure) - 1):
            self.protein_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(protein_encoder_structure[i], protein_encoder_structure[i + 1]),
            )
            self.protein_encoder.add_module("act{}".format(i), self.act_fn)
        self.protein_mu = nn.Linear(protein_encoder_structure[-1], k)  # mu
        self.protein_std = nn.Linear(protein_encoder_structure[-1], k)

    def encode_compound(self, x):
        h = self.compound_encoder(x)
        return self.compound_mu(h), torch.sigmoid(self.compound_std(h))

    def encode_protein(self, x):
        h = self.protein_encoder(x)
        return self.protein_mu(h), torch.sigmoid(self.protein_std(h))

    def decode_compound(self, theta, beta):
        h = theta.mm(beta.t())
        return torch.sigmoid(h)

    def decode_protein(self, theta, beta):
        h = beta.mm(theta.t())
        return torch.sigmoid(h)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x, compound=True, beta=None, theta=None):
        if compound:
            mu, std = self.encode_compound(x)
            theta = self.reparameterize(mu, std)
            return theta, self.decode_compound(theta, beta), mu, std
        else:
            mu, std = self.encode_protein(x)
            beta = self.reparameterize(mu, std)
            return beta, self.decode_protein(theta, beta), mu, std

    def loss(self, x, x_, mu, std, kl_beta):
        # Likelihood
        ll_choices = {
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -(x - x_) ** 2,
            "pois": x * torch.log(x_ + EPS) - x_,
        }

        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))
        ll = torch.sum(ll, dim=1)

        # KL term
        kld = -0.5 * (1 + 2.0 * torch.log(std) - mu.pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(kl_beta * kld - ll)


def learn_qbmvae(biqbmvae, data_matrix, epochs, batch_size, lr, beta_kl, device=torch.device("cpu"), dtype=torch.float32):
    """Training function for BiQBMVAE model"""
    # Optimizers for compound and protein sides
    compound_params = it.chain(
        biqbmvae.compound_encoder.parameters(),
        biqbmvae.compound_decoder.parameters()
    )
    protein_params = it.chain(
        biqbmvae.protein_encoder.parameters(),
        biqbmvae.protein_decoder.parameters()
    )
    rbm_params = biqbmvae.rbm.parameters()

    c_optimizer = torch.optim.Adam(params=compound_params, lr=lr)
    p_optimizer = torch.optim.Adam(params=protein_params, lr=lr)
    rbm_optimizer = torch.optim.Adam(params=rbm_params, lr=lr)

    x = data_matrix
    tx = x.T
    c_idx = np.arange(x.shape[0])
    p_idx = np.arange(tx.shape[0])

    # Calculate mean values
    biqbmvae.compound_mean = torch.tensor(x.mean(axis=0), dtype=dtype, device=device)
    biqbmvae.protein_mean = torch.tensor(tx.mean(axis=0), dtype=dtype, device=device)

    # Move theta and beta to device
    biqbmvae.theta = biqbmvae.theta.to(device)
    biqbmvae.beta = biqbmvae.beta.to(device)
    biqbmvae.mu_theta = biqbmvae.mu_theta.to(device)
    biqbmvae.mu_beta = biqbmvae.mu_beta.to(device)

    best_biqbmvae = None
    best_loss = math.inf

    for epoch in range(epochs):
        # Protein side training
        p_sum_loss = 0
        for i in range(math.ceil(tx.shape[0] / batch_size)):
            p_ids = p_idx[i * batch_size:(i + 1) * batch_size]
            p_batch = tx[p_ids, :]
            p_batch = torch.tensor(p_batch, dtype=dtype, device=device)

            beta, p_batch_, p_mu, p_std = biqbmvae(p_batch, compound=False, theta=biqbmvae.theta)
            p_loss = biqbmvae.loss(p_batch, p_batch_, p_mu, p_std, beta_kl)

            p_optimizer.zero_grad()
            rbm_optimizer.zero_grad()
            p_loss.backward()
            p_optimizer.step()
            rbm_optimizer.step()

            p_sum_loss += p_loss.item()
            with torch.no_grad():
                beta, _, p_mu, _ = biqbmvae(p_batch, compound=False, theta=biqbmvae.theta)
                biqbmvae.beta.data[p_ids] = beta.data
                biqbmvae.mu_beta.data[p_ids] = p_mu.data

        # Compound side training
        c_sum_loss = 0
        for i in range(math.ceil(x.shape[0] / batch_size)):
            c_ids = c_idx[i * batch_size:(i + 1) * batch_size]
            c_batch = x[c_ids, :]
            c_batch = torch.tensor(c_batch, dtype=dtype, device=device)

            theta, c_batch_, c_mu, c_std = biqbmvae(c_batch, compound=True, beta=biqbmvae.beta)
            c_loss = biqbmvae.loss(c_batch, c_batch_, c_mu, c_std, beta_kl)

            c_optimizer.zero_grad()
            rbm_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()
            rbm_optimizer.step()

            c_sum_loss += c_loss.item()
            with torch.no_grad():
                theta, _, c_mu, _ = biqbmvae(c_batch, compound=True, beta=biqbmvae.beta)
                biqbmvae.theta.data[c_ids] = theta.data
                biqbmvae.mu_theta.data[c_ids] = c_mu.data

        total_loss = p_sum_loss + c_sum_loss
        if total_loss < best_loss:
            best_loss = total_loss
            best_biqbmvae = biqbmvae

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')

    # Infer final mu_beta and mu_theta
    with torch.no_grad():
        for i in range(math.ceil(tx.shape[0] / batch_size)):
            p_ids = p_idx[i * batch_size:(i + 1) * batch_size]
            p_batch = tx[p_ids, :]
            p_batch = torch.tensor(p_batch, dtype=dtype, device=device)
            _, _, p_mu, _ = best_biqbmvae(p_batch, compound=False, theta=best_biqbmvae.theta)
            best_biqbmvae.mu_beta.data[p_ids] = p_mu.data

        for i in range(math.ceil(x.shape[0] / batch_size)):
            c_ids = c_idx[i * batch_size:(i + 1) * batch_size]
            c_batch = x[c_ids, :]
            c_batch = torch.tensor(c_batch, dtype=dtype, device=device)
            _, _, c_mu, _ = best_biqbmvae(c_batch, compound=True, beta=best_biqbmvae.beta)
            best_biqbmvae.mu_theta.data[c_ids] = c_mu.data

    return best_biqbmvae


def learn(bivae, data_matrix, epochs, batch_size, lr, beta_kl, device=torch.device("cpu"),dtype=torch.float32):
    compound_params = it.chain(bivae.compound_encoder.parameters(),
                               bivae.compound_mu.parameters(),
                               bivae.compound_std.parameters())
    protein_params = it.chain(bivae.protein_encoder.parameters(),
                              bivae.protein_mu.parameters(),
                              bivae.protein_std.parameters())
    c_optimizer = torch.optim.Adam(params=compound_params, lr=lr)
    p_optimizer = torch.optim.Adam(params=protein_params, lr=lr)
    x = data_matrix
    tx = x.T
    c_idx = np.arange(x.shape[0])
    p_idx = np.arange(tx.shape[0])
    best_bivae = None
    best_loss = math.inf
    for epoch in range(epochs):
        # protein side
        p_sum_loss = 0
        for i in range(math.ceil(tx.shape[0] / batch_size)):
            p_ids = p_idx[i * batch_size:(i + 1) * batch_size]
            p_batch = tx[p_ids, : ]
            p_batch = torch.tensor(p_batch, dtype=dtype, device=device)
            beta, p_batch_, p_mu, p_std = bivae(p_batch, compound=False, theta=bivae.theta)
            p_loss = bivae.loss(p_batch, p_batch_, p_mu, p_std, beta_kl)
            p_optimizer.zero_grad()
            p_loss.backward()
            p_optimizer.step()

            p_sum_loss += p_loss.item()
            beta, _, p_mu, _ =  bivae(p_batch, compound=False, theta=bivae.theta)
            bivae.beta.data[p_ids] = beta.data
            bivae.mu_beta.data[p_ids] = p_mu.data

        # compound side
        c_sum_loss = 0
        for i in range(math.ceil(x.shape[0] / batch_size)):
            c_ids = c_idx[i * batch_size:(i + 1) * batch_size]
            c_batch = x[c_ids, :]
            c_batch = torch.tensor(c_batch, dtype=dtype, device=device)
            theta, c_batch_, c_mu, c_std = bivae(c_batch, compound=True, beta=bivae.beta)
            c_loss = bivae.loss(c_batch, c_batch_, c_mu, c_std, beta_kl)
            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()

            c_sum_loss += c_loss.item()
            theta, _, c_mu, _ = bivae(c_batch, compound=True, beta=bivae.beta)
            bivae.theta.data[c_ids] = theta.data
            bivae.mu_theta.data[c_ids] = c_mu.data

        if p_sum_loss+c_sum_loss < best_loss :
            best_loss = p_sum_loss+c_sum_loss
            best_bivae = bivae



    # infer mu_beta
    for i in range(math.ceil(tx.shape[0] / batch_size)):
        p_ids = p_idx[i * batch_size:(i + 1) * batch_size]
        p_batch = tx[p_ids, :]
        p_batch = torch.tensor(p_batch, dtype=dtype, device=device)
        beta, _, p_mu, _ = best_bivae(p_batch, compound=False, theta=bivae.theta)
        best_bivae.mu_beta.data[p_ids] = p_mu.data

    # infer mu_theta
    for i in range(math.ceil(x.shape[0] / batch_size)):
        c_ids = c_idx[i * batch_size:(i + 1) * batch_size]
        c_batch = x[c_ids, :]
        c_batch = torch.tensor(c_batch, dtype=dtype, device=device)
        theta, _, c_mu, _ = best_bivae(c_batch, compound=True, beta=bivae.beta)
        best_bivae.mu_theta.data[c_ids] = c_mu.data

    return best_bivae


class ApplyNodeFunc(nn.Module):
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.InstanceNorm1d(self.mlp.output_dim, affine=False, track_running_stats=False)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.InstanceNorm1d(hidden_dim, affine=False, track_running_stats=False))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):

        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.InstanceNorm1d(hidden_dim, affine=False, track_running_stats=False))

        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)


        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = conved.permute(0,2,1)
        return conved


class BiBAECPI(nn.Module):
    def __init__(self, bivae, n_atom, n_amino, params):
        super(BiBAECPI, self).__init__()
        comp_dim, prot_dim, gin_layers, num_mlp_layers, dropout, alpha, window, layer_cnn, latent_dim, hidden_dim, k,\
            neighbor_pooling_type, graph_pooling_type = params.comp_dim, params.prot_dim, params.gin_layers, \
            params.num_mlp_layers, params.dropout, params.alpha, params.window, params.layer_cnn, params.latent_dim, \
            params.hidden_dim, params.k, params.neighbor_pooling_type, params.graph_pooling_type

        self.embedding_layer_atom = nn.Embedding(n_atom + 1, comp_dim)  # nn.Embedding(n,m)
        self.embedding_layer_amino = nn.Embedding(n_amino + 1, prot_dim)

        self.bivae = bivae
        self.dropout = dropout
        self.alpha = alpha
        self.layer_cnn = layer_cnn

        self.gin = GIN(num_layers=gin_layers, num_mlp_layers=num_mlp_layers, input_dim=comp_dim, hidden_dim=hidden_dim, output_dim=latent_dim, final_dropout=dropout,
          learn_eps=False, graph_pooling_type=graph_pooling_type, neighbor_pooling_type=neighbor_pooling_type)

        self.encoder = Encoder(prot_dim, hid_dim=latent_dim, n_layers=layer_cnn, kernel_size=2 * window + 1, dropout=dropout, device=torch.device('cuda'))

        self.fp0 = nn.Parameter(torch.empty(size=(1024, latent_dim)))
        nn.init.xavier_uniform_(self.fp0, gain=1.414)
        self.fp1 = nn.Parameter(torch.empty(size=(latent_dim, k)))
        nn.init.xavier_uniform_(self.fp1, gain=1.414)

        self.trans_comp = nn.Linear(latent_dim, k)
        self.trans_pro = nn.Linear(latent_dim, k)

        self.out = nn.Linear(k*3, 2)
        self.drop_out = nn.Dropout(p=self.dropout)

    def comp_gin(self, atoms, adj, device):
        # GIN
        atoms_vector = self.embedding_layer_atom(atoms)
        adj = np.array(adj.cpu())
        a = np.nonzero(adj)
        g = dgl.graph(a).to(device)
        atoms_vector = self.gin(g, atoms_vector)

        return atoms_vector

    def prot_encoder(self, amino):
        amino_vector = self.embedding_layer_amino(amino)
        amino_vector = self.encoder(amino_vector)
        amino_vector = F.leaky_relu(amino_vector, self.alpha)
        return amino_vector  # (batch_size, lenth, dim)

    def forward(self, atoms, adjacency, amino, fps, c_id, p_id, device):
        atoms_vector = self.comp_gin(atoms, adjacency, device)

        amino_vector = self.prot_encoder(amino)
        atoms_vector = atoms_vector.squeeze(0)
        amino_vector = torch.sum(amino_vector, dim=1).squeeze(0)

        atoms_vector = self.trans_comp(atoms_vector)
        amino_vector = self.trans_pro(amino_vector)

        theta_c = self.bivae.mu_theta[c_id].to(device)
        beta_p = self.bivae.mu_beta[p_id].to(device)
        fea_com = theta_c * atoms_vector
        fea_pro = beta_p * amino_vector
        fea_com = F.leaky_relu(fea_com, 0.1)
        fea_pro = F.leaky_relu(fea_pro, 0.1)

        fps_vector = F.leaky_relu(torch.matmul(fps, self.fp0), 0.1)
        fps_vector = F.leaky_relu(torch.matmul(fps_vector, self.fp1), 0.1)

        fusion_feature = torch.cat((fea_com, fea_pro, fps_vector))
        result = self.out(fusion_feature)
        result = result.reshape(1,2)
        return result

