import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn.pytorch import GINConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import EGATConv
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from torch.nn.parameter import Parameter



class Multi_Omics_DDM(nn.Module):
    def __init__(
            self,
            in_dim_omics_1: int,
            in_dim_omics_2: int,
            num_hidden_omics_1: int,
            num_hidden_omics_2: int,
            num_layers: int = 1,
            nhead: int = 2,
            activation: str = 'prelu',
            feat_drop: float = 0.2,
            attn_drop: float = 0.2,
            norm: str = 'layernorm',
            alpha_l: float = 2,
            beta_schedule: str = 'linear',
            beta_1: float = 0.0001,
            beta_T: float = 0.02,
            T: int = 1000,
    ):
        super(Multi_Omics_DDM, self).__init__()
        self.in_dim_omics_1 = in_dim_omics_1
        self.in_dim_omics_2 = in_dim_omics_2
        self.num_hidden_omics_1 = num_hidden_omics_1
        self.num_hidden_omics_2 = num_hidden_omics_2

        self.DDM_omics_1 = DDM(in_dim=self.in_dim_omics_1, num_hidden=self.num_hidden_omics_1, num_layers=num_layers,
                               nhead=nhead, activation=activation, feat_drop=feat_drop, attn_drop=attn_drop,
                               norm=norm, alpha_l=alpha_l, beta_schedule=beta_schedule, beta_1=beta_1,
                               beta_T=beta_T, T=T)

        self.DDM_omics_2 = DDM(in_dim=self.in_dim_omics_2, num_hidden=self.num_hidden_omics_2, num_layers=num_layers,
                               nhead=nhead, activation=activation, feat_drop=feat_drop, attn_drop=attn_drop,
                               norm=norm, alpha_l=alpha_l, beta_schedule=beta_schedule, beta_1=beta_1,
                               beta_T=beta_T, T=T)

        """
        self.Decoder_omics_1 = DDM(in_dim=self.num_hidden_omics_1, num_hidden=self.in_dim_omics_1, num_layers=num_layers,
                               nhead=nhead, activation=activation, feat_drop=feat_drop, attn_drop=attn_drop,
                               norm=norm, alpha_l=alpha_l, beta_schedule=beta_schedule, beta_1=beta_1,
                               beta_T=beta_T, T=T)

        self.Decoder_omics_2 = DDM(in_dim=self.num_hidden_omics_2, num_hidden=self.in_dim_omics_2, num_layers=num_layers,
                                   nhead=nhead, activation=activation, feat_drop=feat_drop, attn_drop=attn_drop,
                                   norm=norm, alpha_l=alpha_l, beta_schedule=beta_schedule, beta_1=beta_1,
                                   beta_T=beta_T, T=T)
        """

        self.atten_cross = AttentionLayer(self.num_hidden_omics_1, self.num_hidden_omics_2)

        self.dec_cross_omics_1 = Cross_Decoder(self.num_hidden_omics_1, self.in_dim_omics_1)
        self.dec_cross_omics_2 = Cross_Decoder(self.num_hidden_omics_2, self.in_dim_omics_2)



    def forward(self, dgl_omics_1, dgl_omics_2):

        """
        rec_omics_1, emb_omics_1, loss_rec_1 = self.DDM_omics_1(dgl_omics_1, dgl_omics_1.ndata['feat'])
        rec_omics_2, emb_omics_2, loss_rec_2 = self.DDM_omics_2(dgl_omics_2, dgl_omics_2.ndata['feat'])
        """

        loss_rec_1 = self.DDM_omics_1(dgl_omics_1, dgl_omics_1.ndata['feat'])
        loss_rec_2 = self.DDM_omics_2(dgl_omics_2, dgl_omics_2.ndata['feat'])
        emb_omics_1 = self.DDM_omics_1.embed(dgl_omics_1, dgl_omics_1.ndata['feat'], 100)
        emb_omics_2 = self.DDM_omics_2.embed(dgl_omics_2, dgl_omics_2.ndata['feat'], 100)

        # between-modality attention aggregation layer
        emb_combined, alpha_omics_1_2 = self.atten_cross(emb_omics_1, emb_omics_2)

        # reverse the integrated representation back into the original expression space with modality-specific decoder

        comb_recon_omics_1 = self.dec_cross_omics_1(emb_combined, dgl_omics_1.adj().to_dense())
        comb_recon_omics_2 = self.dec_cross_omics_2(emb_combined, dgl_omics_2.adj().to_dense())

        """
        # construct dgl object with combined features
        dgl_comb = dgl.DGLGraph().to(device=torch.device('cuda:0'))
        dgl_comb.add_nodes(dgl_omics_1.number_of_nodes())
        dgl_comb.add_edges(dgl_omics_1.edges()[0], dgl_omics_1.edges()[1])
        dgl_comb.ndata['feat'] = emb_combined

        rec_comb_loss_1 = self.Decoder_omics_1(dgl_comb, dgl_comb.ndata['feat'])
        comb_recon_omics_1 = self.Decoder_omics_1.embed(dgl_comb, dgl_comb.ndata['feat'], 100)
        rec_comb_loss_2 = self.Decoder_omics_2(dgl_comb, dgl_comb.ndata['feat'])
        comb_recon_omics_2 = self.Decoder_omics_2.embed(dgl_comb, dgl_comb.ndata['feat'], 100)
        """


        results = {'emb_omics_1': emb_omics_1,
                   'emb_omics_2': emb_omics_2,
                   'emb_combined': emb_combined,
                   # 'rec_omics_1': rec_omics_1,
                   # 'rec_omics_2': rec_omics_2,
                   'comb_recon_omics_1': comb_recon_omics_1,
                   'comb_recon_omics_2': comb_recon_omics_2,
                   'loss_rec_omics_1': loss_rec_1,
                   'loss_rec_omics_2': loss_rec_2,
                   # 'emb_latent_omics1_across_recon': emb_latent_omics1_across_recon,
                   # 'emb_latent_omics2_across_recon': emb_latent_omics2_across_recon,
                   # 'alpha_omics1': alpha_omics1,
                   # 'alpha_omics2': alpha_omics2,
                   'alpha': alpha_omics_1_2
                   }
        return results



class DDM(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            norm: str,
            alpha_l: float = 2,
            beta_schedule: str = 'linear',
            beta_1: float = 0.0001,
            beta_T: float = 0.02,
            T: int = 1000,
            **kwargs

    ):
        super(DDM, self).__init__()
        self.T = T
        beta = get_beta_schedule(beta_schedule, beta_1, beta_T, T)
        self.register_buffer(
            'betas', beta
        )
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar)
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
        )

        self.alpha_l = alpha_l
        assert num_hidden % nhead == 0
        self.net = Denoising_Unet(in_dim=in_dim,
                                  num_hidden=num_hidden,
                                  out_dim=in_dim,
                                  num_layers=num_layers,
                                  nhead=nhead,
                                  activation=activation,
                                  feat_drop=feat_drop,
                                  attn_drop=attn_drop,
                                  negative_slope=0.2,
                                  norm=norm)

        self.time_embedding = nn.Embedding(T, num_hidden)
        self.norm_x = nn.LayerNorm(in_dim, elementwise_affine=False)


    def forward(self, g, x):
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1],))
        t = torch.randint(self.T, size=(x.shape[0],), device=x.device)
        x_t, time_embed, g = self.sample_q(t, x, g)
        loss = self.node_denoising(x, x_t, time_embed, g)
        # loss_item = {"loss": loss.item()}
        return loss

    """
    def forward(self, g, x):
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1],))

        t = torch.randint(self.T, size=(x.shape[0],), device=x.device)
        x_t, time_embed, g = self.sample_q(t, x, g)

        rec, hidden, loss = self.node_denoising(x, x_t, time_embed, g)
        # loss_item = {"loss": loss.item()}
        return rec, hidden, loss
    """


    def sample_q(self, t, x, g):
        if not self.training:
            def udf_std(nodes):
                return {"std": nodes.mailbox['m'].std(dim=1, unbiased=False)}

            g.update_all(fn.copy_u('feat', 'm'), udf_std)
            g.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'miu'))

            miu, std = g.ndata['std'], g.ndata['miu']
        else:
            miu, std = x.mean(dim=0), x.std(dim=0)
        noise = torch.randn_like(x, device=x.device)
        noise = noise * std + miu
        noise = self.norm_x(noise)
        noise = torch.sign(x) * torch.abs(noise)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise
        )
        time_embed = self.time_embedding(t)
        return x_t, time_embed, g


    def node_denoising(self, x, x_t, time_embed, g):
        out, _ = self.net(g, x_t=x_t, time_embed=time_embed)
        loss = loss_fn(out, x, self.alpha_l)
        return loss

    """
    def node_denoising(self, x, x_t, time_embed, g):
        out, hid = self.net(g, x_t=x_t, time_embed=time_embed)
        loss = loss_fn(out, x, self.alpha_l)
        return out, hid, loss
    """


    def embed(self, g, x, T):
        t = torch.full((1,), T, device=x.device)
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1],))
        x_t, time_embed, g = self.sample_q(t, x, g)
        _, hidden = self.net(g, x_t=x_t, time_embed=time_embed)
        return hidden


def loss_fn(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def exists(x):
    return x is not None


class Denoising_Unet(nn.Module):
    """\
        Denoising_Unet.

        Parameters
        ----------
        in_dim: int
            Dimension of input features.
        num_hidden: int
            Dimension of hidden layer.
        out_dim: int
            Dimension of output features.
        num_layers: int
            The number of GAT layers.
        nhead: int
            The number of head in GAT.
        activation: str
            Activation function.
        feat_drop: float
            The value of hyperparameter dropout in GATConv.
        attn_drop: float
            The value of hyperparameter dropout in GATConv.
        negative_slope: float
            The value of hyperparameter in GATConv.
        norm: str
            The layer normalization method of MLP

        Returns
        -------
        Denoised representations.

        """

    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 norm,
                 ):
        super(Denoising_Unet, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.activation = activation

        self.mlp_in_t = MlpBlock(in_dim=in_dim, hidden_dim=num_hidden * 2, out_dim=num_hidden,
                                 norm=norm, activation=activation)

        self.mlp_middle = MlpBlock(num_hidden, num_hidden, num_hidden, norm=norm, activation=activation)

        self.mlp_out = MlpBlock(num_hidden, out_dim, out_dim, norm=norm, activation=activation)

        self.down_layers.append(GATConv(num_hidden, num_hidden // nhead, nhead, feat_drop, attn_drop, negative_slope))
        self.up_layers.append(GATConv(num_hidden, num_hidden, 1, feat_drop, attn_drop, negative_slope))

        for _ in range(1, num_layers):
            self.down_layers.append(GATConv(num_hidden, num_hidden // nhead, nhead, feat_drop,
                                            attn_drop, negative_slope))
            self.up_layers.append(GATConv(num_hidden, num_hidden // nhead, nhead, feat_drop,
                                          attn_drop, negative_slope))
        self.up_layers = self.up_layers[::-1]

    def forward(self, g, x_t, time_embed):
        h_t = self.mlp_in_t(x_t)
        down_hidden = []
        for l in range(self.num_layers):
            if h_t.ndim > 2:
                h_t = h_t + time_embed.unsqueeze(1).repeat(1, h_t.shape[1], 1)
            else:
                pass
            h_t = self.down_layers[l](g, h_t)
            h_t = h_t.flatten(1)
            down_hidden.append(h_t)
        h_middle = self.mlp_middle(h_t)

        h_t = h_middle
        out_hidden = []
        for l in range(self.num_layers):
            h_t = h_t + down_hidden[self.num_layers - l - 1]
            if h_t.ndim > 2:
                h_t = h_t + time_embed.unsqueeze(1).repeat(1, h_t.shape[1], 1)
            else:
                pass
            h_t = self.up_layers[l](g, h_t)
            h_t = h_t.flatten(1)
            out_hidden.append(h_t)
        out = self.mlp_out(h_t)
        out_hidden = torch.cat(out_hidden, dim=-1)

        return out, out_hidden


class Residual(nn.Module):
    def __init__(self, fnc):
        super().__init__()
        self.fnc = fnc

    def forward(self, x, *args, **kwargs):
        return self.fnc(x, *args, **kwargs) + x


class MlpBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 norm: str = 'layernorm', activation: str = 'prelu'):
        super(MlpBlock, self).__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.res_mlp = Residual(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                              create_norm(norm)(hidden_dim),
                                              create_activation(activation),
                                              nn.Linear(hidden_dim, hidden_dim)))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = create_activation(activation)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.res_mlp(x)
        x = self.out_proj(x)
        x = self.act(x)
        return x


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    else:
        return nn.Identity


class AttentionLayer(nn.Module):
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.

    Returns
    -------
    Aggregated representations and modality weights.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha


class Cross_Decoder(nn.Module):
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Reconstructed representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Cross_Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x
