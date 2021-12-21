import torch
import torch.nn as nn
from torch.nn.init import uniform_
import numpy as np
from utils import gumbel_softmax


class ComplEx(nn.Module):
    def __init__(self, dim,
                 num_n, num_r,
                 reg_n, reg_r,
                 drop_in, drop_r, drop_out,
                 interval, reciprocal=True):
        super(ComplEx, self).__init__()

        self.dim = dim
        self.num_n = num_n
        self.num_r = num_r
        self.reg_n = reg_n
        self.reg_r = reg_r
        self.drop_in = nn.Dropout(drop_in)
        self.drop_r = nn.Dropout(drop_r)
        self.drop_out = nn.Dropout(drop_out)
        self.interval = interval
        self.reciprocal = reciprocal

        reci_mul = 1 if not reciprocal else 2
        self.n_embedding = nn.Parameter(torch.Tensor(num_n, 2, dim))
        self.r_embedding = nn.Parameter(torch.Tensor(num_r * reci_mul, 2, dim))

        self.reset_parameters()

    def reset_parameters(self):
        uniform_(self.n_embedding, a=-self.interval, b=self.interval)
        uniform_(self.r_embedding, a=-self.interval, b=self.interval)

    ###########################################################################
    #     complex arithmetics
    ###########################################################################

    def c(self, real, imag):
        return torch.cat([real.unsqueeze(-2), imag.unsqueeze(-2)], -2)

    def transpose(self, x):
        real = x[:, 0, :].t()
        imag = x[:, 1, :].t()
        return self.c(real, imag)

    def conj(self, x):
        real = x[:, 0, :]
        imag = -x[:, 1, :]
        return self.c(real, imag)

    def norm_sq(self, x):
        real = x[:, 0, :]
        imag = -x[:, 1, :]
        return real.pow(2) + imag.pow(2)

    def cmul(self, x, y):
        ac = x[:, 0, :].mul(y[:, 0, :])
        bd = x[:, 1, :].mul(y[:, 1, :])
        ad = x[:, 0, :].mul(y[:, 1, :])
        bc = x[:, 1, :].mul(y[:, 0, :])
        real = ac - bd
        imag = ad + bc
        return self.c(real, imag)

    def cmm(self, x, y):
        ac = x[:, 0, :].mm(y[:, 0, :])
        bd = x[:, 1, :].mm(y[:, 1, :])
        ad = x[:, 0, :].mm(y[:, 1, :])
        bc = x[:, 1, :].mm(y[:, 0, :])
        real = ac - bd
        imag = ad + bc
        return self.c(real, imag)

    def cdot(self, x, y):
        ac = x[:, 0, :].mm(y[:, 0, :].t())
        bd = x[:, 1, :].mm(-y[:, 1, :].t())
        ad = x[:, 0, :].mm(-y[:, 1, :].t())
        bc = x[:, 1, :].mm(y[:, 0, :].t())
        real = ac - bd
        imag = ad + bc
        return self.c(real, imag)

    def cdot_real(self, x, y):
        ac = x[:, 0, :].mm(y[:, 0, :].t())
        bd = x[:, 1, :].mm(-y[:, 1, :].t())
        real = ac - bd
        return real

    def cdot_real_bmm(self, x, y):
        ac = torch.bmm(y[:, :, 0, :], x[:, 0, :, None]).squeeze(-1)
        bd = torch.bmm(-y[:, :, 1, :], x[:, 1, :, None]).squeeze(-1)
        real = ac - bd
        return real

    ###########################################################################
    #     discretization
    ###########################################################################

    def c_discretize(self, x, tau, beta, hard):
        centroids = self.n_embedding
        a = self.cdot_real(x, centroids)
        weight, logits = gumbel_softmax(a, tau, beta, hard)
        res = weight.mm(centroids.view(self.num_n, -1))
        return res.view(-1, 2, self.dim), logits

    ###########################################################################
    #     regularization
    ###########################################################################

    def Lp_regularization(self, s_idx, r_idx, o_idx, p=2):
        ent_idx = torch.cat([s_idx, o_idx])
        rel_idx = r_idx.view(-1)
        rel_idx = rel_idx if self.reciprocal else rel_idx.fmod(self.num_r)
        ent = torch.index_select(self.n_embedding, 0, ent_idx)
        rel = torch.index_select(self.r_embedding, 0, rel_idx)
        ent = ent.view(-1, self.dim)
        rel = rel.view(-1, self.dim)

        ent_norm = ent.abs_().pow(p).sum(-1).mean(-1)
        rel_norm = rel.abs_().pow(p).sum(-1).mean(-1)
        norm = self.reg_n * ent_norm + self.reg_r * rel_norm
        return norm

    ###########################################################################
    ###########################################################################

    def forward(self, s_idx, r_idx, alpha, tau, beta, hard=False):
        hard = not self.training

        bs, size = r_idx.shape

        s = self.n_embedding[s_idx]
        s = self.drop_in(s)

        neg_o = self.n_embedding
        neg_o = self.drop_out(neg_o)

        r_embed = self.r_embedding if self.reciprocal \
            else torch.cat([self.r_embedding, self.conj(self.r_embedding)], 0)

        if size == 1:
            r = r_embed[r_idx[:, 0]]
            r = self.drop_r(r)
            sr = self.cmul(s, r)
            score = self.cdot_real(sr, neg_o)

        else:
            for step in range(r_idx.size(1) - 1):
                r = r_embed[r_idx[:, step]]
                r = self.drop_r(r)
                sr = self.cmul(s, r)
                s_d, _ = self.c_discretize(sr, tau, beta, hard)
                if np.random.random() > alpha:
                    s = s + s_d
                else:
                    s = s_d
            r = r_embed[r_idx[:, -1]]
            r = self.drop_r(r)
            sr = self.cmul(s, r)
            score = self.cdot_real(sr, neg_o)

        return score

    ###########################################################################
    ###########################################################################

    def __repr__(self):
        return '{}({})\n   (entities:  {},\n    relations: {})'.format(
                                                    self.__class__.__name__,
                                                    self.dim,
                                                    self.num_n,
                                                    self.num_r
                                                    )
