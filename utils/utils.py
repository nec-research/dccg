import torch
import numpy as np


def gumbel_softmax(logits, tau, beta, hard, dim=-1):
    noise = -torch.empty_like(
        logits, memory_format=torch.legacy_contiguous_format)
    gumbels = noise.exponential_().log()
    gumbels = logits + gumbels*beta
    gumbels = gumbels / tau
    y_soft = gumbels.softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        zeroes = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format)
        y_hard = zeroes.scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret, gumbels


def get_beta(beta, tau, gamma):
    T = 1. - beta / tau
    T_new = T * np.exp(gamma)
    return (1. - T_new) * tau
