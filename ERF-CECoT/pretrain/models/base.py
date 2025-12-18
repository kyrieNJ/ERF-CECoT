# -*- coding: utf-8 -*-
# coding=utf-8
import torch
import copy, math
import torch.nn.functional as F
import torch.nn as nn
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    #att
    # b = p_attn.sum(2, keepdim=True)
    # c = b.squeeze(2)
    # d = c.sum(1, keepdim=True)
    # e = d.squeeze(1)
    # print(e)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn