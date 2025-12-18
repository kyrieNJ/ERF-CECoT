# -*- coding: utf-8 -*-
# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


class RelationAttention(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=64, dropout=0.1):
        # in_dim: the dimension fo query vector
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, feature, dep_tags_v, dmask):
        '''
        C feature/context [N, L, 2D]
        Q dep_tags_v          [N, L, D]
        mask dmask          [N, L]
        '''
        Q = self.fc1(dep_tags_v)
        Q = self.relu(Q)
        Q = self.fc2(Q)  # (N, L, 1)
        Q = Q.squeeze(2)  # N, L
        Q = F.softmax(mask_logits(Q, dmask), dim=1)

        Q = self.dropout(Q.unsqueeze(2))
        out = torch.bmm(feature.transpose(1, 2), Q)
        out = out.squeeze(2)
        # out = torch.sigmoid(out)
        return out  # ([B, D])


class RelationAttention2(nn.Module):
    def __init__(self, in_dim=300):
        # in_dim: the dimension fo query vector
        super().__init__()
        self.linear_q = nn.Linear(in_dim, 2 * in_dim)
        self.linear_k = nn.Linear(2 * in_dim, 2 * in_dim)
        self.linear_v = nn.Linear(2 * in_dim, 2 * in_dim)

    def forward(self, feature, dep_tags_v, dmask):
        '''
        Q dep_tags_v          [N, L, D]
        K feature/context [N, L, D]
        V feature/context [N, L, D]
        mask dmask          [N, L]
        '''
        d_k = dep_tags_v.size(-1)
        Q = self.linear_q(dep_tags_v)
        K = self.linear_k(feature)
        V = self.linear_v(feature)

        scores = torch.matmul(Q, K.transpose(-2, -1)) \
                 / math.sqrt(d_k)  # N,L,L

        att = F.softmax(mask_logits(scores.sum(1, keepdim=True), dmask.unsqueeze(1)), dim=2)  # N,1,L
        out = torch.matmul(att, V).squeeze(1)
        return out  # ([B, D])

class InteractiveAttention(nn.Module):
    def __init__(self ,dropout=0.1):
        # in_dim: the dimension fo query vector
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, feature, kgfeature):
        alpha_mat = torch.matmul(feature, kgfeature.transpose(1, 2))
        alpha = F.softmax(alpha_mat, dim=2)

        # torch.set_printoptions(edgeitems=50)
        #print("alpha:",alpha.sum(1,keepdim=True))

        Q = self.dropout(alpha)
        out = torch.matmul(Q, kgfeature)
        # print("out:",out.size())
        return out


