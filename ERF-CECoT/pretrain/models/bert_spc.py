# -*- coding: utf-8 -*-
# coding=utf-8
# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from transformers import BertModel
# bert = BertModel.from_pretrained("/home/hc/wyj/data/bert-base-uncased")
# bert = BertModel.from_pretrained("/home/amax/yyd/data/bert-base-uncased")
bert = BertModel.from_pretrained('./bert-base-uncased')
# bert = BertModel.from_pretrained('bert-base-uncased')

class BERT_SPC(nn.Module):
    def __init__(self, args):
        super(BERT_SPC, self).__init__()
        self.Bert_encoder = bert
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(args.hidden_dim, args.num_class)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.Bert_encoder(text_bert_indices, token_type_ids=bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
