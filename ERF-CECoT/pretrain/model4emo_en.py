# -*- coding: utf-8 -*-
# coding=utf-8
# coding:utf-8
import sys
from pdb import set_trace as stop


sys.path.append('../')
import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F
from torch import nn

from transformers import RobertaModel,RobertaForSequenceClassification




bert = RobertaModel.from_pretrained("/roberta-base")

# bert = BertModel.from_pretrained("../ChineseRoberta")

class ContrastiveBERT_EMO(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = bert
        self.in_drop = nn.Dropout(0.5)


        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)  # 输出7个情绪标签
        )



    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask,output_hidden_states=True)
        outputs=outputs.last_hidden_state
        outputs = self.in_drop(outputs)  # bert输出时第一次dropout


        bert_out=outputs[:, 0, :]

        # 通过投影头
        logits = self.classifier(bert_out)

        return logits

    def save_bert(self, path):
        """保存BERT主体（不包括投影头）用于下游任务"""
        torch.save(self.bert.state_dict(), path)


class PT_BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.classifier = nn.Linear(768, 7)  # 分类器t
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)  # 输出7个情绪标签
        )

        self.Bert_encoder = bert#Bert模型

    def forward(self, inputs):


        text_bert_sequence,attention_mask= inputs[0],inputs[1]
        bert_out = self.Bert_encoder(text_bert_sequence, attention_mask=attention_mask)
        bert_out1 = bert_out.last_hidden_state[:, 0, :]

        logits = self.classifier(bert_out1)
        logits2 = self.classifier2(bert_out1)

        return logits,bert_out.last_hidden_state.sum(1, keepdim=True).squeeze(1)
