# -*- coding: utf-8 -*-
# coding=utf-8
# coding:utf-8
import sys
from pdb import set_trace as stop


sys.path.append('../')
import torch

from transformers import BertModel, BertConfig
import torch.nn.functional as F
from torch import nn

from models.model_utils import InteractiveAttention


# bert = RobertaModel.from_pretrained('../roberta-base')
bert = BertModel.from_pretrained("../ChineseRoberta")

class ContrastiveBERT(nn.Module):
    def __init__(self, model_name='../ChineseRoberta', proj_dim=256):
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=config)
        self.in_drop = nn.Dropout(0.5)


        self.classifier = nn.Sequential(
            nn.Linear(768*2, 256),
            nn.ReLU(),
            nn.Dropout(0.55),
            nn.Linear(256, 7)  # 输出7个情绪标签
        )

        self.interactive_attention = InteractiveAttention()
        self.gru = nn.LSTM(input_size=768, hidden_size=768, bidirectional=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        outputs=outputs.last_hidden_state

        bert_out,_=self.gru(outputs)


        bert_out=bert_out[:, 0, :]

        # 通过投影头
        logits = self.classifier(bert_out)
        for batch_idx in range(logits.size(0)):
            first_dim_values = logits[batch_idx, :]  # 形状为 [6]
            for i in range(first_dim_values.size(0)):
                value = first_dim_values[i].item()
                if value >5 :
                    print(f"第{batch_idx}维第{i}个元素的值: {value}")
                elif value <-8:
                    print(f"第{batch_idx}维第{i}个元素的值: {value}")
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
