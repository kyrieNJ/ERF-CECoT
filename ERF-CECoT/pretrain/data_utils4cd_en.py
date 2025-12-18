# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding=utf-8
import os
from transformers import RobertaTokenizer
import spacy
from vocab import Vocab
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import json

tokenizer = RobertaTokenizer.from_pretrained("/roberta-base")
# tokenizer = BertTokenizer.from_pretrained('../ChineseRoberta')


def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    return data


class CDDataset(Dataset):
    def __init__(self, fname,args):
        cd_data_dict = parse_json_all(fname)

        user_count=0
        all_data = []
        maxtwilen=0
        for icdtext in tqdm(cd_data_dict):
            text=icdtext['mind_EN']
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=args.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            lable=icdtext['cd_label_nb']

            data = {
                'bert_text_sequence':encoding['input_ids'].flatten(),
                'attention_mask':encoding['attention_mask'].flatten(),
                'label': int(lable),
            }
            all_data.append(data)
            user_count+=1
        print('maxtwilen:', maxtwilen)
        print('user_count:', user_count)
        self.data = all_data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
