# -*- coding: utf-8 -*-
# coding=utf-8
# encoding=utf-8
import os

from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torch.utils.data import DataLoader
from model4cd_en import ContrastiveBERT_CD
from model4emo_en import ContrastiveBERT_EMO


import torch
import random
import math
from tqdm import tqdm
import spacy
import argparse
from vocab import Vocab
import numpy as np

import pickle
import torch.nn.functional as F
from transformers import BertTokenizer
import json
from transformers import RobertaTokenizer, RobertaModel
from GateCR import GatingNetwork
# device = torch.device(device=2)

tokenizer = RobertaTokenizer.from_pretrained("/roberta-base")
# 载入newSenticKG
if os.path.getsize('./senticdict.pkl') > 0:
    with open('./senticdict.pkl', 'rb') as f:
        sentic = pickle.load(f)
else:
    print('Not exist senticdict.pkl')

nlp = spacy.load('en_core_web_lg')


def read_emolabel(cdidx):
    cdlabel = ''
    if cdidx == 0:
        # cdlabel = '愤怒'
        cdlabel = 'anger'
    elif cdidx == 1:
        # cdlabel = '厌恶'
        cdlabel = 'disgust'
    elif cdidx == 2:
        # cdlabel = '恐惧'
        cdlabel = 'fear'
    elif cdidx == 3:
        # cdlabel = '悲伤'
        cdlabel = 'sadness'
    elif cdidx == 4:
        # cdlabel = '惊讶'
        cdlabel = 'surprise'
    elif cdidx == 5:
        # cdlabel = '喜悦'
        cdlabel = 'joy'
    elif cdidx == 6:
        cdlabel = '无情绪'
    elif cdidx == -1:
        cdlabel = '首轮'
    else:
        print('cderror')
    return cdlabel

emotrans_dict = {
    "0": ['angry','annoyed','furious','jealous'],
    "1": ['disgusted'],
    "2": ['afraid','terrified','anxious','apprehensive'],
    "3": ['sad','lonely','guilty','nostalgic','disappointed','devastated','embarrassed','sentimental','ashamed'],
    "4": ['surprised','anticipating','prepared'],
    "5": ['excited','proud','grateful','impressed','hopeful','confident','joyful','content','trusting','faithful','caring'],
    "6": [],
}
#--------------------------------data-utlis-bert---------------------------------
def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    tmpdict=data
    return tmpdict


#---------------------------------bert-Infer--------------------------------

def get_parameters(model, model_init_lr, multiplier):
    parameters = []
    enc_param_optimizer = list(model.named_parameters())
    lr = model_init_lr
    for layer in range(12, -1, -1):
        layer_params = {
            'params': [p for n, p in enc_param_optimizer if f'encoder.layer.{layer}.' in n],
            'lr': lr,
            'weight_decay': 0.0
        }
        parameters.append(layer_params)
        lr *= multiplier
    return parameters

class Inferer:
    def __init__(self, args):
        self.args = args

        self.cdmodel = args.cd_model_class()
        self.parameters = [p for p in self.cdmodel.parameters() if p.requires_grad]
        self.cdmodel.to(args.device)
        self.cdmodel.load_state_dict(torch.load(self.args.cd_state_dict_path))
        self._print_args()
        #****
        self.optimizer = torch.optim.AdamW([
            {'params': self.cdmodel.bert.parameters(), 'lr': args.learning_rate},
            {'params': self.cdmodel.classifier.parameters(), 'lr': args.bert_lr},
            # {'params': self.model.classifier2.parameters(), 'lr': args.bert_lr}

        ], weight_decay=args.l2reg, betas=(0.9, 0.98))

        self.global_f1 = 0.
        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=args.device.index))

        self.cdmodel = self.cdmodel
        self.cdmodel.to(args.device)
        # switch model to evaluation mode
        self.cdmodel.eval()

        self.emomodel = args.emo_model_class()
        self.parameters = [p for p in self.emomodel.parameters() if p.requires_grad]
        self.emomodel.to(args.device)
        self.emomodel.load_state_dict(torch.load(self.args.emo_state_dict_path))
        self._print_args()
        #****
        self.optimizer = torch.optim.AdamW([
            {'params': self.emomodel.bert.parameters(), 'lr': args.learning_rate},
            {'params': self.emomodel.classifier.parameters(), 'lr': args.bert_lr},
            # {'params': self.model.classifier2.parameters(), 'lr': args.bert_lr}

        ], weight_decay=args.l2reg, betas=(0.9, 0.98))

        self.global_f1 = 0.
        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=args.device.index))

        self.emomodel = self.emomodel
        self.emomodel.to(args.device)
        # switch model to evaluation mode
        self.emomodel.eval()

        self.gating_net = GatingNetwork(7).to(args.device)
        self.gating_net.to(args.device)

        self.gating_net.load_state_dict(torch.load(self.args.gate_state_dict_path))
        self.gating_net.eval()

        torch.autograd.set_grad_enabled(False)
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.cdmodel.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.args):
            print('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))


    def processtext_cd(self, user_data_dict):
        user_count = 0
        all_data = []
        for key, item in tqdm(user_data_dict.items()):
            topic = f"“{item['prompt']}”.\n"
            topic_encoding = tokenizer.encode_plus(
                topic,
                add_special_tokens=True,
                max_length=args.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            for iconv in item['conversation']:
                if len(iconv['C9']) <= 1:
                    continue
                last_dig_encoding=torch.zeros(128,dtype=torch.int64)
                last_dig_mask=torch.zeros(128,dtype=torch.int64)
                if len(iconv['history'])!=1:
                    last_dig=iconv['history'][len(iconv['history'])-3][1]
                    last_dig_emb = tokenizer.encode_plus(
                        last_dig,
                        add_special_tokens=True,
                        max_length=args.max_len,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )
                    last_dig_encoding= last_dig_emb['input_ids'].flatten()
                    last_dig_mask=last_dig_emb['attention_mask'].flatten()
                lencon = len(iconv['history'])
                text = f"“{iconv['history'][lencon - 1][1]}”。"
                text_dig_encoding = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=args.max_len,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                #intensity
                document2 = nlp(text)
                value_ints=0
                seq_len = len(document2)
                vw_count=0
                for tmpto in document2:
                    if tmpto.text in sentic.keys():
                        value_ints+=float(sentic[tmpto.text][2])
                        vw_count+=1
                if vw_count==0:
                    value_ints=0
                else:
                    value_ints=value_ints/vw_count
                data = {
                    'topic_seq': topic_encoding['input_ids'].flatten(),
                    'topic_mask':topic_encoding['attention_mask'].flatten(),
                    'last_dig_encoding': last_dig_encoding,
                    'last_dig_mask': last_dig_mask,
                    'text_seq': text_dig_encoding['input_ids'].flatten(),
                    'text_mask': text_dig_encoding['attention_mask'].flatten(),
                    'value_ints':value_ints,
                    'dp_emo_logit_res':torch.tensor(iconv['dp_emo_logit_res']),

                }
                all_data.append(data)
                user_count += 1
        return all_data


    def processtext_cd_esc(self, user_data_dict):
        user_count = 0
        all_data = []
        for item in tqdm(user_data_dict):
            topic = f"“{item['situation']}”.\n"
            topic_encoding = tokenizer.encode_plus(
                topic,
                add_special_tokens=True,
                max_length=args.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            for iconv in item['conversation']:
                if iconv['speaker'] == 'seeker':
                    continue
                last_dig_encoding=torch.zeros(128,dtype=torch.int64)
                last_dig_mask=torch.zeros(128,dtype=torch.int64)
                if len(iconv['history'])!=1:
                    skflag=0
                    for citem2 in reversed(iconv['history']):
                        if citem2[0] == 'seeker':
                            skflag+=1
                            if skflag==2:
                                last_dig = f"“{citem2[1]}”."
                                break
                    last_dig_emb = tokenizer.encode_plus(
                        last_dig,
                        add_special_tokens=True,
                        max_length=args.max_len,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )
                    last_dig_encoding= last_dig_emb['input_ids'].flatten()
                    last_dig_mask=last_dig_emb['attention_mask'].flatten()
                for citem in reversed(iconv['history']):
                    if citem[0]=='seeker':
                        text = f"“{citem[1]}”."
                        break

                text_dig_encoding = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=args.max_len,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                #intensity
                document2 = nlp(text)
                value_ints=0
                vw_count=0
                for tmpto in document2:
                    if tmpto.text in sentic.keys():
                        value_ints+=float(sentic[tmpto.text][2])
                        vw_count+=1
                if vw_count==0:
                    value_ints=0
                else:
                    value_ints=value_ints/vw_count
                data = {
                    'topic_seq': topic_encoding['input_ids'].flatten(),
                    'topic_mask':topic_encoding['attention_mask'].flatten(),
                    'last_dig_encoding': last_dig_encoding,
                    'last_dig_mask': last_dig_mask,
                    'text_seq': text_dig_encoding['input_ids'].flatten(),
                    'text_mask': text_dig_encoding['attention_mask'].flatten(),
                    'value_ints':value_ints,
                    'dp_emo_logit_res': torch.tensor(iconv['dp_emo_logit_res']),
                }
                all_data.append(data)
                user_count += 1
        return all_data

    def evaluate_all(self, user_data_dict):
        pdata=self.processtext_cd(user_data_dict)
        # pdata=self.processtext_cd_esc(user_data_dict)
        count=0
        cd_dict={}
        emo_dict={}
        pdata1 = DataLoader(dataset=pdata, batch_size=args.batch_size, shuffle=False)
        for i_batch, sample_batched in tqdm(enumerate(pdata1)):
        # for i_batch, sample_batched in enumerate(pdata1):
            topic_seq = sample_batched['topic_seq'].to(self.args.device)
            topic_mask= sample_batched['topic_mask'].to(self.args.device)
            topic_outputs = self.cdmodel(topic_seq, topic_mask)
            topic_outputs_emo = self.emomodel(topic_seq, topic_mask)

            topic_outputs=torch.argmax(topic_outputs, -1)
            topic_outputs_emo=torch.argmax(topic_outputs_emo, -1)


            text_seq = sample_batched['text_seq'].to(self.args.device)
            text_mask= sample_batched['text_mask'].to(self.args.device)
            text_outputs = self.cdmodel(text_seq, text_mask)
            text_outputs_emo = self.emomodel(text_seq, text_mask)
            text_outputs_emo_log=text_outputs_emo
            text_outputs=torch.argmax(text_outputs, -1)
            # text_outputs_emo=torch.argmax(text_outputs_emo, -1)

            text_outputs_emo = self.gating_net(text_outputs_emo_log, sample_batched['dp_emo_logit_res'].to(self.args.device))
            text_outputs_emo = torch.argmax(text_outputs_emo, -1)

            last_dig_encoding= sample_batched['last_dig_encoding'].to(self.args.device)
            last_dig_mask= sample_batched['last_dig_mask'].to(self.args.device)

            all_emo = ''

            catflag = 1
            for ibatch in range(0, args.batch_size):
                if torch.equal(last_dig_encoding[ibatch:ibatch + 1], torch.zeros(1,128).to(self.args.device)):
                    # tmp_last_dig_outputs=torch.tensor([-1]).to(self.args.device)
                    tmp_last_dig_outputs_emo= torch.tensor([-1]).to(self.args.device)
                else:
                    # tmp_last_dig_outputs = self.cdmodel(last_dig_encoding[ibatch:ibatch + 1], last_dig_mask[ibatch:ibatch + 1])
                    # tmp_last_dig_outputs = torch.argmax(tmp_last_dig_outputs, -1)
                    tmp_last_dig_outputs_emo = self.emomodel(last_dig_encoding[ibatch:ibatch + 1], last_dig_mask[ibatch:ibatch + 1])
                    tmp_last_dig_outputs_emo = torch.argmax(tmp_last_dig_outputs_emo, -1)

                if catflag == 1:
                    # all_cd = tmp_last_dig_outputs
                    all_emo=tmp_last_dig_outputs_emo
                    catflag = 0
                else:
                    # all_cd = torch.cat((all_cd, tmp_last_dig_outputs), 0)
                    all_emo = torch.cat((all_emo, tmp_last_dig_outputs_emo), 0)

            for idx in range(len(topic_outputs)):
                tmpdata={
                    'topic_outputs_cd':topic_outputs[idx].item(),
                    # 'last_dig_outputs_cd':all_cd[idx].item(),
                    'text_outputs_cd':text_outputs[idx].item(),
                }
                cd_dict[count]=tmpdata
                tmpdataemo={
                    'topic_outputs_emo':topic_outputs_emo[idx].item(),
                    'last_dig_outputs_emo':all_emo[idx].item() ,
                    'text_outputs_emo':text_outputs_emo[idx].item(),
                    'value_ints':sample_batched['value_ints'][idx].item()
                }
                emo_dict[count]=tmpdataemo

                count+=1
        return cd_dict,emo_dict

    def processtext_cd_esc_sig(self, textcontent):
        text_dig_encoding = tokenizer.encode_plus(
            textcontent,
            add_special_tokens=True,
            max_length=args.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        #intensity
        document2 = nlp(textcontent)
        value_ints=0
        vw_count=0
        for tmpto in document2:
            if tmpto.text in sentic.keys():
                value_ints+=float(sentic[tmpto.text][2])
                vw_count+=1
        if vw_count==0:
            value_ints=0
        else:
            value_ints=value_ints/vw_count
        data = {
            'text_seq': text_dig_encoding['input_ids'].flatten(),
            'text_mask': text_dig_encoding['attention_mask'].flatten(),
            'value_ints':value_ints,
        }
        return data


    def evaluate_sig(self, user_data_dict):
        pdata=self.processtext_cd_esc_sig(user_data_dict)
        count=0
        cd_dict={}
        emo_dict={}
        pdata1 = DataLoader(dataset=pdata, batch_size=1, shuffle=False)
        for i_batch, sample_batched in tqdm(enumerate(pdata1)):
        # for i_batch, sample_batched in enumerate(pdata1):

            text_seq = sample_batched['text_seq'].to(self.args.device)
            text_mask= sample_batched['text_mask'].to(self.args.device)
            text_outputs = self.cdmodel(text_seq, text_mask)
            text_outputs_emo = self.emomodel(text_seq, text_mask)

            text_outputs=torch.argmax(text_outputs, -1)
            text_outputs_emo=torch.argmax(text_outputs_emo, -1)

            all_emo = ''

            tmpdata={
                'text_outputs_cd':text_outputs.item(),
            }
            cd_dict=tmpdata
            tmpdataemo={
                'last_dig_outputs_emo':all_emo.item() ,
                'text_outputs_emo':text_outputs_emo.item(),
                'value_ints':sample_batched['value_ints'].item()
            }
            emo_dict=tmpdataemo

        return cd_dict,emo_dict

    def processtext_cd_dataset(self, user_data_dict):
        user_count = 0
        all_data = []
        for key, item in tqdm(user_data_dict.items()):
            for iconv in item['conversation']:
                if len(iconv['C9']) <= 1:
                    continue
                lencon = len(iconv['history'])
                text = f"“{iconv['history'][lencon - 1][1]}”。"
                text_dig_encoding = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=args.max_len,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                emo = iconv['emotion']
                for ikey in emotrans_dict.keys():
                    if emo in emotrans_dict[ikey]:
                        emo= int(ikey)
                        break
                data = {
                    'text_seq': text_dig_encoding['input_ids'].flatten(),
                    'text_mask': text_dig_encoding['attention_mask'].flatten(),
                    'dp_emo_logit_res':torch.tensor(iconv['dp_emo_logit_res']),
                    'emo_label':emo,
                }
                all_data.append(data)
                user_count += 1
        return all_data



    def evaluate_emo(self, user_data_dict):
        pdata=self.processtext_cd_dataset(user_data_dict)

        t_targets_all, t_outputs_all = None, None
        n_correct, n_total = 0, 0

        pdata1 = DataLoader(dataset=pdata, batch_size=args.batch_size, shuffle=False)
        for i_batch, sample_batched in tqdm(enumerate(pdata1)):
        # for i_batch, sample_batched in enumerate(pdata1):
            text_seq = sample_batched['text_seq'].to(self.args.device)
            text_mask= sample_batched['text_mask'].to(self.args.device)
            #ori
            text_outputs_emo = self.emomodel(text_seq, text_mask)

            # GATE
            text_outputs_emo_log=text_outputs_emo
            text_outputs_emo = self.gating_net(text_outputs_emo_log, sample_batched['dp_emo_logit_res'].to(self.args.device))

            #LLM
            # text_outputs_emo=sample_batched['dp_emo_logit_res'].to(self.args.device)
            t_targets = sample_batched['emo_label'].to(self.args.device)
            n_correct += (torch.argmax(text_outputs_emo, -1) == t_targets).sum().item()
            n_total += len(text_outputs_emo)
            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = text_outputs_emo
            else:
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, text_outputs_emo), dim=0)
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='macro')

        return acc,f1



if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--cd_state_dict_path", type=str, default="state_dict/PT_BERTEncoder_cd_EN.pkl")
    parser.add_argument("--emo_state_dict_path", type=str, default="state_dict/PT_BERTEncoder_emo_EN.pkl")
    parser.add_argument("--gate_state_dict_path", type=str, default="state_dict/PT_BERTEncoder_emo_EN_gate.pkl")

    parser.add_argument('--cd_model_name', default='PT_BERTEncoder_cd', type=str)
    parser.add_argument('--emo_model_name', default='PT_BERTEncoder_emo', type=str)

    parser.add_argument('--dataset', default='CL', type=str)

    # orthogonal_，xavier_uniform_
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="learning rate.")
    parser.add_argument("--bert_lr", type=float, default=1e-4, help="learning rate for bert.")
    parser.add_argument("--l2reg", type=float, default=0.01, help="weight decay rate.")
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of total training epochs.")

    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--log_step", type=int, default=20, help="Print log every k steps.")

    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument("--max_len", type=int, default=128)


    args = parser.parse_args()

    model_classes = {
        'PT_BERTEncoder_cd': ContrastiveBERT_CD,
        'PT_BERTEncoder_emo': ContrastiveBERT_EMO,

    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # defa ult lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamw': torch.optim.AdamW,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    args.cd_model_class = model_classes[args.cd_model_name]
    args.emo_model_class = model_classes[args.emo_model_name]

    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)
    # args.device = torch.device("cuda:0")
    args.torch_version = torch.__version__

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    fname=  {
        'train': '../datasets/SWDD/SWDD_3k7_train.json',
        'test': 'emp_test_gate.json'
        # 'test': 'esc_test_gate.json'

    }

    user_data_dict = parse_json_all(fname['test'])
    ins = Inferer(args)
    cd_dict,emo_dict = ins.evaluate_all(user_data_dict)
    print(len(cd_dict))
    with open('../llama-factory/extra_data/emp_train_cd_label.json', "w",encoding="utf-8-sig") as f:
        json.dump(cd_dict, f, indent=2, ensure_ascii=False)
    with open('../llama-factory/extra_data/emp_train_emo_label_ints_gate.json', "w",encoding="utf-8-sig") as f:
        json.dump(emo_dict, f, indent=2, ensure_ascii=False)
    # acc,f1 = ins.evaluate_emo(user_data_dict)
    # print("max_acc:", acc)
    # print("max_f1:", f1)

