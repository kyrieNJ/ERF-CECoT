# -*- coding: utf-8 -*-
# coding=utf-8
# encoding=utf-8
import json
import sys
from torch.utils.data import DataLoader
# from CL_test import ContrastiveBERT
from model4emo_en import ContrastiveBERT_EMO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import math
import argparse
import numpy as np
import torch.nn as nn
from transformers import BertModel

from sklearn import metrics
from vocab import Vocab
from data_utils4emo_en import CDDataset
from tqdm import tqdm
import pickle
import torch.utils.data.distributed
# from torchmetrics.regression import FocalLoss
import torchvision.ops
import torch.nn.functional as F


class Instructor:
    def __init__(self, args):
        self.args = args

        #数据加载阶段
        self.trainset = CDDataset(args.dataset_file['train'],args)
        self.testset = CDDataset(args.dataset_file['test'],args)
        #模型训练阶段
        self.model = args.model_class()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        # self.model.load_state_dict(torch.load(self.args.state_dict_path))
        self.model.to(args.device)
        self._print_args()

        self.optimizer = torch.optim.AdamW([
            {'params': self.model.bert.parameters(), 'lr': args.learning_rate},
            {'params': self.model.classifier.parameters(), 'lr': args.bert_lr},

        ], weight_decay=args.l2reg,betas=(0.9, 0.98))

        self.global_acc = 0.
        self.global_im = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=args.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.args):
            print('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.args.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0
        # criterion2 = ContrastiveLoss()

        for epoch in range(self.args.num_epoch):
            print('-' * 100)
            print('epoch: ', epoch)
            n_correct, n_total,loss_total = 0, 0,0
            increase_flag = False
            self.model.train()  # 训练过程
            for i_batch, sample_batched in tqdm(enumerate(self.train_data_loader)):
            # for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]

                targets = sample_batched['label'].to(self.args.device)
                outputs = self.model(inputs[0], attention_mask=inputs[1])
                # loss= criterion(outputs, targets)

                multilabel_targets = F.one_hot(targets, num_classes=7).float()
                loss = torchvision.ops.sigmoid_focal_loss(
                    inputs=outputs,
                    targets=multilabel_targets,
                    alpha=0.25,  # 平衡正负样本权重（可选）
                    gamma=2.0,  # 调节难易样本权重（γ越大，对难样本关注越高）
                    reduction='mean'
                )
                loss.backward()

                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)

                if global_step % self.args.log_step == 0:
                    train_loss = loss_total / n_total
                    train_acc = n_correct / n_total
                    # print('train_acc:',train_acc)
                    print("train loss: {:.4f}, train acc: {:.4f}, ".format(train_loss, train_acc))

            print("开始测试：")
            test_acc, test_f1 = self._evaluate_acc_f1(self.test_data_loader)#评估过程
            if test_acc > max_test_acc:
                all_max_test_acc=test_acc

                max_test_acc = test_acc
                max_test_f1 = test_f1
                increase_flag = True
                print('>>> best model saved.')

            print('loss: {:.4f}, acc: {:.4f}, \n test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc, test_f1))

            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 8:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0

        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in tqdm(enumerate(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
                t_targets = t_sample_batched['label'].to(self.args.device)
                t_outputs = self.model(t_inputs[0], t_inputs[1])

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), average='macro')

        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()

        self.train_data_loader = DataLoader(dataset=self.trainset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=2)
        self.test_data_loader = DataLoader(dataset=self.testset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=2)

        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/' + self.args.model_name + '_' + self.args.dataset + '.txt', 'w', encoding='utf-8')

        max_acc = 0
        max_f1 = 0

        for i in range(args.repeat):
            print('repeat: ', (i+1))
            f_out.writelines('repeat: '+str(i+1))
            # self._reset_params()
            max_test_acc, max_test_f1 = self._train(criterion, self.optimizer)
            print('max_test_acc: {0}     max_test_f1: {1}        '.format(max_test_acc, max_test_f1))
            f_out.writelines('max_test_acc: {0}, max_test_f1: {1}\n'.format(max_test_acc, max_test_f1))
            if(max_acc <= max_test_acc or max_f1 <= max_test_f1 ):
                max_acc = max_test_acc
                max_f1 = max_test_f1
            print('-' * 100)

        print("max_acc:", max_acc)
        print("max_f1:", max_f1)
        f_out.writelines('------------------------------------------------------------------------\n')
        f_out.writelines('max_acc: {0}, max_f1: {1}\n'.format(max_acc, max_f1))
        f_out.writelines('------------------------------------------------------------------------\n')

        f_out.close()

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

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='PT_BERTEncoder', type=str)
    # parser.add_argument("--state_dict_path", type=str, default="state_dict/re_2_emo_cl_contrastive_Roberta.pth")
    parser.add_argument('--method', default='PT', type=str,help="PT,CL")
    parser.add_argument('--dataset', default='emo_EN', type=str)
    #orthogonal_，xavier_uniform_
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="learning rate.")
    parser.add_argument("--bert_lr", type=float, default=1e-4, help="learning rate for bert.")
    parser.add_argument("--l2reg", type=float, default=0.01, help="weight decay rate.")
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of total training epochs.")

    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--log_step", type=int, default=10, help="Print log every k steps.")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument("--max_len", type=int, default=128)

    parser.add_argument('--repeat', default=1, type =int)

    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument("--lower", default=True, help="Lowercase all words.")
    parser.add_argument("--direct", default=False)
    parser.add_argument("--loop", default=True)
    parser.add_argument("--reset_pooling", default=False, action="store_true")
    parser.add_argument("--output_merge",type=str,default="none",help="merge method to use, (none, gate)",)
    args = parser.parse_args()


    model_classes = {
        'PT_BERTEncoder':ContrastiveBERT_EMO,

    }
    input_colses = {
        'PT_BERTEncoder': [
            'bert_text_sequence',
            'attention_mask',
        ],
    }

    dataset_files = {
        'emo': {
            'train': '../datasets/CD/C2D2_emo_train.json',
            # 'train':"../datasets/CD/re_2_scene_simp_emo_CL_train_dataset.json",
            'test': '../datasets/CD/C2D2_emo_test.json'
        },
        'emo_EN': {
            'train': '../datasets/CD/C2D2_emo_train_EN.json',
            'test': '../datasets/CD/C2D2_emo_test_EN.json'
        },
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamw': torch.optim.AdamW,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    args.model_class = model_classes[args.model_name]
    args.inputs_cols = input_colses[args.model_name]
    args.dataset_file = dataset_files[args.dataset]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device is None else torch.device(args.device)
    # args.device = torch.device('cpu')
    args.torch_version = torch.__version__

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    ins = Instructor(args)
    ins.run()

