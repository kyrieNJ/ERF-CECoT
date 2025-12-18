# save as MoE_gate_example.py and run
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.model_utils import InteractiveAttention
from transformers import BertModel, BertConfig
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import random
import math
import argparse
import numpy as np
from sklearn import metrics
from transformers import RobertaTokenizer, RobertaModel

# -----------------------
# Utilities
# -----------------------


##DATASETS
# tokenizer = BertTokenizer.from_pretrained('../ChineseRoberta')
tokenizer = RobertaTokenizer.from_pretrained("/roberta-base")

def parse_json_all(path):
    with open(path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
    return data

def text_to_roberta_sequence(text, maxtextlen):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=110,  # 支持最长512 token[6](@ref)
        return_tensors="pt"
    )
    if maxtextlen < len(text)-2:
        maxtextlen = len(text)-2
        print(maxtextlen)
    return inputs['input_ids'].squeeze(0),inputs['attention_mask'].squeeze(0),maxtextlen

class CDDataset(Dataset):
    def __init__(self, fname,args):
        cd_data_dict = parse_json_all(fname)
        user_count=0
        all_data = []
        maxtextlen=0
        maxtwilen=0
        for icdtext in tqdm(cd_data_dict):
            text = icdtext['mind_EN']
            # bert_text_sequence,maxtextlen = text_to_bert_sequence(text, args.max_len,maxtextlen)
            # bert_text_sequence,attention_mask,maxtextlen = text_to_roberta_sequence(text,maxtextlen)
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=args.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            lable=icdtext['emo_label_nb']

            data = {
                'bert_text_sequence':encoding['input_ids'].flatten(),
                'attention_mask':encoding['attention_mask'].flatten(),
                'dp_emo_logit_res':torch.tensor(icdtext['dp_emo_logit_res']),
                'label': int(lable)
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


# -----------------------
# Example Expert Models
# -----------------------
bert = RobertaModel.from_pretrained("/roberta-base")
# bert = BertModel.from_pretrained("../ChineseRoberta")

class ContrastiveBERT(nn.Module):
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

        # self.gru = nn.LSTM(input_size=768, hidden_size=768, bidirectional=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        outputs = outputs.last_hidden_state
        outputs = self.in_drop(outputs)  # bert输出时第一次dropout

        bert_out = outputs[:, 0, :]

        # 通过投影头
        logits = self.classifier(bert_out)
        return logits

    def save_bert(self, path):
        """保存BERT主体（不包括投影头）用于下游任务"""
        torch.save(self.bert.state_dict(), path)

# -----------------------
# Gating Network
# -----------------------
class GatingNetwork(nn.Module):
    """
    A small MLP gating network.
    Input is a concatenation of features (e.g. expert logits/probs, entropies, sample-level features).
    Output is weights over experts (softmax).
    """
    def __init__(self, input_dim, hidden_dim=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_dim, 7)  # 输出7个情绪标签
        )

    def forward(self, features1,features2):

        # features: (B, input_dim)
        logits1 = self.mlp(features1)
        logits2 = self.mlp(features2)

        logits1=logits1.mean(1).unsqueeze(1)
        logits2=logits2.mean(1).unsqueeze(1)

        fin_rep = torch.cat((logits1, logits2), 1)

        batch_num = fin_rep.shape[0]
        catflag = 1

        cat_outputs1=torch.randn(1,1)
        cat_outputs2=torch.randn(1,1)
        for ibatch in range(0,batch_num):
            tmpalpha1 = F.softmax(fin_rep[ibatch:ibatch+1,:], dim=1)
            if catflag==1:
                cat_outputs1=torch.matmul(tmpalpha1[:,:1], features1[ibatch:ibatch+1,:])
                cat_outputs2=torch.matmul(tmpalpha1[:,1:2], features2[ibatch:ibatch+1,:])
                catflag=0
            else:
                tmpcat_outputs1 = torch.matmul(tmpalpha1[:, :1], features1[ibatch:ibatch + 1, :])
                tmpcat_outputs2 = torch.matmul(tmpalpha1[:, 1:2], features2[ibatch:ibatch + 1, :])
                cat_outputs1=torch.cat((cat_outputs1, tmpcat_outputs1), 0) #bx160x768
                cat_outputs2=torch.cat((cat_outputs2, tmpcat_outputs2), 0) #bx160x768

        fin_logits=cat_outputs1+cat_outputs2
        return fin_logits.squeeze(1)


class Instructor:
    def __init__(self, args):
        self.args = args

        #数据加载阶段
        self.trainset = CDDataset(args.dataset_file['train'],args)
        self.testset = CDDataset(args.dataset_file['test'],args)
        #模型训练阶段
        self.emomodel = ContrastiveBERT()
        self.parameters = [p for p in self.emomodel.parameters() if p.requires_grad]
        self.emomodel.to(args.device)
        self.emomodel.load_state_dict(torch.load(self.args.emo_state_dict_path))

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=args.device.index))

        # Option: freeze experts and only train gating (common if experts are pretrained)
        freeze_experts = True
        if freeze_experts:
            for p in self.emomodel.parameters():
                p.requires_grad = False


        # gating input_dim: for each expert we used max_prob and entropy -> 2*E
        gating_input_dim = 7
        self.gating_net = GatingNetwork(gating_input_dim).to(args.device)
        # self.gating_net = GatingNetwork2().to(args.device)

        self.parameters = [p for p in self.gating_net.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW([
            {'params': self.gating_net.parameters(), 'lr': args.learning_rate},
        ], weight_decay=args.l2reg)
        self.train_data_loader = DataLoader(dataset=self.trainset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=2)
        self.test_data_loader = DataLoader(dataset=self.testset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=2)
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.gating_net.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.args):
            print('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))


    # -----------------------
    # Simple training example
    # -----------------------
    def demo_train(self):
        optimizer = self.optimizer
        criterion = nn.CrossEntropyLoss()

        max_acc = 0
        max_f1 = 0
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0

        for epoch in range(self.args.num_epoch):
            n_correct, n_total,loss_total = 0, 0,0
            print('-' * 100)
            print('epoch: ', epoch)
            increase_flag = False
            self.gating_net.train()  # 训练过程
            self.emomodel.eval()

            for i_batch, sample_batched in tqdm(enumerate(self.train_data_loader)):
            # for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
                model_log=self.emomodel(inputs[0], inputs[1])
                outputs = self.gating_net(model_log, inputs[2])
                targets = sample_batched['label'].to(self.args.device)

                loss = criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gating_net.parameters(), 1.0)  # 梯度裁剪
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
            test_acc, test_f1 = self._evaluate_acc_f1(self.test_data_loader)  # 评估过程
            if test_acc > max_test_acc:
                all_max_test_acc = test_acc

                max_test_acc = test_acc
                max_test_f1 = test_f1
                increase_flag = True
                torch.save(self.gating_net.state_dict(), 'state_dict/' + self.args.model_name +'_'+ self.args.dataset + '_gate.pkl')
                print('>>> best model saved.')

            print('loss: {:.4f}, acc: {:.4f}, \n test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc,
                                                                                           test_f1))

            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0

        print('max_test_acc: {0}     max_test_f1: {1}        '.format(max_test_acc, max_test_f1))
        if (max_acc <= max_test_acc or max_f1 <= max_test_f1):
            max_acc = max_test_acc
            max_f1 = max_test_f1
        print('-' * 100)

        print("max_acc:", max_acc)
        print("max_f1:", max_f1)

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.gating_net.eval()
        self.emomodel.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in tqdm(enumerate(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.args.device) for col in self.args.inputs_cols]
                t_targets = t_sample_batched['label'].to(self.args.device)
                model_log=self.emomodel(t_inputs[0], t_inputs[1])
                t_outputs = self.gating_net(model_log, t_inputs[2])
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


if __name__ == "__main__":
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='PT_BERTEncoder', type=str)
    parser.add_argument("--emo_state_dict_path", type=str, default="state_dict/PT_BERTEncoder_emo_EN.pkl")
    parser.add_argument('--method', default='PT', type=str, help="PT,CL")
    parser.add_argument('--dataset', default='emo_EN', type=str)
    # orthogonal_，xavier_uniform_
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate.")
    parser.add_argument("--l2reg", type=float, default=0.01, help="weight decay rate.")
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of total training epochs.")

    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--log_step", type=int, default=10, help="Print log every k steps.")
    parser.add_argument('--device', default=None, type=str)

    parser.add_argument("--seed", type=int, default=5565)#17
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    model_classes = {
        'PT_BERTEncoder': ContrastiveBERT,

    }
    input_colses = {
        'PT_BERTEncoder': [
            'bert_text_sequence',
            'attention_mask',
            'dp_emo_logit_res',
        ],
    }

    dataset_files = {
        'emo_EN': {
            'train': '../datasets/CD/C2D2_emo_train_EN_GATE.json',
            # 'train':"../datasets/CD/re_2_scene_simp_emo_CL_train_dataset.json",
            'test': '../datasets/CD/C2D2_emo_test_EN_GATE.json'
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
    args.torch_version = torch.__version__

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ins = Instructor(args)
    ins.demo_train()
