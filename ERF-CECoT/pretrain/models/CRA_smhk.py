import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CrossModal_Ranked_Attention(nn.Module):
    def __init__(self ,dropout=0.1):
        # in_dim: the dimension fo query vector
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.proj_t=nn.Linear(768,300)
        self.proj_cd=nn.Linear(768,300)
        self.proj_im=nn.Linear(2048,300)

        self.WqT=nn.Linear(300,300)
        self.WqI=nn.Linear(300,300)
        self.WqCD=nn.Linear(300,300)

        self.WkT=nn.Linear(300,300)
        self.WkI=nn.Linear(300,300)
        self.WkCD=nn.Linear(300,300)

        self.W1=nn.Linear(50,50)
        self.W2=nn.Linear(50,50)
        self.W3=nn.Linear(50,1)


    def forward(self, T_feature, IM_feature,CD_feature):
        proj_T=self.proj_t(T_feature)
        proj_IM=self.proj_im(IM_feature)
        proj_CD=self.proj_cd(CD_feature)

        Len_num=T_feature.shape[1]
        sqrt_d=np.sqrt(proj_IM.shape[2])
        sing_list=[]
        for li in range(0,Len_num):
            sing_list.append(1/Len_num)
        bat_list=[]
        bat_list.append(sing_list)
        N_tensor = torch.tensor(bat_list).to(T_feature.device)

        #Text
        qT= self.WqT(torch.matmul(N_tensor, proj_T))
        kT= self.WkT(proj_T)
        alphaT=torch.matmul(qT,kT.transpose(1,2))/sqrt_d
        ZT=F.sigmoid(alphaT)

        #Image
        qI= self.WqI(torch.matmul(N_tensor, proj_IM))
        kI= self.WkI(proj_IM)
        alphaI=torch.matmul(qI,kI.transpose(1,2))/sqrt_d
        ZI=F.sigmoid(alphaI)

        #CD
        qCD= self.WqCD(torch.matmul(N_tensor, proj_CD))
        kCD= self.WkCD(proj_CD)
        alphaCD=torch.matmul(qCD,kCD.transpose(1,2))/sqrt_d
        ZCD=F.sigmoid(alphaCD)

        M1=ZI*ZT
        M2=ZCD*ZT

        M=torch.cat([M1,M2],1)
        alpha_f=F.softmax(self.W3(M), dim=1)
        alpha_ti=alpha_f[:,:1,:]
        alpha_cd=alpha_f[:,1:,:]

        w_proj_IM=alpha_ti*proj_IM
        w_proj_CD=alpha_cd*proj_CD

        return proj_T,w_proj_IM,w_proj_CD


class Ranked_Attention(nn.Module):
    def __init__(self ,dropout=0.1):
        # in_dim: the dimension fo query vector
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.proj_t=nn.Linear(300,300)
        # self.proj_cd=nn.Linear(768,300)

        self.WqT=nn.Linear(300,300)
        self.WqCD=nn.Linear(300,300)

        self.WkT=nn.Linear(300,300)
        self.WkCD=nn.Linear(300,300)

        # self.W1=nn.Linear(50,50)
        # self.W2=nn.Linear(50,50)
        self.W3=nn.Linear(160,1)


    def forward(self, T_feature,CD_feature):
        proj_T=self.proj_t(T_feature)
        proj_CD=self.proj_t(CD_feature)

        Len_num=T_feature.shape[1]
        sqrt_d=np.sqrt(proj_CD.shape[2])
        sing_list=[]
        for li in range(0,Len_num):
            sing_list.append(1/Len_num)
        bat_list=[]
        bat_list.append(sing_list)
        N_tensor = torch.tensor(bat_list).to(T_feature.device)

        #Text
        qT= self.WqT(torch.matmul(N_tensor, proj_T))
        kT= self.WkT(proj_T)
        alphaT=torch.matmul(qT,kT.transpose(1,2))/sqrt_d
        ZT=F.sigmoid(alphaT)

        #CD
        qCD= self.WqCD(proj_CD)
        kCD= self.WkCD(proj_CD)
        alphaCD=torch.matmul(qCD,kCD.transpose(1,2))/sqrt_d
        ZCD=F.sigmoid(alphaCD)

        Mt =torch.randn(1,1,1)
        catflag=1

        for ibatch in range(0,Len_num):
            ZCD_i=ZCD[:,ibatch:ibatch+1,:]
            Mt_i = ZCD_i * ZT

            if catflag==1:
                Mt=Mt_i
                catflag=0
            else:
                Mt=torch.cat((Mt, Mt_i), 1)

        alpha_f=F.softmax(self.W3(Mt), dim=1)

        HCD =torch.randn(1,1,1)
        catflag=1

        for ibatch in range(0,Len_num):
            PCD_i=proj_CD[:,ibatch:ibatch+1,:]
            M_i=alpha_f[:,ibatch:ibatch+1,:]
            HCD_i=torch.matmul(M_i, PCD_i)
            # Mt_i = ZCD_i * ZT

            if catflag==1:
                HCD=HCD_i
                catflag=0
            else:
                HCD=torch.cat((HCD, HCD_i), 1)

        return proj_T,HCD

