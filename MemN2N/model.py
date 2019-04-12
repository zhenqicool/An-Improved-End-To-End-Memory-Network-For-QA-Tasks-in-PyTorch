import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import math


class MemN2N(nn.Module):

    def __init__(self, params, vocab):
        super(MemN2N, self).__init__()
        self.input_size = len(vocab)
        self.embed_size = params.embed_size
        self.memory_size = params.memory_size
        self.num_hops = params.num_hops
        self.use_bow = params.use_bow
        self.use_lw = params.use_lw
        self.use_ls = params.use_ls
        self.local_atten = params.local_atten
        self.atten_dot_bias = params.atten_dot_bias
        self.atten_gen = params.atten_gen
        self.atten_con = params.atten_con
        self.atten_per = params.atten_per
        self.nl_up = params.nl_up
        self.vocab = vocab

        # create parameters according to different type of weight tying
        pad = self.vocab.stoi['<pad>']
        self.A = nn.ModuleList([nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)])
        self.A[-1].weight.data.normal_(0, 0.05)
        self.C = nn.ModuleList([nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)])
        self.C[-1].weight.data.normal_(0, 0.05)

        if self.local_atten:
            self.W_p = nn.ParameterList(nn.Parameter(I.normal_(torch.empty(self.embed_size, 1), 0, 0.05)) \
                                        for _ in range(self.num_hops))
            self.v_p = nn.ParameterList(nn.Parameter(I.normal_(torch.empty(1, self.memory_size), 0, 0.05)) \
                                        for _ in range(self.num_hops))

        if self.atten_dot_bias:
            self.DA_b = nn.ParameterList(nn.Parameter(I.normal_(torch.empty(self.memory_size, 1), 0, 0.05)) \
                                     for _ in range(self.num_hops))

        if self.atten_gen:
            # General attention
            self.GA_w = nn.ParameterList(nn.Parameter(I.normal_(torch.empty(self.embed_size, self.embed_size), 0, 0.05))\
                                       for _ in range(self.num_hops))
        if self.atten_con:
            # Concat attention
            self.CA_w1 = nn.ParameterList(nn.Parameter(I.normal_(torch.empty(self.embed_size, 1), 0, 0.05))\
                                       for _ in range(self.num_hops))
            self.CA_w2 = nn.ParameterList(nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.05)) \
                                          for _ in range(self.num_hops))

        if self.atten_per:
            # Concat attention
            self.PA_w1 = nn.ParameterList(nn.Parameter(I.normal_(torch.empty(self.embed_size, 1), 0, 0.05)) \
                                          for _ in range(self.num_hops))
            self.PA_w2 = nn.ParameterList(
                nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.05)) \
                for _ in range(self.num_hops))
            self.GLU = nn.ReLU6()
            self.PA_w3 = nn.ParameterList(
                nn.Parameter(I.normal_(torch.empty(self.memory_size, self.memory_size), 0, 0.05)) \
                for _ in range(self.num_hops))

        if self.use_lw:
            for _ in range(1, self.num_hops):
                self.A.append(self.A[-1])
                self.C.append(self.C[-1])
            self.B = nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)
            self.B.weight.data.normal_(0, 0.05)
            self.out = nn.Parameter(
                I.normal_(torch.empty(self.input_size, self.embed_size), 0, 0.05))
            if self.nl_up:
                # self.NL_w2 = nn.Parameter(I.normal_(torch.empty(1), 0, 0.05))
                self.Nl_Tah = nn.GLU()
                self.H = nn.Linear(self.embed_size, self.embed_size)
                self.H.weight.data.normal_(0, 0.05)
                self.NL_w3 = nn.Parameter(I.normal_(torch.empty(int(self.embed_size/2), self.embed_size), 0, 0.05))
            else:
                self.H = nn.Linear(self.embed_size, self.embed_size)
                self.H.weight.data.normal_(0, 0.05)
        else:
            for _ in range(1, self.num_hops):
                self.A.append(self.C[-1])
                self.C.append(nn.Embedding(self.input_size, self.embed_size, padding_idx=pad))
                self.C[-1].weight.data.normal_(0, 0.05)
            self.B = self.A[0]
            self.out = self.C[-1].weight  # 最后的W权重矩阵

        # temporal matrix
        self.TA = nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.05))
        self.TC = nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.05))

    def forward(self, story, query):
        sen_size = query.shape[-1]
        weights = self.compute_weights(sen_size)
        state = (self.B(query) * weights).sum(1)  # 32*20

        sen_size = story.shape[-1]
        weights = self.compute_weights(sen_size)
        for i in range(self.num_hops):
            memory = (self.A[i](story.view(-1, sen_size)) * weights).sum(1).view(
                *story.shape[:-1], -1)  # 32*50*20
            memory += self.TA
            output = (self.C[i](story.view(-1, sen_size)) * weights).sum(1).view(
                *story.shape[:-1], -1)
            output += self.TC

            # @ is inner production 32*50*20 @ 32*20*1  unsqueeze->squeeze(32 * 50 * 1 -> 32 * 50)
            if self.local_atten:
                probs = (memory @ state.unsqueeze(-1)).squeeze()
                p_t = (self.memory_size * torch.sigmoid(self.v_p[i] @ torch.tanh(memory @ state.unsqueeze(-1))).squeeze()).type(torch.IntTensor)  # 32 * 50 * 1 -> 32 *50
                p_t_score = torch.Tensor([[math.exp(-((i-j)**2)/(2*15*15)) for i in range(50)] for j in p_t])
                if torch.cuda.is_available():
                    p_t_score = p_t_score.cuda()
            elif self.atten_dot_bias:
                probs = (memory @ state.unsqueeze(-1) + self.DA_b[i]).squeeze()
            elif self.atten_gen:
                probs = (memory @ self.GA_w[i] @ state.unsqueeze(-1)).squeeze()
            elif self.atten_con:
                probs = (memory @ self.CA_w1[i] + self.CA_w2[i] @ state.unsqueeze(-1)).squeeze()
            elif self.atten_per:
                probs = (self.PA_w3[i] @ self.GLU(memory @ self.PA_w1[i] + self.PA_w2[i] @ state.unsqueeze(-1))).squeeze()
            else:
                probs = (memory @ state.unsqueeze(-1)).squeeze()

            if not self.use_ls:
                if self.local_atten:
                    probs = F.softmax(probs, dim=-1) * p_t_score
                else:
                    probs = F.softmax(probs, dim=-1)  # dim = -1 对列之间作softmax运算
            response = (probs.unsqueeze(1) @ output).squeeze()  # 32 * 1 * 20 -> 32 * 20
            if self.use_lw:
                if self.nl_up:
                    state = self.Nl_Tah(self.H(response) + state) @ self.NL_w3
                else:
                    state = self.H(response) + state

            else:
                state = response + state

        return F.log_softmax(F.linear(state, self.out), dim=-1)

    def compute_weights(self, J):
        """ weights are invariable when training"""
        d = self.embed_size
        if self.use_bow:
            weights = torch.ones(J, d)
        else:
            func = lambda j, k: 1 - (j + 1) / J - (k + 1) / d * (1 - 2 * (j + 1) / J)    # 0-based indexing
            weights = torch.from_numpy(np.fromfunction(func, (J, d), dtype=np.float32))
        return weights.cuda() if torch.cuda.is_available() else weights
