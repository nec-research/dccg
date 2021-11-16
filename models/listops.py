# MIT License

# Copyright (c) 2018 Ethan Fetaya, Thomas Kipf

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import gumbel_softmax


num_tokens = 14


class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.dim = dim
        self.embed = nn.Embedding(num_tokens, dim)
        self.rnn_q = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)#, dropout=.1)
        self.rnn_k = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True)#, dropout=.1)
        
    def forward(self, x):
        x = self.embed(x)
        q, _ = self.rnn_q(x)
        k, _ = self.rnn_k(x)
        A = torch.bmm(q, k.transpose(1, 2))
        return A
    
    
class GNN(nn.Module):
    def __init__(self, dim, resid):
        super(GNN, self).__init__()
        self.dim = dim
        self.num_rounds = 5
        self.resid = resid
        self.embed = nn.Embedding(num_tokens, dim)
        self.msg_fc1 = nn.Linear(dim*2, dim)
        self.msg_fc2 = nn.Linear(dim, dim)
        self.out_fc1 = nn.Linear(dim, dim)
        self.out_fc2 = nn.Linear(dim, 10, bias=False)
        self.dropout = nn.Dropout(p=.1)
        
    def discretize(self, x, tau, beta, hard):
        centroids = self.embed.weight[4:]
        logits = self.out_fc1(x.view(-1, self.dim))
        logits = self.out_fc2(F.relu(logits))
        a, _ = gumbel_softmax(logits, tau, beta, hard=hard)
        return torch.mm(a, centroids).view(x.shape)  
        
    def msg(self, x, A, tok):
        size = x.shape[1]
        x_emb = self.embed(tok).unsqueeze(2).repeat(1, 1, size, 1)
#         x_i = x.unsqueeze(2).repeat(1, 1, size, 1)
        x_j = x.unsqueeze(1).repeat(1, size, 1, 1)
        pre_msg = torch.cat([x_emb, x_j], dim=-1)
        msg = F.relu(self.msg_fc1(pre_msg))
        msg = self.dropout(msg)
        msg = F.relu(self.msg_fc2(msg))
        A_inp = A.transpose(1, 2)
        msg = torch.mul(A_inp[:,:,:,None], msg)
        return msg.sum(2)
    
    def forward(self, tok, A, alpha, tau, beta, hard):
        x = self.embed(tok)
        shape = x.shape
        for _ in range(self.num_rounds):
            msg = self.msg(x, A, tok)
            x = x + msg
            if _ < self.num_rounds - 1:
                x_d = self.discretize(x, tau, beta, hard)
                if not self.resid:
                    x = x_d
                else:
                    if np.random.random() > alpha:
                        x = x + x_d
                    else:
                        x = x_d
        out = F.relu(self.out_fc1(x[:,0,:]))
        out = self.out_fc2(out)
        x = F.relu(self.out_fc1(x.view(-1, self.dim)))
        x = self.out_fc2(x)
        return out, x.view(-1, shape[1], 10)       
    
    
class Net(nn.Module):
    def __init__(self, dim, resid):
        super(Net, self).__init__()
        self.dim = dim
        self.encoder = Encoder(dim)
        self.decoder = GNN(dim, resid)
        
    def forward(self, x, alpha, tau, beta, hard):
        A = self.encoder(x)
        A[:,0,:] = 0
        A, _ = gumbel_softmax(A, tau, beta, hard)
        out, x = self.decoder(x, A, alpha, tau, beta, hard)
        return out, A.detach(), x
    
    
    
  





    
class Encoder_Arb(nn.Module):
    def __init__(self, dim):
        super(Encoder_Arb, self).__init__()
        self.dim = dim
        self.embed = nn.Embedding(num_tokens, dim)
        self.rnn_q = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True, dropout=.1)
        self.rnn_k = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True, dropout=.1)
        
    def forward(self, x):
        x = self.embed(x)
        q, _ = self.rnn_q(x)
        k, _ = self.rnn_k(x)
        A = torch.bmm(q, k.transpose(1, 2))
        return A   
    
    
class GNN_Arb(nn.Module):
    def __init__(self, dim):
        super(GNN_Arb, self).__init__()
        self.dim = dim
        self.num_rounds = 5
        self.embed = nn.Embedding(num_tokens, dim)
        self.msg_fc1 = nn.Linear(dim*2, dim)
        self.msg_fc2 = nn.Linear(dim, dim)
        self.out_fc1 = nn.Linear(dim, dim)
        self.out_fc2 = nn.Linear(dim, 10, bias=False)
        self.dropout = nn.Dropout(p=.1)        
        
    def msg(self, x, A, tok):
        size = x.shape[1]
        x_i = x.unsqueeze(2).repeat(1, 1, size, 1)
        x_j = x.unsqueeze(1).repeat(1, size, 1, 1)
        pre_msg = torch.cat([x_i, x_j], dim=-1)
        msg = F.relu(self.msg_fc1(pre_msg))
        msg = self.dropout(msg)
        msg = F.relu(self.msg_fc2(msg))
        A_inp = A.transpose(1, 2)
        msg = torch.mul(A_inp[:,:,:,None], msg)
        return msg.sum(2)

    
    def forward(self, tok, A):
        x = self.embed(tok)
        shape = x.shape
        for _ in range(self.num_rounds):
            msg = self.msg(x, A, tok)
            x = x + msg
        out = F.relu(self.out_fc1(x[:,0,:]))
        out = self.out_fc2(out)
        x = F.relu(self.out_fc1(x.view(-1, self.dim)))
        x = self.out_fc2(x)
        return out, x.view(-1, shape[1], 10)      
    
    
class Arb(nn.Module):
    def __init__(self, dim):
        super(Arb, self).__init__()
        self.dim = dim
        self.encoder = Encoder_Arb(dim)
        self.decoder = GNN_Arb(dim)
        
    def forward(self, x, tau, hard):
        A = self.encoder(x)
        A[:,0,:] = 0
        A, _ = gumbel_softmax(A, tau, hard)
        out, x = self.decoder(x, A)
        return out, A.detach(), x
    
    
    
    
    
    
class BaseLSTM(nn.Module):
    def __init__(self, dim):
        super(BaseLSTM, self).__init__()
        self.dim = dim
        self.embed = nn.Embedding(num_tokens, dim)
        self.rnn = nn.LSTM(input_size=dim,
                           hidden_size=dim,
                           num_layers=1,
                           batch_first=True,
                           dropout=.1,
                           bidirectional=False)
        self.lin = nn.Linear(dim, dim)
        self.clas = nn.Linear(dim, 10, bias=False)
        
    def forward(self, x):
        x = self.embed(x)
        seq, (h, c) = self.rnn(x)
        x = h.squeeze(0)
        x = F.relu(self.lin(x))
        return self.clas(x)