import torch
import torch.nn as nn
import json
import argparse
import logging
import numpy as np
from copy import deepcopy
from tqdm import tqdm
bar_form = '{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'
from models import Net
from utils import json2data, get_dataloader, get_beta

parser = argparse.ArgumentParser()
parser.add_argument('--tau', type=float, default=1.)
parser.add_argument('--temp', type=int, default=1)
parser.add_argument('--residual', type=int, default=1)
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()


########################
gamma = 0.008
alpha_step = 1 / 500

dim = 60
batch_size = 100
num_epochs = 100
lr = 0.0005
########################





def main(alpha, tau, beta):
    best_dev = 0.
    best_model = None
    for epoch in range(1, num_epochs+1):
        train_loss, alpha, beta = train(epoch, alpha, tau, beta)
        dev_loss, acc, edge, _ = test(dev_dict, alpha=1., tau=tau, beta=1., hard=True)
        form = 'epoch {:2}  alpha {:.3f}  tau {:.3f}  beta {:.3f}  loss {:.3f}  acc {:.3f}  edge {:.3f}'
        out = form.format(epoch, alpha, tau, beta, train_loss, acc, edge)
        logger.info(out)
        if acc > best_dev:
            best_dev = float(acc)
            best_model = deepcopy(model.state_dict())
    analysis(best_model)
    return








tau = args.tau
anneal = bool(args.temp)
resid = bool(args.residual)
device = torch.device('cuda') if bool(args.cuda) else torch.device('cpu')
beta = 1. if not anneal else 0.
alpha = 1. if not resid else 0.



########################################################################
model = Net(dim, resid).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(reduction='sum')
########################################################################




kind_dict = {
    (False, False): 'Base',
    (True,  False): 'Temp',
    (False, True):  'Residual',
    (True, True):   'Residual_Temp'
}
kind = kind_dict[(anneal, resid)]
   
name = 'listops'
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
#                         logging.FileHandler(f'{name}.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
logger.info(f'Run {name} with tau={tau}, Temp.-Matching:{anneal}, Dropout-Residuals:{resid} on {device}')


logger.info('loading data...')
path = 'data/listops/'
with open(path + 'vocab.json', 'r') as vocab_file:
    vocab_raw = json.load(vocab_file)
vocab = {vocab_raw[str(i)]: i for i in range(len(vocab_raw))}
train_dict = json2data(path + 'train1.json', vocab)
train_dict.update(json2data(path + 'train2.json', vocab))
dev_dict = json2data(path + 'dev.json', vocab)
num_tokens = len(vocab)


def train(epoch, alpha, tau, beta):
    model.train()
    bar_format = 'epoch: {:2}'.format(epoch)
    bar_format += bar_form
    train_loss = 0
    train_loader, num_ex = get_dataloader(train_dict, batch_size)
    size = len(train_loader)
    for step, batch in enumerate(tqdm(train_loader, bar_format=bar_format)):
        data, target, _, _, _ = batch
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output, _, _ = model(data, alpha, tau, beta, hard=False)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if not step % (size // 10):
            if anneal:
                beta = get_beta(beta, tau, -gamma)
            if resid:
                alpha = min(1., alpha + alpha_step)
    return train_loss / num_ex, alpha, beta


def test(data_dict, alpha, tau, beta, hard, test_inter=False):
    model.eval()
    test_loss = 0
    correct = 0
    edge_num = 0
    edge_num_true = 0
    edge_correct = 0
    inter_correct = 0
    inter_all = 0
    loader, num_ex = get_dataloader(data_dict, batch_size)
    with torch.no_grad():
        for data, target, A, inter, inter_mask in loader:
            data, target, A = data.to(device), target.to(device), A.to(device)
            inter, inter_mask = inter.to(device), inter_mask.to(device)
            output, B, x = model(data, alpha=1., tau=tau, beta=1., hard=True)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            mask = (data.unsqueeze(-1).repeat(1, 1, data.size(1)) != 0)
            mask[:,0,:] = False
            edge_num_true += (A[mask]==1).nonzero(as_tuple=False).shape[0]
            edge_num += (B[mask]==1).nonzero(as_tuple=False).shape[0]
            edge_correct += ((B[mask]==1) & (B[mask]==A[mask])).nonzero(as_tuple=False).shape[0]
            if test_inter:
                inter_mask[:,0] = 0  #filter final operation for intermediate results
                num = (inter_mask==1).nonzero(as_tuple=False).shape[0]
                if num > 0:
                    inter_all += num
                    inter_out = x[inter_mask.bool()]
                    inter_pred = inter_out.data.max(-1, keepdim=True)[1].squeeze(1)
                    inter_target = inter[inter_mask.bool()] - 4 #inter indices are translated by 4
                    inter_correct += inter_pred.eq(inter_target).nonzero(as_tuple=False).shape[0]  
    test_loss /= num_ex
    acc = correct / num_ex
    inter_acc = inter_correct/inter_all if inter_all else 0.
    return test_loss, acc, edge_correct/edge_num_true, inter_acc







def analysis(state_dict):
    logger.info('loading test data...')
    test_dict = json2data(path + 'test.json', vocab)
    test8_dict = json2data(path + 'test8.json', vocab)
    test10_dict = json2data(path + 'test10.json', vocab)
    logger.info('testing...')
    model.load_state_dict(state_dict)
    _, test_acc, test_edge, test_inter = test(test_dict, alpha=1., tau=tau, beta=1., hard=True, test_inter=True)
    model.decoder.num_rounds = 8
    _, test8_acc, _, _ = test(test8_dict, alpha=1., tau=tau, beta=1., hard=True)
    model.decoder.num_rounds = 10
    _, test10_acc, _, _ = test(test10_dict, alpha=1., tau=tau, beta=1., hard=True)
    logger.info('{:6}\tacc {:.3f}\tedge {:.3f}\tinter {:.3f}'.format('test', test_acc, test_edge, test_inter))
    logger.info('{:6}\tacc {:.3f}'.format('test8', test8_acc))
    logger.info('{:6}\tacc {:.3f}'.format('test10', test10_acc))
    
    
    
    
if __name__ == '__main__':
    main(alpha, tau, beta)