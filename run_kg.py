import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import argparse
import logging
from os.path import expanduser
from models import ComplEx
from utils import get_beta, get_positives, get_negatives
from utils import get_path_dataset, get_ent_rel, get_graph
bar_format = '{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'
home = expanduser("~")


parser = argparse.ArgumentParser()
parser.add_argument('--tau', type=float, default=4)
parser.add_argument('--temp', type=int, default=1)
parser.add_argument('--residual', type=int, default=1)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--fb', type=int, default=1)
parser.add_argument('--zero', type=int, default=5)
args = parser.parse_args()

tau = args.tau
anneal = bool(args.temp)
resid = bool(args.residual)
device = torch.device('cuda') if bool(args.cuda) else torch.device('cpu')
beta = 1. if not anneal else 0.
alpha = 1. if not resid else 0.
dataset = 'freebase' if args.fb else 'wordnet'
zero = args.zero


data_dir = f'data/{dataset}/'

id = np.random.randint(10**9)
name = f'{dataset}_{zero}_{id}'

logging.basicConfig(format='%(asctime)s - %(levelname)s - ' +
                           '%(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        # logging.FileHandler(f'{name}.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
logger.info(f'Run {name} with tau={tau}, Temp.-Matching:{anneal}, ' +
            f'Dropout-Residuals:{resid} on {device}')


configs = {
    'wordnet': {
        'batch_size': 512,
        'dim': 256,
        'num_epochs': 200,
        'eval_every': 10,
        'learning_rate': 1e-3,
        'tau': tau,
        'gamma': 0.008,
        'alpha_step': 1 / 200,
        'residual': resid,
        'anneal': anneal,
        'reg_n': 1e-15,
        'reg_r': 1e-9,
        'drop_in': 0.7,
        'drop_out': 0.1,
        'drop_r': 0.5,
        'interval': 1.0,
        'reciprocal': True
    },
    'freebase': {
        'batch_size': 512,
        'dim': 256,
        'num_epochs': 200,
        'eval_every': 10,
        'learning_rate': 1e-3,
        'tau': tau,
        'gamma': 0.008,
        'alpha_step': 1 / 200,
        'residual': resid,
        'anneal': anneal,
        'reg_n': 1e-15,
        'reg_r': 1e-8,
        'drop_in': 0.1,
        'drop_out': 0.6,
        'drop_r': 0.2,
        'interval': 1.0,
        'reciprocal': True
    }
}

config = configs[dataset]


entity_list, relation_list = get_ent_rel(data_dir)
num_n, num_r = len(entity_list), len(relation_list)
entity_dict = {ent: i for i, ent in enumerate(entity_list)}
relation_dict = {rel: i for i, rel in enumerate(relation_list)}
relation_list_both = relation_list + ['**' + rel for rel in relation_list]
relation_dict_both = {rel: i for i, rel in enumerate(relation_list_both)}
nodes, edges = get_graph(data_dir, entity_list, relation_list)

dev_datasets, dev_facts = zip(*[
    get_path_dataset('dev', size, entity_dict,
                     relation_dict_both, data_dir)
    for size in range(1, 6)
])
dev_loaders = [
    DataLoader(dev_data, shuffle=True, batch_size=256)
    for dev_data in dev_datasets
]

train_datasets = [
    get_path_dataset('train', size, entity_dict,
                     relation_dict_both, data_dir)[0]
    for size in tqdm(range(1, zero + 1), bar_format=bar_format)
]
train_loaders = [
    DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])
    for train_data in train_datasets
]

model = ComplEx(
    dim=config['dim'],
    num_n=num_n,
    num_r=num_r,
    reg_n=config['reg_n'],
    reg_r=config['reg_r'],
    drop_in=config['drop_in'],
    drop_r=config['drop_r'],
    drop_out=config['drop_out'],
    interval=config['interval'],
    reciprocal=config['reciprocal']
)
model.to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = Adam(model.parameters(), lr=config['learning_rate'])


def train(epoch, train_loaders, model, criterion, optimizer, num_r, device,
          alpha, tau, beta):
    model.train()
    train_iters = [iter(loader) for loader in train_loaders]
    num_train = [len(loader.dataset) for loader in train_loaders]
    num_all = sum(num_train)
    perc = np.array([num / num_all for num in num_train])
    num_all_plus = num_all + len(train_loaders[0].dataset)
    beta_step = (num_all_plus // 3) - 1
    running_loss = 0.0
    num = 0
    while np.absolute(perc).sum() > 1e-9:
        iter_idx = np.random.choice(len(train_iters), p=perc)
        itera = train_iters[iter_idx]
        try:
            batch = next(itera)
        except StopIteration:
            perc[iter_idx] = 0
            if np.absolute(perc).sum() < 1e-9:
                break
            else:
                perc = np.array([p / sum(perc) for p in perc])
                continue

        optimizer.zero_grad()
        indizes, s, R, o = batch
        size = R.size(1)
        num += indizes.size(0)
        s, R, o = s.to(device), R.to(device), o.to(device)
        score_o = model(s, R, alpha=alpha, tau=tau, beta=beta, hard=False)
        loss = criterion(score_o, o)
        if size == 1:
            num += indizes.size(0)
            R_rev = (R + num_r).fmod(num_r*2)
            score_s = model(o, R_rev, alpha=alpha, tau=tau,
                            beta=beta, hard=False)
            loss_s = criterion(score_s, s)
            loss += loss_s
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if num > beta_step:
            if anneal:
                beta = get_beta(beta, tau, -config['gamma'])
            if resid:
                alpha = min(1., alpha + config['alpha_step'])
            beta_step += (num_all_plus // 3) - 1

    return running_loss / num, alpha, beta


def evaluate(loader, facts, model, nodes, edges, entity_dict, device, tau):
    model.eval()
    rank_list = list()
    quant_list = list()
    for batch in loader:
        indizes, s, R, o = batch
        s, R = s.to(device), R.to(device)
        scores = model(s, R, alpha=1.0, tau=tau, beta=0.0, hard=True).cpu()
        for i, idx in enumerate(indizes):
            fact = facts[idx]
            P = get_positives(fact, nodes)
            N = get_negatives(fact, edges)
            negatives = set(N) - set(P)
            num_N = len(negatives)
            if num_N == 0:
                continue
            assert num_N == len(N) - len(P)
            answer = o[i].item()
            score = scores[i]
            answer_score = score[answer]
            negative_scores = score[[entity_dict[neg] for neg in negatives]]
            neg_better = (negative_scores > answer_score)
            num_neg_better = neg_better.nonzero(as_tuple=False).size(0)
            rank = num_neg_better + 1
            rank_list.append(rank)
            quant_list.append((num_N - num_neg_better) / num_N)
    return rank_list, quant_list


def validate(loaders, facts_all, model, nodes, edges, entity_dict, device,
             tau, test=False):
    affix = '_test' if test else ''
    rank_all, quant_all = list(), list()
    res = dict()
    for i, loader in enumerate(loaders):
        facts = facts_all[i]
        size = loader.dataset[0][2].size(0)
        rank_list, quant_list = evaluate(loader, facts, model, nodes, edges,
                                         entity_dict, device, tau)
        rank_all += rank_list
        quant_all += quant_list
        num_h10 = (np.array(rank_list) < 11).nonzero()[0].size
        h10 = num_h10 / len(rank_list)
        mq = np.mean(quant_list)
        res.update({
            f'h10_{size}' + affix: h10,
            f'mq_{size}' + affix: mq
        })
    num_h10_all = (np.array(rank_all) < 11).nonzero()[0].size
    h10_all = num_h10_all / len(rank_all)
    mq_all = np.mean(quant_all)
    res.update({
        'h10' + affix: h10_all,
        'mq' + affix: mq_all
    })
    return res


best_h10 = 0.0
logger.info('{:>8}{:>8}{:>8}{:>12}'.format('epoch', 'alpha', 'beta', 'loss'))
for epoch in range(1, config['num_epochs'] + 1):
    loss, alpha, beta = train(epoch, train_loaders, model, criterion,
                              optimizer, num_r, device, alpha, tau, beta)
    logger.info('{:8d}{:8.2f}{:8.2f}{:12.4f}'.format(
        epoch, alpha, beta, loss))
    if not epoch % config['eval_every']:
        logger.info('evaluate...')
        evaluation = validate(
            dev_loaders, dev_facts, model, nodes, edges,
            entity_dict, device, tau, test=False)
        logger.info(str(evaluation))
        h10 = evaluation['h10']
        if h10 > best_h10:
            logger.info('saving state_dict...')
            best_h10 = float(h10)
            best_epoch = int(epoch)
            best_state = deepcopy(model.state_dict())
            # torch.save(best_state, f'{name}.pth')

logger.info('=== finished ===')
logger.info('Best Dev: {}  (epoch {})'.format(np.round(100*best_h10, 2),
                                              best_epoch))
logger.info('Starting test phase with best model')
model.load_state_dict(best_state)
test_datasets, test_facts = zip(*[
    get_path_dataset('test', size, entity_dict,
                     relation_dict_both, data_dir)
    for size in range(1, 6)
])
test_loaders = [
    DataLoader(test_data, shuffle=True, batch_size=256)
    for test_data in test_datasets
]
evaluation = validate(test_loaders, test_facts, model, nodes, edges,
                      entity_dict, device, tau, test=True)
logger.info(str(evaluation))
