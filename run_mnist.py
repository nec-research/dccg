import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import argparse
import logging
from models import MNIST_Net
from utils import MNIST_Addition, test_MNIST, get_beta
bar_form = '{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'


parser = argparse.ArgumentParser()
parser.add_argument('--tau', type=float, default=8.)
parser.add_argument('--temp', type=int, default=1)
parser.add_argument('--residual', type=int, default=1)
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()


########################
tau = args.tau
gamma = 0.008
alpha_step = 1 / 500

batch_size = 16
num_epochs = 30
lr = 0.0001
# lr = 0.0003
########################


resid = bool(args.residual)
anneal = bool(args.temp)

device = torch.device('cuda') if bool(args.cuda) else torch.device('cpu')
alpha = 1. if not resid else 0.
beta = 1. if not anneal else 0.


def main(alpha, tau, beta):
    train(num_epochs, alpha, tau, beta)


kind_dict = {
    (False, False): 'Base',
    (True,  False): 'Temp',
    (False, True):  'Residual',
    (True, True):   'Residual_Temp'
}

kind = kind_dict[(anneal, resid)]


########################################################################
net = MNIST_Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.NLLLoss(reduction='sum')
num_params = net.count_parameters()
########################################################################


path = 'data/mnist/'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])

train_dataset = MNIST_Addition(
    torchvision.datasets.MNIST(root=path, train=True, download=True,
                               transform=transform),
    path+'train_data.txt')
test_dataset = MNIST_Addition(
    torchvision.datasets.MNIST(root=path, train=False, download=True,
                               transform=transform),
    path+'test_data.txt')
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                          shuffle=False, num_workers=1)


name = 'mnist'
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
logger.info(f'the model has a total of {num_params} parameters')


def train(num_epochs, alpha, tau, beta):
    num_iter = len(train_dataset) // batch_size
    log_period = num_iter // 2
    temp_period = num_iter // 8
    i = 1
    running_loss = 0.0
    for epoch in range(num_epochs):
        for data in train_loader:
            optimizer.zero_grad()
            input1, input2, labels = data
            input1 = Variable(input1.to(device))
            input2 = Variable(input2.to(device))
            labels = Variable(labels.to(device))
            outputs = net(input1, input2, alpha, tau, beta, hard=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % temp_period == 0:
                if resid:
                    alpha = min(1., alpha+alpha_step)
                if anneal:
                    beta = get_beta(beta, tau, -gamma)
            if i % log_period == 0:
                f1, acc = test_MNIST(net, tau, test_loader,
                                     len(test_dataset), device)
                inter_loss = running_loss / (log_period*batch_size)
                form = 'epoch {:2}  step {}  alpha {:.3}  beta {:.3} ' + \
                       'loss {:.4}  acc {:.4}  f1 {:.4}'
                logger.info(form.format(epoch, i, alpha, beta,
                                        inter_loss, acc, f1))
                running_loss = 0
            i += 1


if __name__ == '__main__':
    main(alpha, tau, beta)
