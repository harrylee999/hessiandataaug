# Code adapted from PyHessian
# https://github.com/amirgholami/PyHessian

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import torch
import numpy as np
from pyhessian.hessian import hessian
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets
import logging
import argparse
import matplotlib.pyplot as plt
import copy
from statistics import mean
import time
from pathlib import Path
import pickle
from matplotlib import cm

import random
import json
import torch.nn.functional as F
import torch.nn as nn

from mpl_toolkits.mplot3d import Axes3D

class SimpleCNN(nn.Module):
    def __init__(self, input_dim = 400, hidden_dims = [120,84], output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.classifier = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

       
        y = self.classifier(x)
        return y

def getEigen(args, logger):
    logger.info('{}\t{}\t{}\n'.format(args.cfg['method'], args.cfg['data_dir'], args.model_dir))
    device = 'cuda:0'

    model = SimpleCNN().to(device)

    model.load_state_dict(torch.load(args.model_dir))


    #model.apply(lambda m: setattr(m, 'width_mult', 1.0))

    # create loss function
    criterion = torch.nn.CrossEntropyLoss()

    # get dataset 
    transform_all = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10('../data/cifar10/', train=True, transform=transform_all)
    test_dataset = datasets.CIFAR10('../data/cifar10/', train=False, transform=transform_all)

    # train_loader = DataLoader(train_dataset, batch_size=args.cfg['batch_size'],shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.cfg['batch_size'])

    indices = []
    for i in range(10):
        indices += [j for j, (x, y) in enumerate(train_dataset) if y == i][:500]
        
    # 使用 SubsetRandomSampler 来构造 dataloader
    sampler = SubsetRandomSampler(indices)

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)


    model.eval()

    if args.single_batch:
        for inputs, targets in train_loader:
            break
        inputs, targets = inputs.cuda(), targets.cuda()
        hessian_comp = hessian(model, criterion, data=(inputs, targets), device=device)
    else:
        hessian_comp = hessian(model, criterion, dataloader=train_loader, device=device)
    top_n = 2 if args.plot else 1
    logger.info('{}'.format(args.model_dir))
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=top_n)
    logger.info('***Top Eigenvalues: {}'.format(top_eigenvalues))
    
    trace, diag = hessian_comp.trace()
    logger.info('***Trace: {}'.format(np.mean(trace)))
    logger.info('***Diag: {}'.format(diag))
    np.save('{}/diag'.format(args.save_dir), diag)
   
    test(model, test_loader, args.model_name)
    if args.plot:
        print('Plotting Loss Space...')
        plotLoss(model, top_eigenvector, criterion, train_loader, args)

def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def get_Z(model_orig,  model_perb, directions, alphas):
    for m_orig, m_perb, d0, d1 in zip(model_orig.parameters(), model_perb.parameters(), directions[0], directions[1]):
        m_perb.data = m_orig.data + alphas[0] * d0 + alphas[1] * d1
    return model_perb

def plotLoss(model, top_eigenvector, criterion, train_loader, args):
    lams = np.linspace(-1.0, 1.0, 41).astype(np.float32)
    model_perb = copy.deepcopy(model)

    x, y = np.meshgrid(lams, lams, sparse=True)
    z = np.zeros((len(lams), len(lams)))
    for xidx in range(len(lams)):
        for yidx in range(len(lams)):
            print(xidx,yidx)
            lam = [x[0,xidx], y[yidx,0]]
            model_perb = get_Z(model, model_perb, top_eigenvector, lam)
            avg_list = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                avg_list.append(criterion(model_perb(inputs), targets.long()).item())
                if args.single_batch:
                    break
            z[xidx, yidx] = mean(avg_list)
    np.save('{}/z'.format(args.save_dir), z)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Perturbation X')
    ax.set_ylabel('Perturbation Y')
    # ax.set_zlabel('Loss')
    for ii in range(0,360,30):
        ax.view_init(elev=10., azim=ii)
        plt.savefig('{}/angle{}.png'.format(args.save_dir, ii))

def precomputed_plot(args):
    lams = np.linspace(-1.0, 1.0, 41).astype(np.float32)
    x, y = np.meshgrid(lams, lams, sparse=True)
    plt.figure()

    z = np.load("D:\study\postgraduate\课程\机器学习\code\pyhessian\logs\hessian\{}/z.npy".format(args.model_dir.split('.')[1][6:]))
    ax = plt.axes(projection='3d')
    MAX_Z = 120
    ax.set_zlim(0, MAX_Z)
    my_col = cm.cividis(z/MAX_Z)
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.cividis)
    mappable.set_array(np.array([0,MAX_Z]))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='cividis', edgecolor='none', facecolors=my_col)
    xticks  = [-1,-0.5,0,0.5,1]
    ax.set_xlabel(r'$\epsilon_{0}$')
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_ylabel(r'$\epsilon_{1}$')
    ax.set_zlabel('Loss')
    plt.colorbar(mappable,fraction=0.02, pad=0.02, orientation="vertical")
    for ii in range(0,360,30):
        ax.view_init(elev=10., azim=ii)
        plt.savefig("D:\study\postgraduate\课程\机器学习\code/pyhessian/logs/hessian/{}/pdf/angle{}.png".format(args.model_dir.split('.')[1][6:],ii))

def test(model, test_dataloader, model_name):
    model.cuda()
    model.eval()

    test_correct = 0.0
    test_loss = 0.0
    test_sample_number = 0.0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dataloader):
            x = x.cuda()
            target = target.cuda()

            pred = model(x)
            # loss = criterion(pred, target)
            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(target).sum()

            test_correct += correct.item()
            # test_loss += loss.item() * target.size(0)
            test_sample_number += target.size(0)
        acc = (test_correct / test_sample_number)*100
        logging.info("************* {} Acc = {:.2f} **************".format(model_name, acc))
    return acc

def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, metavar='N',
                        help='FIle path to model file')
    parser.add_argument('--single_batch', action='store_true', default=False,
                        help='Use only a single batch')
    parser.add_argument('--precomputed', action='store_true', default=False,
                        help='Visualize landscape plot from precomputed data')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot landscape')
    args, unknown = parser.parse_known_args()
    set_random_seed(7)

    run_folder = '/'.join(args.model_dir.split('/')[:-1])
    args.model_name = args.model_dir.split('/')[-1].split('.')[0]
    if args.precomputed:
        precomputed_plot(args)
    else:
        args.save_dir = '{}/hessian/{}'.format(run_folder, args.model_name)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename='{}/{}.log'.format(args.save_dir, args.model_name), filemode='w', level=logging.INFO)
        logger = logging.getLogger('Hessian')
        with open('{}/{}'.format(run_folder, 'config.txt'), 'r') as f:
            args.cfg = json.loads(f.read())
        if not os.path.isabs(args.cfg['data_dir']):
            args.cfg['data_dir'] = '../'+args.cfg['data_dir']
        getEigen(args, logger)