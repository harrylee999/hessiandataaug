import torch
import numpy as np
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
import random


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



torch.manual_seed(7)  # cpu
torch.cuda.manual_seed(7)  # gpu
np.random.seed(7)  # numpy
random.seed(7)
torch.backends.cudnn.deterministic = True  # cudnn


transform_all = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = datasets.CIFAR10('data/cifar10/', train=True, transform=transform_all)
test_dataset = datasets.CIFAR10('data/cifar10/', train=False, transform=transform_all)

# train_data = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=500)
# 选取每个类别中的前 500 个样本
indices = []
for i in range(10):
    indices += [j for j, (x, y) in enumerate(train_dataset) if y == i][:500]
    
# 使用 SubsetRandomSampler 来构造 dataloader
sampler = SubsetRandomSampler(indices)

train_data = DataLoader(train_dataset, batch_size=64, sampler=sampler)

model = SimpleCNN().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01,weight_decay=1e-5,momentum=0.9,nesterov=True)
model_h = SimpleCNN().to(device)

from pyhessian.hessian import hessian

criterion = torch.nn.CrossEntropyLoss().to(device)


        
def test():
    model.eval()
    with torch.no_grad():
       
        num_corrects = 0
        for data_batch in test_loader:
            images, labels = data_batch
 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicts = torch.max(outputs, -1)
            num_corrects += sum(torch.eq(predicts.cpu(), labels.cpu())).item()
        accuracy = num_corrects / len(test_dataset)
    return accuracy


def train():


    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()])

    model.train()

    for data_batch in train_data:
        images, labels = data_batch
        images, labels = images.to(device), labels.to(device)
        # images = transform_train(images)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    current_acc = test()
    temp_loss = loss.cpu().detach().numpy()

    model_h.load_state_dict(model.state_dict())
    model_h.eval()
    hessian_comp = hessian(model_h, criterion, dataloader=train_data, device=device)
    trace,_ = hessian_comp.trace()

    return temp_loss,current_acc,np.mean(trace)

def mixup_train():


    model.train()
    
    for data_batch in train_data:
        images, labels = data_batch

        images, labels = images.to(device), labels.to(device)



        images, targets_a, targets_b, lam = mixup_data(images, labels)

        images, targets_a, targets_b = map(torch.autograd.Variable, (images,
                                    targets_a, targets_b))


        outputs = model(images)

        loss = mixup_criterion(outputs, targets_a, targets_b, lam)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    current_acc = test()
    temp_loss = loss.cpu().detach().numpy()

    model_h.load_state_dict(model.state_dict())
    model_h.eval()
    hessian_comp = hessian(model_h, criterion, dataloader=train_data, device=device)
    trace,_ = hessian_comp.trace()

    return temp_loss,current_acc,np.mean(trace)

def mixup_data(x, y, gamma=1):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if gamma > 0:
        lam = np.random.beta(gamma, gamma)
       
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
   
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_train(beta = 1,prob = 1):
    
    model.train()

    for data_batch in train_data:
        images, labels = data_batch
        input, labels = images.to(device), labels.to(device)

        p = np.random.rand(1)
        if beta > 0 and p < prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    current_acc = test()
    temp_loss = loss.cpu().detach().numpy()

    model_h.load_state_dict(model.state_dict())
    model_h.eval()
    hessian_comp = hessian(model_h, criterion, dataloader=train_data, device=device)
    trace,_ = hessian_comp.trace()

    return temp_loss,current_acc,np.mean(trace)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


EPOCH = 100
if __name__ == '__main__':

    
    name = "mixup+"

    loss_list = []
    acc_list_test = []
    trace_list = []
    best_acc = 0

    for epoch in range(EPOCH):
        if name == "noaug":
            loss,acc,trace = train()
        elif name == "mixup+":
            if epoch < 90:
                loss,acc,trace = mixup_train()
            else:
                loss,acc,trace = train()
        elif name == "cutmix+":
            if epoch < 90:
                loss,acc,trace = cutmix_train()
            else:
                loss,acc,trace = train()
     
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),f"./result/{name}_best.pt")

        print(epoch,loss,acc,trace)
        loss_list.append(loss)
        acc_list_test.append(acc)
        trace_list.append(trace)
    torch.save(model.state_dict(),f"./result/{name}_last.pt")

    import csv
    rows = zip(loss_list,acc_list_test,trace_list)
    
    with open(f"{name}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for row in rows:
                writer.writerow(row)    