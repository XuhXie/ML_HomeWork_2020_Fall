import torch
import numpy as np
import random
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from pj3 import NetForMNIST

transform_ = transforms.Compose([transforms.Resize((32, 30)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, ], [0.5, ])])
data = datasets.ImageFolder('./data/faces_4', transform=transform_)
num_class= len(data.classes)

len_of_data = len(data)
all_data = []
for data, label in data:
    gray = data[0]
    all_data.append((gray, label))
random.shuffle(all_data)
# print(all_data[0][0].shape)

train_data = all_data[:int(len_of_data * 0.8)]
test_data = all_data[int(len_of_data * 0.8):]

lr = 1e-2
EPOCH = 20
batch_size = 16

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
data_loader = {'train': train_loader, 'test': test_loader}

model = NetForMNIST(32 * 30, num_class=num_class)
model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
# optimizer = optim.SGD(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
writer = SummaryWriter('log/project4/')

global_step = 1
for epoch in range(EPOCH):
    for phase in ['train', 'test']:
        total_loss = 0
        acc_num = 0
        num = 0
        if phase == 'train':
            model = model.train()
        else:
            model = model.eval()
        for (data, label) in data_loader[phase]:
            data = data.view(data.size(0), -1)
            data = Variable(data).cuda()
            label = Variable(label).cuda()

            # data = torch.tensor(data).cuda()
            out = model(data)
            # print(out, label)
            loss = criterion(out, label)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            acc_num += num_correct.data

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                writer.add_scalar('batch_loss', loss, global_step)
                writer.add_scalar('batch_acc', num_correct * 1.0 / label.size(0), global_step)
            total_loss += loss.data * label.size(0)
            num += label.size(0)
        writer.add_scalar('epoch_acc', acc_num * 1.0 / num, epoch)
        print("Epoch {} {}: Loss: {}, Acc: {}".format(epoch + 1, phase,
                                                      total_loss / num, acc_num * 1.0 / num))
    print("\n")
