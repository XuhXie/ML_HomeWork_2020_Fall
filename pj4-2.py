import os
import torch as t
import numpy as np
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import models
import random
import math
import torch.utils.model_zoo as model_zoo

lr = 1e-4
EPOCH = 250
batch_size = 32

dirs = os.listdir("./data/faces_4/")
data_paths = []
for i in dirs:
    file_names = os.listdir('./data/faces_4/' + i)
    file_paths = ['./data/faces_4/' + i + '/' + x for x in file_names]
    data_paths += file_paths

data_size = len(data_paths)
print(data_size)

class_names = set()
for i in data_paths:
    i = i.split('/')[-1]
    i = i.split('_')[2]
    class_names.add(i)

print(class_names)
class_names = list(class_names)
num_class = len(class_names)
class_to_id = {class_names[i]: i for i in range(len(class_names))}
id_to_class = {i: class_names[i] for i in range(len(class_names))}

transform_ = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, ], [0.5, ])])


class FaceDataset(Dataset):
    def __init__(self, data_path, class_to_id, transforms):
        self.data_path = data_path
        self.class_to_id = class_to_id
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.data_path[index]
        label = self.data_path[index].split('/')[-1].split('_')[2]
        label = self.class_to_id[label]

        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = t.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)

        return data, label

    def __len__(self):
        return len(self.data_path)



train_data = data_paths[:int(data_size * 0.8)]
test_data = data_paths[int(data_size * 0.8):]
train_data = FaceDataset(train_data, class_to_id=class_to_id, transforms=transform_)
test_data = FaceDataset(test_data, class_to_id=class_to_id, transforms=transform_)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
data_loader = {'train': train_loader, 'test': test_loader}


class LayerUnit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LayerUnit, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layers(x)


model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=4, bias=True)
model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=lr)
# optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
writer = SummaryWriter('log/project4/project4-2')

global_step = 1
for epoch in range(EPOCH):
    for phase in ['train', 'test']:
        total_loss = 0
        acc_num = 0
        num = 0
        if phase == 'train':
            model.train()
        else:
            model.eval()
        for (data, label) in data_loader[phase]:
            # data = data.view(data.size(0), -1)
            data = Variable(data).cuda()
            label = Variable(label).cuda()

            # data = torch.tensor(data).cuda()
            out = model(data)
            # print(out, label)
            loss = criterion(out, label)
            _, pred = t.max(out, 1)
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
        if phase == 'test':
            writer.add_scalar('epoch_acc', acc_num * 1.0 / num, epoch)
        print("Epoch {} {}: Loss: {}, Acc: {}".format(epoch + 1, phase,
                                                      total_loss / num, acc_num * 1.0 / num))
    print("\n")
