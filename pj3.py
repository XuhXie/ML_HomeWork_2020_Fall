import torch
from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


class NetForMNIST(nn.Module):
    def __init__(self, in_dim, num_class):
        super(NetForMNIST, self).__init__()
        self.layer1 = LayerUnit(in_dim, 300)
        self.layer2 = LayerUnit(300, 100)
        self.fn = nn.Linear(100, num_class)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fn(x)
        return x


if __name__ == '__main__':
    transforms_ = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5, ], [0.5, ])]
    )

    lr = 1e-2
    EPOCH = 60
    num_class = 10
    batch_size = 64

    train_data = datasets.MNIST(root='./data', train=True, transform=transforms_, download=True)
    test_data = datasets.MNIST(root='./data', train=False, transform=transforms_, download=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    data_loader = {'train': train_loader, 'test': test_loader}

    model = NetForMNIST(28 * 28, num_class=num_class)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
    writer = SummaryWriter('log/project3/')

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
                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('acc', num_correct * 1.0 / label.size(0), global_step)
                total_loss += loss.data * label.size(0)
                num += label.size(0)

            print("Epoch {} {}: Loss: {}, Acc: {}".format(epoch + 1, phase,
                                                          total_loss / num, acc_num * 1.0 / num))
        print("\n")
