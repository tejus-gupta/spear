import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(),])

testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True, num_workers=10)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2) 
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
net.load_state_dict(torch.load('../data/cnn-mnist.wts'))
net.to(device)

def attack(net, dataloader, adversary):
    correct = 0
    total = 0
    robustness = 0

    for data in dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, clean_predicted = torch.max(outputs.data, 1)

        adversarial_images = adversary.generate(net, images, labels)

        outputs = net(adversarial_images)
        _, adv_predicted = torch.max(outputs.data, 1)

        total += (clean_predicted == labels).sum().item()
        correct += (torch.mul(adv_predicted == labels , clean_predicted == labels)).sum().item()
        print(correct, total)

    print("Attack success rate: %.3f" % (1 - 1.0*correct/total))
    print(robustness/10000)

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spear.attacks.DeepFool import DeepFool

attack(net, testloader, DeepFool(norm=float("inf")))