import torch
import torchvision.utils
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import cellDataset
import torch.optim as optim
import torch.nn as nn
from u_net import Network
from torch.utils.tensorboard import SummaryWriter
import sys

writer = SummaryWriter("runs/cells2")
torch.set_grad_enabled(True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transformCells = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToPILImage(), transforms.ToTensor()])
dataset = cellDataset("data/lowres/lowres_labels.csv", "data/highres/highres_labels.csv", "data/highres", "data/lowres", transform=transformCells)
dataloader = DataLoader(dataset, shuffle=True, batch_size=5)

batch = next(iter(dataloader))

network = Network()



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, highres in enumerate(dataloader, 0):
        lowres, highres = batch['lowres'], batch['highres']


        optimizer.zero_grad()

        outputs = network(lowres)

        loss = criterion(outputs, highres).to(device)
        loss.backward().to(device)
        optimizer.step()

        running_loss += loss.item().to(device)

        if (i * 1) % 5 == 0:
            # training_loss = "LR=0.001, Momentum = 0.9"
            writer.add_scalar("trainingloss3", running_loss / 5, epoch * len(dataloader) * i)






PATH = './cell_net.pth'
#torch.save(network.state_dict(), PATH)