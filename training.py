import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from test import cellDataset
import torch.optim as optim
import torch.nn as nn
from u_net import Network

torch.set_grad_enabled(True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transformCells = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToPILImage(), transforms.ToTensor()])
dataset = cellDataset("data/lowres/lowres_labels.csv", "data/highres/highres_labels.csv", "data/highres", "data/lowres", transform=transformCells)
dataloader = DataLoader(dataset, shuffle=True, batch_size=5)

batch = next(iter(dataloader))

network = Network()
#network.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, highres in enumerate(dataloader, 0):
        lowres, highres = batch['lowres'], batch['highres']
        #lowres.to(device)
        #highres.to(device)

        optimizer.zero_grad()

        outputs = network(lowres)
        loss = criterion(outputs, highres)
        loss.backward()
        optimizer.step()

PATH = './cell_net.pth'
#torch.save(network.state_dict(), PATH)