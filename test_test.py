import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from test import cellDataset
import torch.optim as optim
import torch.nn as nn
from u_net import Network
import torch.nn.functional as F


torch.set_grad_enabled(True)

transformCells = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToPILImage(), transforms.ToTensor()])
dataset = cellDataset("data/lowres/lowres_labels.csv", "data/highres/highres_labels.csv", "data/highres", "data/lowres", transform=transformCells)
dataloader = DataLoader(dataset, shuffle=True, batch_size=5)

batch = next(iter(dataloader))
#lowres, highres = batch['lowres'], batch['highres']

network = Network()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
i = 0

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, highres in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        lowres, highres = batch['lowres'], batch['highres']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = network(lowres)
        loss = criterion(outputs, highres)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0