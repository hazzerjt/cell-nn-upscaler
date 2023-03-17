import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import cellDataset
import torch.nn as nn
from u_net import Network
from collections import OrderedDict
from run_manager import *


#writer = SummaryWriter("runs/cells2")
torch.set_grad_enabled(True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transformCells = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToPILImage(), transforms.ToTensor()])
dataset = cellDataset("data/lowres/lowres_labels.csv", "data/highres/highres_labels.csv", "data/highres", "data/lowres", transform=transformCells)


#optimizer = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9)Adam
#optimizer = optim.RMSprop(network.parameters(), lr=0.01,  momentum=0.9, foreach=True)
#optimizer = optim.Adam(network.parameters(), lr=0.0001)

#nn.BCEWithLogitsLoss()
#network = Network().to(device)
params = OrderedDict(lr=[0.003], batch_size=[5], number_epocs=[50], criterion=[nn.MSELoss()], kernel_size=[5])
m = RunManager()

for run in RunBuilder.get_runs(params):
    network = Network(kernel_size=run.kernel_size).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=run.lr)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=run.batch_size)
    m.begin_run(run, network, dataloader)
    criterion = run.criterion

    for epoch in range(run.number_epocs):
        running_loss = 0.0
        m.begin_epoch()
        for i, highres in enumerate(dataloader, 0):
            batch = next(iter(dataloader))
            lowres, highres = batch['lowres'], batch['highres']
            lowres = lowres.to(device=device, dtype=torch.float32)
            highres = highres.to(device=device, dtype=torch.float32)

            outputs = network(lowres)

            loss = criterion(outputs, highres)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        m.end_epoch(running_loss)
        print(running_loss)
    m.end_run()
    torch.save(network.state_dict(), "model v1")


