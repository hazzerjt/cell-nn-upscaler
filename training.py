from torch.utils.data import DataLoader
from u_net import Network
from dataset import cellDataset
from run_manager import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms

torch.cuda.empty_cache()
device = torch.device("cuda")
params = OrderedDict(lr=[.01, 0.001], batch_size=[2], number_epocs=[2])
m = RunManager()
network = Network()
network.to(device)
dataset = cellDataset("data/lowres/lowres_labels.csv", "data/highres/highres_labels.csv", "data/highres", "data/lowres")
torch.set_grad_enabled(True)
loss_MSE = nn.MSELoss().to(device)



for run in RunBuilder.get_runs(params):
    dataloader = DataLoader(dataset, run.batch_size, shuffle=True)
    batch = next(iter(dataloader))
    optimizer = torch.optim.Adam(network.parameters(), lr=run.lr)
    m.begin_run(run, network, dataloader)


    for epoch in range(run.number_epocs):
        m.begin_epoch()
        for i in dataloader:
            lowres, highres = batch['lowres'], batch['highres']
            lowres.to(device)
            highres.to(device)

            #characteristics, labels = batch
            preds = network(lowres).to(device)  # Pass Batch
            #loss = F.cross_entropy(preds, highres)  # Calculate Loss
            loss = loss_MSE(preds, highres).to(device)
            optimizer.zero_grad()  # Zero Gradients
            loss.backward()  # Calculate Gradients
            optimizer.step()  # Update Weights
            m.track_loss(loss, lowres)
            #m.track_num_correct(preds, labels)
            torch.cuda.empty_cache()
        m.inform(2)
        torch.cuda.empty_cache()
        m.end_epoch()
    m.end_run()