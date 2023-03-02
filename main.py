import torch
from dataset import cellDataset
from u_net import Network
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import cellDataset
from torch.utils.data import DataLoader

torch.set_grad_enabled(False)

transformCells = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToPILImage(), transforms.ToTensor()])
dataset = cellDataset("data/lowres/lowres_labels.csv", "data/highres/highres_labels.csv", "data/highres", "data/lowres", transform=transformCells)
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

network = Network()

batch = next(iter(dataloader))
lowres, highres = batch['lowres'], batch['highres']

images, labels = batch

plt.imshow(lowres[0,0,:,:], cmap="gray")
plt.show()
print(labels)

preds = network(lowres)

plt.imshow(preds[0,0,:,:].detach().numpy(), cmap="gray")
plt.show()