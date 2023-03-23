import torch
from dataset import cellDataset
from u_net import Network
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import cellDataset
from torch.utils.data import DataLoader

torch.set_grad_enabled(False)

transformCells = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToPILImage(), transforms.ToTensor()])
dataset = cellDataset("data/lowres/lowres_labels.csv", "data/highres/highres_labels.csv", "data/highres", "data/lowres")
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

network = Network(kernel_size=5)
network.load_state_dict(torch.load("model v1 sharpening"))
network.eval()

batch = next(iter(dataloader))
lowres, highres = batch['lowres'], batch['highres']

plt.imshow(lowres[0,0,:,:], cmap="gray")
plt.savefig("Lowres Image")
plt.show()
plt.imshow(highres[0,0,:,:], cmap="gray")
plt.savefig("Highres Image")
plt.show()
print(lowres.size())

preds = network(lowres)
print(preds.size())

plt.imshow(preds[0,0,:,:].detach().numpy(), cmap="gray")
plt.savefig("Upscaled Image")
plt.show()