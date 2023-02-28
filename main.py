import torch
from dataset import cellDataset
from u_net import Network
import matplotlib.pyplot as plt
from torchvision import transforms

transformCells = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToPILImage(), transforms.ToTensor()])
dataset = cellDataset("data/train.csv", transform=transformCells)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

torch.set_grad_enabled(True)

network = Network()



batch = next(iter(dataloader))
images, labels, index = batch
plt.imshow(images[0,0,:,:], cmap="gray")
plt.show()

print(labels)

preds = network(images)

preds = preds.to("cpu")
plt.imshow(preds[0,0,:,:].detach().numpy(), cmap="gray")
plt.show()