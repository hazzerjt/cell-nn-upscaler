#from dataset import cellDataset
from dataset_test import cellDataset
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
from torch import optim

transformCells = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToPILImage(), transforms.ToTensor()])
dataset = cellDataset("data/train.csv", transform=transformCells, image_size="low")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)


batch = iter(dataloader)
print(batch)
images, labels, index = batch

#for i in range(0, 5):
    #if (labels==1):
    #if batch[2] == 1:
        #images, labels, index = batch
        #plt.imshow(images[0,0,:,:], cmap="gray")