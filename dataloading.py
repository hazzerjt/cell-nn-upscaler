import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

dataset = datasets.ImageFolder('data\Cells', transforms.Compose([transforms.ToTensor()]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
