import matplotlib.pyplot as plt
import numpy as np
from dataset import cellDataset
import random

dataset = cellDataset("data/train.csv", transform=None)

plt.figure(figsize=(12,6))
for i in range(10):
    idx = random.randint(0, len(dataset))
    image, class_name, class_index = dataset[idx]
    ax=plt.subplot(2,5,i+1)
    ax.title.set_text(class_name + '-' + str(class_index))
    plt.imshow(image[0,:,:], cmap="gray")

plt.show()