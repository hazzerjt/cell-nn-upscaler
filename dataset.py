from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image

class cellDataset(Dataset):
    def __init__(self, csv_file, root_dir="", image_size="both", transform=None):
        self.image_size = image_size
        self.annotation_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation_df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.annotation_df.iloc[idx, 1])
        image = read_image(image_path)
        class_name = self.annotation_df.iloc[idx, 2]
        class_index = self.annotation_df.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, class_name, class_index
