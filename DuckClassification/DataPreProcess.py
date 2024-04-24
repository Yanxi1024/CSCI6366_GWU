import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms

class DuckDataSet(Dataset):

    def __init__(self, path, target = "not_duck", transform = None) -> None:
        # self.features = []; self.targets = []
        self.imagesNameList = [f"{path}{name}" for name in os.listdir(path)]
        self.length = len(self.imagesNameList)
        self.targets = self.length * ([1] if target == "duck" else [0])

        self.transform = transform

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image = Image.open(self.imagesNameList[index])
        if self.transform:
            image = self.transform(image.convert("RGB"))

        return image, self.targets[index]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
