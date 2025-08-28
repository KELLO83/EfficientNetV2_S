import torchvision.transforms.v2 as v2
from torch.utils import data
import torch
import pathlib
import cv2

class Sun_glasses_Dataset(data.Dataset):
    def __init__(self, wear_data=None, no_wear_data=None):
        self.wear_data = wear_data
        self.no_wear_data = no_wear_data
        self.transform  = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32 ,scale=True),
            v2.Resize(size=(320, 320)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        wear_label = torch.zeros(len(self.wear_data), dtype=torch.long)
        no_wear_label = torch.ones(len(self.no_wear_data), dtype=torch.long)
        self.labels = torch.cat([wear_label, no_wear_label], dim=0)

        self.data = self.wear_data + self.no_wear_data
    def __len__(self):
        return len(self.wear_data) + len(self.no_wear_data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = cv2.imread(str(item))
        tensor_image = self.transform(image)

        label = self.labels[idx]
        return tensor_image , label

