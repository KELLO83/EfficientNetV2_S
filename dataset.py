import torchvision.transforms.v2 as v2
from torch.utils import data
import torch
import cv2

class Sun_glasses_Dataset(data.Dataset):
    def __init__(self, wear_data=None, no_wear_data=None , train = True):
        self.train = train
        self.wear_data = wear_data
        self.no_wear_data = no_wear_data
        self.train_transform  = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32 ,scale=True),
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.3),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
            v2.Resize(size=(320, 320)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform  = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32 ,scale=True),
            v2.Resize(size=(320, 320)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        wear_label = torch.ones(len(self.wear_data), dtype=torch.long)
        no_wear_label = torch.zeros(len(self.no_wear_data), dtype=torch.long)
        self.labels = torch.cat([wear_label, no_wear_label], dim=0)

        self.data = self.wear_data + self.no_wear_data
    def __len__(self):
        return len(self.wear_data) + len(self.no_wear_data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = cv2.imread(str(item))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            tensor_image = self.train_transform(image)
        else:
            tensor_image = self.val_transform(image)

        label = self.labels[idx]
        return tensor_image , label

