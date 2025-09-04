import torchvision.transforms.v2 as v2
from torch.utils import data
import torch
import cv2

class Sun_glasses_Dataset(data.Dataset):
    def __init__(self, wear_data=None, no_wear_data=None, transform=None):
        self.wear_data = wear_data if wear_data is not None else []
        self.no_wear_data = no_wear_data if no_wear_data is not None else []
        self.transform = transform

        self.data = self.wear_data + self.no_wear_data
        
        wear_label = torch.ones(len(self.wear_data), dtype=torch.long)
        no_wear_label = torch.zeros(len(self.no_wear_data), dtype=torch.long)
        self.labels = torch.cat([wear_label, no_wear_label], dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_path = self.data[idx]
        try:
            image = cv2.imread(str(item_path))
            if image is None:
                # 이미지를 읽지 못한 경우 에러 처리
                raise FileNotFoundError(f"Image not found or unable to read: {item_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {item_path}: {e}")
            # 에러 발생 시, 임시 이미지나 다음 이미지로 넘어가는 대신 에러를 발생시켜 문제를 인지하도록 함
            raise e

        tensor_image = self.transform(image) if self.transform else v2.functional.to_tensor(image)

        label = self.labels[idx]
        return tensor_image, label

