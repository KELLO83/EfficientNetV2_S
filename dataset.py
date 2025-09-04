import torchvision.transforms.v2 as v2
from torch.utils import data
import torch
import cv2
import pathlib # Added for pathlib.Path
from typing import List, Tuple, Callable, Dict, Any # Added for type hints

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

class Domain_Sun_Glasses_Dataset(data.Dataset):
    """
    A dataset for Domain-Adversarial training.
    It returns an image, its class label (wear/no-wear), and its domain label.
    """
    def __init__(self, domain_A_wear_data=None, domain_A_nowear_data=None,
                 domain_B_wear_data=None, domain_B_nowear_data=None, transform=None):

        self.transform = transform

        # Consolidate paths and create corresponding labels
        self.paths = []
        self.class_labels = []
        self.domain_labels = []

        # Domain A (label 0)
        if domain_A_wear_data:
            self.paths.extend(domain_A_wear_data)
            self.class_labels.extend([1] * len(domain_A_wear_data))
            self.domain_labels.extend([0] * len(domain_A_wear_data))

        if domain_A_nowear_data:
            self.paths.extend(domain_A_nowear_data)
            self.class_labels.extend([0] * len(domain_A_nowear_data))
            self.domain_labels.extend([0] * len(domain_A_nowear_data))

        # Domain B (label 1)
        if domain_B_wear_data:
            self.paths.extend(domain_B_wear_data)
            self.class_labels.extend([1] * len(domain_B_wear_data))
            self.domain_labels.extend([1] * len(domain_B_wear_data))

        if domain_B_nowear_data:
            self.paths.extend(domain_B_nowear_data)
            self.class_labels.extend([0] * len(domain_B_nowear_data))
            self.domain_labels.extend([1] * len(domain_B_nowear_data))

        # Convert labels to tensors
        self.class_labels = torch.tensor(self.class_labels, dtype=torch.long)
        self.domain_labels = torch.tensor(self.domain_labels, dtype=torch.long)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        item_path = self.paths[idx]
        try:
            image = cv2.imread(str(item_path))
            if image is None:
                raise FileNotFoundError(f"Image not found or unable to read: {item_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {item_path}: {e}")
            raise e

        tensor_image = self.transform(image) if self.transform else v2.functional.to_tensor(image)

        class_label = self.class_labels[idx]
        domain_label = self.domain_labels[idx]

        return tensor_image, class_label, domain_label


class SPMLDataset(data.Dataset):
    """
    SPML (Single Positive Multi-Label) 학습을 위한 커스텀 데이터셋.
    단일 정수 라벨을 4개의 클래스에 대한 원-핫 인코딩 벡터로 변환합니다.

    클래스 정의:
    0: 평범한사람
    1: 모자쓴사람
    2: 선글라쓴사람
    3: 마스크착용한사람
    """
    def __init__(self,
                 file_paths: List[pathlib.Path],
                 labels: List[int],
                 transform: Callable = None):
        """
        데이터셋을 초기화합니다.

        Args:
            file_paths (List[pathlib.Path]): 이미지 파일 경로 리스트.
            labels (List[int]): 각 이미지에 해당하는 단일 정수 라벨 리스트 (0, 1, 2, 3).
            transform (Callable, optional): 이미지에 적용할 변환 함수. 기본값은 None.
        """
        if len(file_paths) != len(labels):
            raise ValueError("file_paths와 labels의 길이가 일치해야 합니다.")

        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

        self.class_names: List[str] = ["평범한사람", "모자쓴사람", "선글라쓴사람", "마스크착용한사람"]
        self.num_classes: int = len(self.class_names)


        if not all(0 <= label < self.num_classes for label in labels):
            raise ValueError(f"라벨은 0에서 {self.num_classes - 1} 사이의 정수여야 합니다.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(str(file_path))
        if image is None:
            raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if self.transform:
            image = self.transform(image)

        one_hot_label = torch.nn.functional.one_hot(torch.tensor(label, dtype=torch.long), num_classes=self.num_classes).float()

        return image, one_hot_label
