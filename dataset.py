import random
import numpy as np
import torchvision.transforms.v2 as v2
from torch.utils import data
import torch
import cv2
import pathlib # Added for pathlib.Path
from typing import List, Tuple, Callable, Dict, Any, Optional # Added for type hints


class custom_dataset_FDA(data.Dataset):
    def __init__(self, wear_data=None, no_wear_data=None , train = True , img_size =  320,
                 fda_images: Optional[List[pathlib.Path]] = None, fda_beta: float = 0.0, fda_prob: float = 0.0):
        self.train = train
        self.wear_data = wear_data
        self.no_wear_data = no_wear_data
        self.fda_images = fda_images or []
        self.fda_beta = max(0.0, fda_beta)
        self.fda_prob = max(0.0, min(1.0, fda_prob))
        self.train_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.08, hue=0.01)], p=0.25),
            v2.RandomGrayscale(p=0.05),

            v2.RandomApply([
                v2.RandomChoice([
                    v2.RandomAffine(degrees=6, translate=(0.015, 0.015), scale=(0.97, 1.03), shear=(-2, 2)),
                    v2.RandomPerspective(distortion_scale=0.08)
                ])
            ], p=0.6),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
            
            v2.Resize(size=(img_size, img_size)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.05, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0),
        ])
        
        self.val_transform  = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32 ,scale=True),
            v2.Resize(size=(img_size, img_size)),
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
        if self.train and self.fda_images and self.fda_beta > 0 and random.random() < self.fda_prob:
            image = self._apply_fda(image)
        if self.train:
            tensor_image = self.train_transform(image)
        else:
            tensor_image = self.val_transform(image)

        label = self.labels[idx]
        return tensor_image , label

    def _apply_fda(self, src_img: np.ndarray) -> np.ndarray:
        """Apply Fourier Domain Adaptation using a random image from the FDA pool."""
        style_path = random.choice(self.fda_images)
        style_img = cv2.imread(str(style_path))
        if style_img is None:
            return src_img

        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        h, w = src_img.shape[:2]
        if h == 0 or w == 0:
            return src_img

        style_img = cv2.resize(style_img, (w, h), interpolation=cv2.INTER_LINEAR)

        return self._fda_source_to_target(src_img, style_img, self.fda_beta)

    @staticmethod
    def _fda_source_to_target(src_img: np.ndarray, tgt_img: np.ndarray, beta: float) -> np.ndarray:
        src_img = src_img.astype(np.float32)
        tgt_img = tgt_img.astype(np.float32)

        src_fft = np.fft.fft2(src_img, axes=(0, 1))
        tgt_fft = np.fft.fft2(tgt_img, axes=(0, 1))

        src_amp, src_phase = np.abs(src_fft), np.angle(src_fft)
        tgt_amp = np.abs(tgt_fft)

        src_amp_shift = np.fft.fftshift(src_amp, axes=(0, 1))
        tgt_amp_shift = np.fft.fftshift(tgt_amp, axes=(0, 1))

        h, w = src_img.shape[:2]
        if beta <= 0:
            return src_img

        radius = max(1, int(np.floor(min(h, w) * beta)))
        c_h, c_w = h // 2, w // 2
        h1 = max(0, c_h - radius)
        h2 = min(h, c_h + radius)
        w1 = max(0, c_w - radius)
        w2 = min(w, c_w + radius)

        if h1 < h2 and w1 < w2:
            src_amp_shift[h1:h2, w1:w2] = tgt_amp_shift[h1:h2, w1:w2]

        src_amp = np.fft.ifftshift(src_amp_shift, axes=(0, 1))
        mixed_fft = src_amp * np.exp(1j * src_phase)
        mixed_img = np.fft.ifft2(mixed_fft, axes=(0, 1))
        mixed_img = np.real(mixed_img)
        mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
        return mixed_img


class custom_dataset_FDA_CORAL(data.Dataset):
    def __init__(self, wear_data=None, no_wear_data=None , train = True , img_size =  320,
                 fda_images: Optional[List[pathlib.Path]] = None, fda_beta: float = 0.0, fda_prob: float = 0.0,
                 return_domain: bool = False):
        self.train = train
        self.wear_data = wear_data
        self.no_wear_data = no_wear_data
        self.fda_images = fda_images or []
        self.fda_beta = max(0.0, fda_beta)
        self.fda_prob = max(0.0, min(1.0, fda_prob))
        self.return_domain = return_domain
        self.train_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.08, hue=0.01)], p=0.25),
            v2.RandomGrayscale(p=0.05),

            v2.RandomApply([
                v2.RandomChoice([
                    v2.RandomAffine(degrees=6, translate=(0.015, 0.015), scale=(0.97, 1.03), shear=(-2, 2)),
                    v2.RandomPerspective(distortion_scale=0.08)
                ])
            ], p=0.6),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
            
            v2.Resize(size=(img_size, img_size)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.05, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0),
        ])
        
        self.val_transform  = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32 ,scale=True),
            v2.Resize(size=(img_size, img_size)),
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
        applied_fda = False
        if self.train and self.fda_images and self.fda_beta > 0 and random.random() < self.fda_prob:
            image = self._apply_fda(image)
            applied_fda = True
        if self.train:
            tensor_image = self.train_transform(image)
        else:
            tensor_image = self.val_transform(image)

        label = self.labels[idx]
        if self.return_domain:
            domain_label = torch.tensor(1 if applied_fda else 0, dtype=torch.long)
            return tensor_image, label, domain_label
        return tensor_image , label

    def _apply_fda(self, src_img: np.ndarray) -> np.ndarray:
        """Apply Fourier Domain Adaptation using a random image from the FDA pool."""
        style_path = random.choice(self.fda_images)
        style_img = cv2.imread(str(style_path))
        if style_img is None:
            return src_img

        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        h, w = src_img.shape[:2]
        if h == 0 or w == 0:
            return src_img

        style_img = cv2.resize(style_img, (w, h), interpolation=cv2.INTER_LINEAR)

        return self._fda_source_to_target(src_img, style_img, self.fda_beta)

    @staticmethod
    def _fda_source_to_target(src_img: np.ndarray, tgt_img: np.ndarray, beta: float) -> np.ndarray:
        src_img = src_img.astype(np.float32)
        tgt_img = tgt_img.astype(np.float32)

        src_fft = np.fft.fft2(src_img, axes=(0, 1))
        tgt_fft = np.fft.fft2(tgt_img, axes=(0, 1))

        src_amp, src_phase = np.abs(src_fft), np.angle(src_fft)
        tgt_amp = np.abs(tgt_fft)

        src_amp_shift = np.fft.fftshift(src_amp, axes=(0, 1))
        tgt_amp_shift = np.fft.fftshift(tgt_amp, axes=(0, 1))

        h, w = src_img.shape[:2]
        if beta <= 0:
            return src_img

        radius = max(1, int(np.floor(min(h, w) * beta)))
        c_h, c_w = h // 2, w // 2
        h1 = max(0, c_h - radius)
        h2 = min(h, c_h + radius)
        w1 = max(0, c_w - radius)
        w2 = min(w, c_w + radius)

        if h1 < h2 and w1 < w2:
            src_amp_shift[h1:h2, w1:w2] = tgt_amp_shift[h1:h2, w1:w2]

        src_amp = np.fft.ifftshift(src_amp_shift, axes=(0, 1))
        mixed_fft = src_amp * np.exp(1j * src_phase)
        mixed_img = np.fft.ifft2(mixed_fft, axes=(0, 1))
        mixed_img = np.real(mixed_img)
        mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
        return mixed_img


class custom_dataset(data.Dataset):
    def __init__(self, wear_data=None, no_wear_data=None , train = True , img_size =  320):
        self.train = train
        self.wear_data = wear_data
        self.no_wear_data = no_wear_data
        self.train_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.08, hue=0.01)], p=0.25),

            v2.RandomApply([
                v2.RandomChoice([
                    v2.RandomAffine(degrees=6, translate=(0.015, 0.015), scale=(0.97, 1.03), shear=(-2, 2)),
                    v2.RandomPerspective(distortion_scale=0.08)
                ])
            ], p=0.6),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),

            v2.Resize(size=(img_size, img_size)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.05, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0),
        ])
        
        self.val_transform  = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32 ,scale=True),
            v2.Resize(size=(img_size, img_size)),
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



class BalancedDomainDataset(data.Dataset):
    """
    A dataset designed to work with BalancedBatchSampler.

    It takes four lists of data paths, concatenates them, and assigns labels.
    It also stores the lengths of each data source for the sampler.
    """
    def __init__(self, wear_data1, wear_data2, no_wear_data1, no_wear_data2, train=True):
        self.train = train
        
        self.data_sources = [wear_data1, wear_data2, no_wear_data1, no_wear_data2]
        self.lengths = [len(d) for d in self.data_sources]
        self.cumulative_lengths = torch.cumsum(torch.tensor(self.lengths), dim=0).tolist()

        self.data = [item for sublist in self.data_sources for item in sublist]

        # Create labels: 1 for wear, 0 for no-wear
        wear_labels1 = torch.ones(len(wear_data1), dtype=torch.long)
        wear_labels2 = torch.ones(len(wear_data2), dtype=torch.long)
        no_wear_labels1 = torch.zeros(len(no_wear_data1), dtype=torch.long)
        no_wear_labels2 = torch.zeros(len(no_wear_data2), dtype=torch.long)
        self.labels = torch.cat([wear_labels1, wear_labels2, no_wear_labels1, no_wear_labels2], dim=0)

        self.train_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.3),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
            v2.Resize(size=(640, 640)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(640, 640)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = cv2.imread(str(item))
            if image is None:
                raise FileNotFoundError(f"Image not found or unable to read: {item}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {item}: {e}")
            # Return a placeholder tensor and label to avoid crashing the training loop
            return torch.randn(3, 640, 640), torch.tensor(-1, dtype=torch.long)

        if self.train:
            tensor_image = self.train_transform(image)
        else:
            tensor_image = self.val_transform(image)

        label = self.labels[idx]
        return tensor_image, label

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
