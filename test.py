import torch
from backbone.model import EfficientNetV2_S
import pathlib
import natsort
import torchvision.transforms.v2 as v2
import cv2
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class SunglassDataset(Dataset):
    """
    Custom Dataset for sunglass wear/no-wear classification.
    - Label 1: Sunglass worn (Positive)
    - Label 0: Sunglass not worn (Negative)
    """
    def __init__(self, wear_files, nowear_files, transform=None):
        self.transform = transform
        self.file_list = wear_files + nowear_files
        self.labels = [1] * len(wear_files) + [0] * len(nowear_files)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        

        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        if self.transform:
            image = self.transform(image)
            
        return image, label

weight_path = 'best_model.pth'
model = EfficientNetV2_S()
model.load_state_dict(torch.load(weight_path))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(size=(320, 320), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# wear_path = pathlib.Path('/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/sunglass/refining_Yaw')
# nowear_path = pathlib.Path('/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/earing/refining_Yaw')

wear = pathlib.Path('/home/ubuntu/KOR_DATA/sunglass_dataset/wear')
no_wear = pathlib.Path('/home/ubuntu/KOR_DATA/sunglass_dataset/nowear')

wear_files = natsort.natsorted(list(wear.glob('*.jpg')))
nowear_files = natsort.natsorted(list(no_wear.glob('*.jpg')))

print(f"Found {len(wear_files)} images with sunglasses (positive class, label 1).")
print(f"Found {len(nowear_files)} images without sunglasses (negative class, label 0).")


full_dataset = SunglassDataset(wear_files=wear_files, nowear_files=nowear_files, transform=transform)
data_loader = DataLoader(full_dataset, batch_size=128, shuffle=False, num_workers=4)

tp, tn, fp, fn = 0, 0, 0, 0

with torch.no_grad():
    all_labels = []
    all_preds = []
    for images, labels in tqdm(data_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    tp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 1))
    tn = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 0))
    fp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))
    fn = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1))




total = len(full_dataset)
print("\n--- Evaluation Complete ---")
print(f"Total Images: {total}")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

accuracy = (tp + tn) / total if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5, 4))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
im = plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.colorbar(im)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
plt.savefig('confusion_matrix.png')