import torch
from backbone.model import EfficientNetV2_S , EfficientNetV2_S_DANN , EfficientNetV2_L
import pathlib
import natsort
import torchvision.transforms.v2 as v2
import cv2
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict

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

#weight_path = 'effcientnet640_s_sunglasses/efficientnetv2_s_dann_best.pth'
weight_path = 'checkpoints/efficientnetv2_s_dann_best.pth'
# Load weights onto the CPU to avoid GPU memory issues
weight = torch.load(weight_path, map_location='cpu')

# if hasattr(weight, 'state_dict'):
#     weight = weight.state_dict()
# else:
#     weight = weight

try:
    if weight['model_state_dict'] : 
        weight = weight['model_state_dict']
except:
    pass

print("--- Keys from loaded weights ---")
for k in weight.keys():
    print(k)
print("---------------------------------")

# The model was compiled, so we need to remove the '_orig_od.' prefix
new_state_dict = OrderedDict()
for k, v in weight.items():
    if k.startswith('_orig_mod'):
        name = k.replace('_orig_mod.', '')
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v

print("--- Keys after cleaning ---")
for k in new_state_dict.keys():
    print(k)


model = EfficientNetV2_S_DANN(num_classes=2, num_domains=2)
# Load the cleaned state dict
#model = EfficientNetV2_S(num_classes=2)
result = model.load_state_dict(new_state_dict, strict=True)

print("--- Loading Model Weights ---")
print(f"Loading from: {weight_path}")
if result.missing_keys:
    print("Missing keys:", result.missing_keys)
if result.unexpected_keys:
    print("Unexpected keys:", result.unexpected_keys)

if not result.missing_keys and not result.unexpected_keys:
    print("Weights loaded successfully!")
else:
    print("Weights loaded with some mismatches.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(size=(640,640), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



wear = pathlib.Path('/home/ubuntu/KOR_DATA/high_resolution_wear_hat')
no_wear = pathlib.Path('/home/ubuntu/KOR_DATA/high_resolution_not_wear_hat')

# no_wear = pathlib.Path('/home/ubuntu/KOR_DATA/sunglass_dataset/nowear')
# no_wear = pathlib.Path('elfin_data')

wear_files = natsort.natsorted(list(wear.glob('**/*.jpg')))
nowear_files = natsort.natsorted(list(no_wear.glob('**/*.jpg')))



print(f"Found {len(wear_files)} images with item (positive class, label 1).")
print(f"Found {len(nowear_files)} images without item (negative class, label 0).")

if not wear_files or not nowear_files:
    print(len(wear_files), len(nowear_files))
    raise ValueError("No image files found in the specified directories.")

# wear_files = []

full_dataset = SunglassDataset(wear_files=wear_files, nowear_files=nowear_files, transform=transform)
data_loader = DataLoader(full_dataset, batch_size= 64, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

tp, tn, fp, fn = 0, 0, 0, 0

# Lists to store results for analysis
all_labels = []
all_preds = []
all_similarities = []

model.eval()
with torch.no_grad():
    for images, labels in tqdm(data_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        
        # DANN model returns (label_output, domain_output)
        # For inference, we only need the label_output and set alpha=0

        # if isinstance(model, EfficientNetV2_S_DANN):
        #     label_outputs, _ = model(images, alpha=0)
        # else:
        # label_outputs, _ = model(images, alpha=0)

        label_outputs  , _ = model(images)
        
        # --- 1. Get Predictions ---
        predicted = torch.argmax(label_outputs, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # --- 2. Get Similarities (Probabilities for the positive class) ---
        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(label_outputs, dim=1)
        # The similarity is the probability of the 'wear' class (class 1)
        similarities = probabilities[:, 1]
        all_similarities.extend(similarities.cpu().numpy())

# Convert lists to numpy arrays for easier analysis
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_similarities = np.array(all_similarities)

# --- Calculate Metrics ---
tp = np.sum((all_preds == 1) & (all_labels == 1))
tn = np.sum((all_preds == 0) & (all_labels == 0))
fp = np.sum((all_preds == 1) & (all_labels == 0))
fn = np.sum((all_preds == 0) & (all_labels == 1))

total = len(full_dataset)
print("--- Evaluation Complete ---")
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

# --- Plotting ---
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# 1. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
# Handle potential division by zero if a class has no samples
cm_sum = cm.sum(axis=1)[:, np.newaxis]
with np.errstate(divide='ignore', invalid='ignore'):
    cm_normalized = np.where(cm_sum > 0, cm.astype('float') / cm_sum, 0)

plt.figure(figsize=(15, 15))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.title('Normalized Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No-Wear', 'Wear'])
plt.yticks(tick_marks, ['No-Wear', 'Wear'])

# Add text annotations
thresh = cm_normalized.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
             horizontalalignment="center",
             color="white" if cm_normalized[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_annotated.png', bbox_inches='tight')
plt.show()


# 2. Similarity Distribution Plot
pos_similarities = all_similarities[all_labels == 1]
neg_similarities = all_similarities[all_labels == 0]

plt.figure(figsize=(10, 6))
plt.hist(neg_similarities, bins=50, alpha=0.7, label='Negative (No-Wear) Similarities', color='blue', density=True)
plt.hist(pos_similarities, bins=50, alpha=0.7, label='Positive (Wear) Similarities', color='red', density=True)

plt.title('Distribution of Similarity Scores')
plt.xlabel('Similarity Score (Probability of "Wear" class)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('similarity_distribution.png')
plt.show()
