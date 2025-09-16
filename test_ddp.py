import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
import argparse
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

class custom_dataset(Dataset):
    """
    Custom Dataset for  wear/no-wear classification.
    - Label 1:  worn (Positive)
    - Label 0:  not worn (Negative)
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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # Use a different port from training
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def main_worker(rank, world_size, args):
    if world_size > 1:
        setup_ddp(rank, world_size)

    main_process = (rank == 0)
    local_rank = rank

    weight_path = args.weights
    weight = torch.load(weight_path, map_location='cpu')

    try:
        if weight['model_state_dict'] :
            weight = weight['model_state_dict']
    except:
        pass

    if main_process:
        print("--- Keys from loaded weights ---")
        for k in weight.keys():
            print(k)
        print("---------------------------------")

    new_state_dict = OrderedDict()
    for k, v in weight.items():
        if k.startswith('_orig_mod'):
            name = k.replace('_orig_mod.', '')
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    if main_process:
        print("--- Keys after cleaning ---")
        for k in new_state_dict.keys():
            print(k)

    if args.model == 's_dann':
        model = EfficientNetV2_S_DANN(num_classes=2, num_domains=2)
    elif args.model == 'l':
        model = EfficientNetV2_L(num_classes=2)
    else:
        model = EfficientNetV2_S(num_classes=2)

    result = model.load_state_dict(new_state_dict, strict=True)

    if main_process:
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

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
    model = model.to(device)
    model.eval()

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(args.img_size, args.img_size), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    wear = pathlib.Path(args.wear)
    no_wear = pathlib.Path(args.nowear)

    wear_files = natsort.natsorted(list(wear.glob('**/*.jpg')))
    nowear_files = natsort.natsorted(list(no_wear.glob('**/*.jpg')))

    if main_process:
        print(f"Found {len(wear_files)} images with item (positive class, label 1).")
        print(f"Found {len(nowear_files)} images without item (negative class, label 0).")

    if not wear_files or not nowear_files:
        if main_process:
            print(len(wear_files), len(nowear_files))
            raise ValueError("No image files found in the specified directories.")

    full_dataset = custom_dataset(wear_files=wear_files, nowear_files=nowear_files, transform=transform)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    data_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        num_workers=os.cpu_count() // world_size if world_size > 1 else os.cpu_count(),
        pin_memory=True,
    )

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    all_labels = []
    all_preds = []
    all_similarities = []

    model.eval()
    
    if main_process:
        fp = 0
        fn = 0

    pbar = tqdm(total=len(data_loader), disable=not main_process)

    with torch.no_grad():
        if sampler is not None:
            sampler.set_epoch(0)
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            label_outputs = model(images)
            if isinstance(label_outputs, (tuple, list)):
                label_outputs = label_outputs[0]

            predicted = torch.argmax(label_outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            probabilities = torch.softmax(label_outputs, dim=1)
            similarities = probabilities[:, 1]
            all_similarities.extend(similarities.cpu().numpy())

            fp_batch = torch.sum((predicted == 1) & (labels == 0))
            fn_batch = torch.sum((predicted == 0) & (labels == 1))

            if is_dist_avail_and_initialized():
                dist.all_reduce(fp_batch, op=dist.ReduceOp.SUM)
                dist.all_reduce(fn_batch, op=dist.ReduceOp.SUM)

            if main_process:
                fp += fp_batch.item()
                fn += fn_batch.item()
                pbar.set_postfix(OrderedDict(fp=fp, fn=fn))
                pbar.update(1)

    pbar.close() if main_process else None

    if world_size > 1:
        gathered_labels = [None for _ in range(world_size)]
        gathered_preds = [None for _ in range(world_size)]
        gathered_sims = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_labels, all_labels)
        dist.all_gather_object(gathered_preds, all_preds)
        dist.all_gather_object(gathered_sims, all_similarities)

        if main_process:
            all_labels = np.array([x for sub in gathered_labels for x in sub])
            all_preds = np.array([x for sub in gathered_preds for x in sub])
            all_similarities = np.array([x for sub in gathered_sims for x in sub])
    else:
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_similarities = np.array(all_similarities)

    if main_process:
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        tn = np.sum((all_preds == 0) & (all_labels == 0))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))

        total = len(all_labels)
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

        cm = confusion_matrix(all_labels, all_preds)
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

    if world_size > 1:
        cleanup_ddp()

def parse_args():
    parser = argparse.ArgumentParser(description='DDP evaluation with mp.spawn')
    parser.add_argument('--wear', type=str, help='Directory with positive class images (wear)')
    parser.add_argument('--nowear', type=str, help='Directory with negative class images (no-wear)')
    parser.add_argument('--img-size', type=int, default=640, help='Square image size for resizing')
    parser.add_argument('--model', type=str, default='s', choices=['s', 's_dann', 'l'], help='Model variant: s, s_dann, or l')
    parser.add_argument('--weights', type=str, default='checkpoints/best_model.pth', help='Path to model weights .pth')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per process')

    args = parser.parse_args()

    # Hardcoded paths as a fallback
    if not args.wear:
        args.wear = '/media/ubuntu/새 볼륨/dataset/034.마스크 착용 한국인 안면 이미지 데이터/01.데이터/1.Training/원천데이터/yolo_face_detection_result_1'
    if not args.nowear:
        args.nowear = '/home/ubuntu/KOR_DATA/high_resolution_not_wear_hat'
    
    return args

def main():
    args = parse_args()
    print("--- Evaluation Arguments ---")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------")

    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Spawning {world_size} processes for DDP evaluation.")
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("Running on single device.")
        main_worker(0, 1, args)

if __name__ == '__main__':
    main()
