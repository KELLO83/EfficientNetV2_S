import logging
import torch
from torch.utils import data
from tqdm import tqdm
import wandb
import os
import pathlib
from dataset import custom_dataset
import numpy as np
import cv2
import logging
import os
from backbone.model import EfficientNetV2_S , EfficientNetV2_L
import torchinfo
from sklearn.model_selection import train_test_split
from utils.early_stop import EarlyStopping
from utils.polynomialLRWarmup import PolynomialLRWarmup
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from collections import OrderedDict
from natsort import natsorted
from torch.utils.data import DataLoader
import torchvision
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def wandb_init(name='None'):
    name = name
    wandb.init(
        project='EfficientNetV2_mask',
        name=name,
    )

def data_check(loader: DataLoader, is_bgr: bool):
    """
    Visualizes a batch of images from the DataLoader to check augmentations.
    Uses inverse ImageNet normalization to display.
    """
    logging.info("Fetching a batch of data to visualize...")
    it = iter(loader)

    # Inverse of Normalize(mean, std) used in dataset.py
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    cv2.namedWindow('Data Check - Press any key to continue', cv2.WINDOW_NORMAL)
    for images, _ in it:
        images_unnorm = images.clone()
        images_unnorm.mul_(std).add_(mean)
        grid = torchvision.utils.make_grid(images_unnorm, nrow=8)
        grid = torch.clamp(grid, 0, 1)

        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np * 255).astype(np.uint8)

        if not is_bgr:
            grid_np = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Data Check - Press any key to continue', grid_np)
        key = cv2.waitKey(0)
        if key == 27:
            logging.info("ESC key pressed. Exiting data check.")
            break

    cv2.destroyAllWindows()
    return

def find_max_batch_size(model, input_shape, device):
    if 'cuda' not in str(device):
        logging.info("CUDA not found. Skipping max batch size search.")
        return None

    model.to(device)
    model.eval()
    max_found_bs = 0
    mid = 64
    limit = 1024 * 5
    while max_found_bs <= limit:
        if mid == 64:
            pass
        else:
            mid = max_found_bs * 2
        try:
            dummy_input = torch.randn(mid, *input_shape, device=device)
            with torch.no_grad():
                _ = model(dummy_input)
            max_found_bs = mid

            print(f"✅ Batch size {mid} succeeded.")
            mid = mid * 2
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"❌ Batch size {mid} failed (OOM).")
                return max_found_bs
            else:
                print(f"An unexpected error occurred: {e}")
                raise e
        finally:
            torch.cuda.empty_cache()
    print(f"Max batch size search completed. Maximum found 한계선 {limit}: {max_found_bs}")
    return max_found_bs

def get_model(model_name):
    if model_name == 'efficientnetv2_s':
        model = EfficientNetV2_S()
    elif model_name == 'efficientnetv2_l':
        model = EfficientNetV2_L()
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model

def load_extra_data(path, fraction):
    p = pathlib.Path(path)
    if not p.exists():
        logging.warning(f"Extra data path not found: {path}")
        return []
    subfolders = natsorted([f for f in p.iterdir() if f.is_dir()])
    num_folders_to_use = int(len(subfolders) * fraction)
    selected_folders = subfolders[:num_folders_to_use]
    selected_images = []
    for folder in selected_folders:
        selected_images.extend(natsorted(list(folder.glob('*.jpg'))))
    if not selected_images:
        logging.warning(f"No image files found in the specified extra data directory: {path}")
    return selected_images

    

def main_worker(rank, world_size, args):
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    if rank == 0 and args.wandb:
        wandb_init(name=f'{args.wandb_name}')

    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    wear_path = pathlib.Path(args.wear_dir)
    nowear_path = pathlib.Path(args.nowear_dir)

    wear_list = list(wear_path.glob('*.jpg')) if wear_path.exists() else []
    no_wear_list = list(nowear_path.glob('**/*.jpg')) if nowear_path.exists() else []

    if args.data_fraction > 0:
        if rank == 0:
            logging.info(f"Loading {args.data_fraction * 100:.0f}% of extra data from {args.wear_dir2} and {args.nowear_dir2}")
        no_wear_list.extend(load_extra_data(args.nowear_dir2, args.data_fraction))


    wear2_path = pathlib.Path(args.wear_dir2)

    wear_list.extend(list(wear2_path.glob('**/*.jpg')) if wear2_path.exists() else [])


    if not wear_list or not no_wear_list:
        raise ValueError("Image lists are empty. Please check data paths.")

    if rank == 0:
        logging.info(f"wear num : {len(wear_list)}")
        logging.info(f"no wear num : {len(no_wear_list)}")

    wear_labels = [1] * len(wear_list)
    no_wear_labels = [0] * len(no_wear_list)

    all_data = wear_list + no_wear_list
    all_labels = wear_labels + no_wear_labels

    train_list, val_list, train_labels, val_labels = train_test_split(
        all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    train_wear = [path for path, label in zip(train_list, train_labels) if label == 1]
    train_no_wear = [path for path, label in zip(train_list, train_labels) if label == 0]

    val_wear = [path for path, label in zip(val_list, val_labels) if label == 1]
    val_no_wear = [path for path, label in zip(val_list, val_labels) if label == 0]

    train_dataset = custom_dataset(wear_data=train_wear, no_wear_data=train_no_wear , img_size= args.img_size , train=True)
    val_dataset = custom_dataset(wear_data=val_wear, no_wear_data=val_no_wear , train=False , img_size= args.img_size)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, num_workers=os.cpu_count()//world_size, sampler=train_sampler)
    valloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=os.cpu_count()//world_size, sampler=val_sampler)

    # Run data_check safely under DDP (only rank 0 shows window)

    if args.data_check and rank == 0:
        data_check(trainloader, is_bgr=False)
        if world_size > 1:
            dist.barrier()
        
    if len(train_no_wear) > 0 and len(train_wear) > 0:
        count_no_wear = len(train_no_wear)
        count_wear = len(train_wear)
        
        # Calculate weight for each class to handle imbalance.
        # The weight for a class is inversely proportional to its number of samples.
        # This gives higher penalty for misclassifying the minority class.
        weight_for_class_0 = (count_no_wear + count_wear) / (2.0 * count_no_wear)
        weight_for_class_1 = (count_no_wear + count_wear) / (2.0 * count_wear)
        
        # The order in the tensor must match the class indices (0, 1)
        class_weights = torch.tensor([weight_for_class_0, weight_for_class_1], device=device)
        
        if rank == 0:
            logging.info(f"Handling class imbalance. Samples: [No-Wear: {count_no_wear}, Wear: {count_wear}]")
            logging.info(f"Applying weights: [No-Wear: {weight_for_class_0:.2f}, Wear: {weight_for_class_1:.2f}]")
    else:
        class_weights = None

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = get_model(args.model).to(device)
    model = torch.compile(model)

    if args.pretrained and os.path.isfile(args.weight_path):
        logging.info(f"Loading weights from {args.weight_path}")
        checkpoint = torch.load(args.weight_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model_weight = checkpoint['model_state_dict']
        else:
            model_weight = checkpoint
            
        result = model.load_state_dict(model_weight, strict=False)
        logging.info(f"Weight loading result: {result}")

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if rank == 0 and args.wandb:
        wandb.watch(model, log='all', log_freq=100)

    if rank == 0:
        backbone_params = sum(p.numel() for p in model.parameters())
        trainable_backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info('==' * 30)
        logging.info(f"Model total params: {backbone_params:,}")
        logging.info(f"Model trainable params: {trainable_backbone_params:,}")
        logging.info(f"traineable params percentage: {100 * trainable_backbone_params / backbone_params:.2f}%")
        logging.info('==' * 30)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    #scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    scheduler = PolynomialLRWarmup(
        optimizer, warmup_iters=10, total_iters=args.epochs, power=1.0 , limit_lr = 1e-7
    )

    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_val_acc = 0.0
    scaler = torch.amp.GradScaler()
    
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar_train = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=(rank != 0))
        for data_input, label in pbar_train:
            data_input = data_input.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type):
                logit = model(data_input)
                loss = criterion(logit, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(logit.data, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()
            
            if rank == 0:
                pbar_train.set_postfix({'loss': loss.item()})

        train_accuracy = 100 * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            pbar_val = tqdm(valloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", disable=(rank != 0))
            for data_input, label in pbar_val:
                data_input = data_input.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type):
                    logit = model(data_input)
                    loss = criterion(logit, label)

                val_loss += loss.item()
                _, predicted = torch.max(logit.data, 1)
                total_val += label.size(0)
                correct_val += (predicted == label).sum().item()
                if rank == 0:
                    pbar_val.set_postfix({'loss': loss.item()})

        val_accuracy = 100 * correct_val / total_val
        
        if rank == 0 and args.wandb:
            wandb.log({
               "Train Loss": train_loss / len(trainloader),
               "Train Accuracy": train_accuracy,
               "Val Loss": val_loss / len(valloader),
               "Val Accuracy": val_accuracy,
               "Learning Rate": scheduler.get_last_lr()[0]
            })

        if rank == 0:
            logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss/len(trainloader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss/len(valloader):.4f}, Val Acc: {val_accuracy:.2f}%")


            if (epoch + 1) % 2 == 0 :
                state_to_save = model.module.state_dict() if world_size > 1 else model.state_dict()
                
                cleaned_state_dict = OrderedDict()
                for k, v in state_to_save.items():
                    if '_orig_mod.' in k:
                        name = k.replace('_orig_mod.', '')
                    else:
                        name = k
                    cleaned_state_dict[name] = v
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': cleaned_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc
                }

                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                    
                torch.save(checkpoint, os.path.join('checkpoints' , f'{args.model}_{epoch+1}.pth'))
                logging.info("Saved best model and optimizer state.")
                logging.info(f"Saved checkpoint at epoch {epoch+1}.")


            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                os.makedirs('checkpoints', exist_ok=True)

                state_to_save = model.module.state_dict() if world_size > 1 else model.state_dict()
                
                cleaned_state_dict = OrderedDict()
                for k, v in state_to_save.items():
                    if '_orig_mod' in k:
                        name = k.replace('_orig_mod.', '')
                    else:
                        name = k
                    cleaned_state_dict[name] = v
                

                checkpoint = {
                'epoch': epoch,
                'model_state_dict': cleaned_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc
                }
                torch.save(checkpoint, os.path.join('checkpoints','best_model.pth'))
                logging.info("Saved best model and optimizer state.")

            early_stopping(val_loss / len(valloader))

        if world_size > 1:
            stop_signal = torch.tensor(1.0 if rank == 0 and early_stopping.early_stop else 0.0, device=device)
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1.0:
                if rank == 0:
                    logging.info("Early stopping triggered.")
                break
        else:
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break
        
        scheduler.step()

    if world_size > 1:
        cleanup_ddp()




def main():
    parser = argparse.ArgumentParser(description='EfficientNetV2 Training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='efficientnetv2_s', choices=['efficientnetv2_s', 'efficientnetv2_l'], help='Model type')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--weight_path', type=str, default='checkpoints/best_model.pth', help='Path to model weights')

    # Data arguments
    parser.add_argument('--wear_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/마스크_원시데이터/TS1')
    parser.add_argument('--wear_dir2' , type=str , default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/마스크_원시데이터/TS2')
    parser.add_argument('--nowear_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/neckslice/refining_yaw_yaw', help='Directory for "no wear" images')
    parser.add_argument('--nowear_dir2', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/glasses/refining_yaw_yaw', help='Directory for additional "no wear" images')
    parser.add_argument('--data_fraction', type=float, default=1, help='Fraction of extra data to use (0.0 to 1.0)')
    parser.add_argument('--img_size', type=int, default=384, help='Input image size (assumed square)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--find_batch_size', action='store_true', help='Find the maximum batch size before training')
    parser.add_argument('--no_confirm', action='store_true', help='Skip argument confirmation prompt')
    parser.add_argument('--data_check', action='store_true', help='Visualize a batch of data to check augmentations')

    # W&B arguments
    parser.add_argument('--wandb', action='store_true', help='Use wandb or not')
    parser.add_argument('--wandb_name', type=str, default='efficientnetv2_experiment', help='wandb experiment name')
    
    args = parser.parse_args()

    if args.find_batch_size:
        print("Finding max batch size...")
        temp_model = get_model(args.model)
        max_bs = find_max_batch_size(temp_model, input_shape=(3, 320, 320), device='cuda:0')
        if max_bs:
            print(f"Maximum possible batch size is {max_bs}. Setting batch_size to this value.")
            args.batch_size = max_bs

    print("--- Training Arguments ---")
    for var in vars(args):
        print(f"{var}: {getattr(args, var)}")
    print("--------------------------")
    
    if not args.no_confirm:
        input("Press Enter to proceed...")
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        logging.info(f"Using {world_size} GPUs for DDP.")
        mp.spawn(main_worker,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    else:
        logging.info("Using single GPU.")
        main_worker(0, 1, args)

if __name__ == '__main__':
    main()
