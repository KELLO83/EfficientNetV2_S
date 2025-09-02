import logging
import torch
from torch.utils import data
from tqdm import tqdm
import wandb
import os
import pathlib
from dataset import Sun_glasses_Dataset
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
        project='EfficientNetV2_S',
        name=name,
    )

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

def main_worker(rank, world_size, args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    if rank == 0 and args.wandb:
        wandb_init(name=f'{args.wandb_name}')

    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # wear = pathlib.Path('/home/ubuntu/KOR_DATA/sunglass_dataset/wear')
    # no_wear = pathlib.Path('/home/ubuntu/KOR_DATA/sunglass_dataset/nowear')

    wear = pathlib.Path('/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/sunglass/refining_Yaw')

    no_wear = pathlib.Path('/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/earing/refining_Yaw')
    nowear_plus = pathlib.Path('/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/glasses/refining_Yaw')

    wear_list = list(wear.glob('**/*.jpg'))
    no_wear_list = list(no_wear.glob('**/*.jpg')) + list(nowear_plus.glob('**/*.jpg'))
    
    if wear_list == [] or no_wear_list == []:
        print("wear list {} or no wear list {} is empty.".format(wear_list, no_wear_list))
        raise ValueError("No image files found in the specified directories.")

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

    train_dataset = Sun_glasses_Dataset(wear_data=train_wear, no_wear_data=train_no_wear)
    val_dataset = Sun_glasses_Dataset(wear_data=val_wear, no_wear_data=val_no_wear)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, num_workers=os.cpu_count()//world_size, sampler=train_sampler)
    valloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=os.cpu_count()//world_size, sampler=val_sampler)

    if len(train_no_wear) > 0 and len(train_wear) > 0:
        weight_for_no_wear = len(train_wear) / len(train_no_wear)
        class_weights = torch.tensor([1.0, weight_for_no_wear], device=device)
    else:
        class_weights = None

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


    if args.model == 'efficientnetv2_s':
        model = EfficientNetV2_S().to(device)

    elif args.model == 'efficientnetv2_l':
        model = EfficientNetV2_L().to(device)

    if args.pretrained and os.path.isfile(args.weight_path):
        logging.info(f"Loading weights from {args.weight_path}")
        checkpoint = torch.load(args.weight_path, map_location=device)
        if hasattr(checkpoint, 'model_state_dict'):
            model_weight = checkpoint['model_state_dict']
        else:
            model_weight = checkpoint
        result = model.load_state_dict(model_weight)
        logging.info("Missing keys:", result.missing_keys)
        logging.info("Unexpected keys:", result.unexpected_keys)
        logging.info(f"{result}")
        

    torch.backends.cudnn.benchmark = True
    model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if rank == 0 and args.wandb:
        wandb.watch(model, log='all', log_freq=100)

    # if rank == 0:
    #     torchinfo.summary(
    #         model=model,
    #         input_size=(1, 3, 320, 320),
    #         verbose=True,
    #         col_names=["input_size", "output_size", "trainable"],
    #         row_settings=["depth"],
    #         mode='eval'
    #     )

    if rank == 0:
        backbone_params = sum(p.numel() for p in model.parameters())
        trainable_backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info('==' * 30)
        logging.info(f"Model total params: {backbone_params:,}")
        logging.info(f"Model trainable params: {trainable_backbone_params:,}")
        logging.info(f"traineable params percentage: {100 * trainable_backbone_params / backbone_params:.2f}%")
        logging.info('==' * 30)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
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

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                os.makedirs('checkpoints', exist_ok=True)


                checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
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
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--model', type=str, default='efficientnetv2_s', help='Model type: efficientnetv2_s or efficientnetv2_l')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained weights or not')
    parser.add_argument('--weight_path' ,type=str, default='checkpoints/best_mode.pth', help='Path to model weights')
    parser.add_argument('--find_batch_size', default=False, help='Find the maximum batch size before training')
    parser.add_argument('--wandb', action='store_true', default=False, help='Use wandb or not')
    parser.add_argument('--wandb_name', type=str, default='efficientnetv2_L', help='wandb experiment name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default= 256//2, help='Batch size for training')
    args = parser.parse_args()

    args.wandb = True

    if args.find_batch_size:
        print("Finding max batch size...")
        temp_model = EfficientNetV2_S()
        max_bs = find_max_batch_size(temp_model, input_shape=(3, 320, 320), device='cuda:0')
        if max_bs:
            print(f"Maximum possible batch size is {max_bs}. You can set --batch_size {max_bs} for the next run.")
            max_bs = max_bs 
            args.batch_size = max_bs

    for var in vars(args):
        print(f"{var} : {getattr(args, var)}")
    input("Please check the arguments. If you want to proceed, press Enter...")
    
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
