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
import timm
from backbone.model import EfficientNetV2_S

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def wandb_init(Config):
    name = Config.name
    del Config.name
    wandb.init(
        project='EfficientNetV2_S',
        name=name,        
        config=Config,
    )


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wear = pathlib.Path('/home/ubuntu/KOR_DATA/sunglass_dataset/wear')
    no_wear = pathlib.Path('/home/ubuntu/KOR_DATA/sunglass_dataset/nowear')

    wear_list = list(wear.glob('**/*.jpg'))
    no_wear_list = list(no_wear.glob('**/*.jpg'))
    print("wear num : ", len(wear_list))
    print("no wear num : ", len(no_wear_list))

    wear_num = len(wear_list)
    no_wear_num = len(no_wear_list)

    dataset = Sun_glasses_Dataset(wear_data=wear_list, no_wear_data=no_wear_list)
    trainloader = data.DataLoader(dataset, batch_size=1, shuffle=True , pin_memory=True , num_workers=os.cpu_count())

    pos_weight_value = no_wear_num / wear_num

    pos_weight = torch.tensor([pos_weight_value])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    model = EfficientNetV2_S()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(100):

        model.train()
        train_loss = 0.0
        
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}/{100-1} [Train]")
        for ii, (data_input, label) in pbar:
            data_input = data_input.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).long()

            with torch.amp.autocast(device_type='cuda', enabled=True):
                logit = model(data_input)
                loss = criterion(logit, label)
                print("loss : ", loss.item())


def visual_loader(iter, mean, std):
    """ Dataloader 확인용 tensor 정상 매핑확인"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i, (item , label) in enumerate(iter):
        image_tensor = item[0]
        image_numpy = image_tensor.cpu().numpy()
        image_numpy = image_numpy.transpose(1, 2, 0)
        denormalized_image = (image_numpy * std) + mean
        denormalized_image = np.clip(denormalized_image, 0, 1)
        image_to_show = (denormalized_image * 255).astype(np.uint8)
        if label.item() == 0:
            logging.info(f'{label.item()} : wearing glasses')
        print(label)

        cv2.imshow('image', image_to_show)
        cv2.waitKey(0)
if __name__ == '__main__':
    main()