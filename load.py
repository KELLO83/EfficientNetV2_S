import torch
from backbone.model import EfficientNetV2_S
import pathlib
import natsort
import torchvision.transforms.v2 as v2
import cv2
import os
from tqdm import tqdm
weight_path = 'best_model.pth'


model =  EfficientNetV2_S()
result = model.load_state_dict(torch.load(weight_path))
import torchinfo

torchinfo.summary(
    model=model,
    input_size=(1, 3, 320, 320),
    verbose=True,
    col_names=["input_size", "output_size", "trainable"],
    row_settings=["depth"],
    mode='eval'
)
print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)


file_list = pathlib.Path('/home/ubuntu/KOR_DATA/sunglass_dataset/nowear/no_wear_data_aligend').glob('*.jpg')
file_list = natsort.natsorted(list(file_list))

tp = 0
fp = 0
print(f"Total images to test: {len(file_list)}")
for file in tqdm(file_list):
    image = cv2.imread(str(file))
    basename = os.path.basename(file)
    tensor_image = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32 ,scale=True),
            v2.Resize(size=(320, 320)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(image)
    
    tensor_image = tensor_image.unsqueeze(0).to('cuda:0')
    model = model.to('cuda:0')
    model.eval()
    with torch.no_grad():
        output = model(tensor_image)
        predicted = torch.argmax(output, dim=1)

        if predicted.item() == 1:
            print(f"False Positive detected for file: {basename}")
            fp += 1
            # cv2.namedWindow("False Positive", cv2.WINDOW_NORMAL)
            # cv2.imshow("False Positive", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        else:
            tp += 1

print(f"Total False Positives: {fp} out of {len(file_list)}")
