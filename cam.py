import argparse
import os
import pathlib
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from backbone.model import EfficientNetV2_S, EfficientNetV2_L


def get_model(model_name: str):
    if model_name == 'efficientnetv2_s':
        return EfficientNetV2_S()
    elif model_name == 'efficientnetv2_l':
        return EfficientNetV2_L()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def find_module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """Traverse modules by dotted path, e.g., 'model.conv_head'"""
    mod = root
    for part in path.split('.'):
        if not hasattr(mod, part):
            raise AttributeError(f"Module path '{path}' invalid. Missing '{part}'.")
        mod = getattr(mod, part)
    return mod


def overlay_heatmap(img_bgr: np.ndarray, cam_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    # The cam_map from the library is already normalized between 0-1
    cam_resized = cv2.resize((cam_map * 255).astype(np.uint8), (w, h))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Note: missing keys: {missing}, unexpected keys: {unexpected}")


def build_val_transform(img_size: int):
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(img_size, img_size)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def gather_paths_from_dirs(wear_dir: Optional[str], nowear_dir: Optional[str], limit: Optional[int]) -> List[pathlib.Path]:
    paths: List[pathlib.Path] = []
    if wear_dir:
        wd = pathlib.Path(wear_dir)
        if wd.exists():
            wear_paths = list(wd.glob('**/*.jpg'))
            paths.extend(wear_paths[:limit] if limit else wear_paths)
    if nowear_dir:
        nd = pathlib.Path(nowear_dir)
        if nd.exists():
            n_paths = list(nd.glob('**/*.jpg'))
            paths.extend(n_paths[:limit] if limit else n_paths)
    return paths


def run_cam_on_image(
    model: torch.nn.Module,
    device: torch.device,
    img_path: pathlib.Path,
    transform,
    target_module: torch.nn.Module,
    out_dir: pathlib.Path,
    method: str = 'both',
    img_size: int = 640,
):
    orig_bgr = cv2.imread(str(img_path))
    if orig_bgr is None:
        print(f"Skip unreadable image: {img_path}")
        return
    orig_bgr = cv2.resize(orig_bgr, (img_size, img_size))

    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)

    model.eval()
    # Get model prediction
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    targets = [ClassifierOutputTarget(pred)]

    if method in ('gradcampp', 'both'):
        cam_gcpp = GradCAMPlusPlus(model=model, target_layers=[target_module])
        gcpp_cam_map = cam_gcpp(input_tensor=tensor, targets=targets)[0, :]
        gcpp_overlay = overlay_heatmap(orig_bgr, gcpp_cam_map)
        cv2.namedWindow('Grad-CAM++', cv2.WINDOW_NORMAL)
        cv2.imshow("Grad-CAM++", gcpp_overlay)

    if method in ('layercam', 'both'):
        cam_layer = LayerCAM(model=model, target_layers=[target_module])
        layer_cam_map = cam_layer(input_tensor=tensor, targets=targets)[0, :]
        layer_overlay = overlay_heatmap(orig_bgr, layer_cam_map)
        cv2.namedWindow('Layer-CAM', cv2.WINDOW_NORMAL)
        cv2.imshow("Layer-CAM", layer_overlay)

    if method != 'none':
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(description='CAM visualization using pytorch-grad-cam')
    # Model
    parser.add_argument('--model', type=str, default='efficientnetv2_s', choices=['efficientnetv2_s', 'efficientnetv2_l'])
    parser.add_argument('--weight_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--target_layer', type=str, default='model.conv_head', help="Dotted path to target layer inside the wrapper model")
    # Data
    parser.add_argument('--image', type=str, default=None, help='Single image path')
    parser.add_argument('--wear', type=str, default=None, help='Directory of wear images (.jpg)')
    parser.add_argument('--nowear', type=str, default=None, help='Directory of no-wear images (.jpg)')
    parser.add_argument('--limit', type=int, default=50, help='Max images per directory when using dirs')
    # CAM
    parser.add_argument('--method', type=str, default='both', choices=['gradcampp', 'layercam', 'both'])
    parser.add_argument('--output_dir', type=str, default='cam_outputs')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model).to(device)
    load_checkpoint(model, args.weight_path, device)
    model.eval()

    args.wear = '/media/ubuntu/새 볼륨/dataset/034.마스크 착용 한국인 안면 이미지 데이터/01.데이터/1.Training/원천데이터/yolo_face_detection_result_1'
    args.nowear = '/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/neckslice/refining_yaw_yaw'

    # Resolve target layer
    try:
        target_module = find_module_by_path(model, args.target_layer)
    except AttributeError as e:
        print(str(e))
        print("Tip: for EfficientNetV2_S wrapper, try 'model.conv_head' or a specific block like 'model.blocks.5.14'\n")
        return

    transform = build_val_transform(args.img_size)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths: List[pathlib.Path] = []
    if args.image:
        img_paths = [pathlib.Path(args.image)]
    else:
        img_paths = gather_paths_from_dirs(args.wear, args.nowear, args.limit)
        if not img_paths:
            print('No images found. Provide --image or valid --wear_dir/--nowear_dir')
            return

    for p in img_paths:
        run_cam_on_image(model, device, p, transform, target_module, out_dir, method=args.method, img_size=args.img_size)


if __name__ == '__main__':
    main()
