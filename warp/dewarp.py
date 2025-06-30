from PIL import Image
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import argparse
import warnings

from warp.model import DocScanner
from warp.seg import U2NETP
warnings.filterwarnings("ignore")

def setup_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def get_device(gpu_id: int = 0) -> torch.device:
    """
    Select device based on gpu_id: non-negative uses that CUDA device if available;
    negative or None selects CPU.
    """
    if gpu_id is None or gpu_id < 0:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    logging.warning('CUDA not available, falling back to CPU')
    return torch.device('cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mask = U2NETP(3, 1)
        self.backbone = DocScanner()

    def forward(self, x, threshold: float = 0.5):
        mask, _1, _2, _3, _4, _5, _6 = self.mask(x)
        mask = (mask > threshold).float()
        x = mask * x

        bm = self.backbone(x, iters=12, test_mode=True)
        bm = (2 * (bm / 286.8) - 1) * 0.99

        return bm

def reload_seg(model: nn.Module,
               path: str = None,
               device: str = 'cuda:0') -> nn.Module:
    device = torch.device(device) if torch.cuda.is_available() else 'cpu'
    logging.info(f"Reloading segmentation model from {path} on device {device}")
    if path is None:
        logging.warning("No path provided for segmentation model, using default weights")
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info("Segmentation model reloaded successfully")

    return model.to(device).eval()

def reload_rec(model: nn.Module,
               path: str = None,
               device: str = 'cuda:0') -> nn.Module:
    if path is None:
        logging.warning("No path provided for recognition model, using default weights")
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location=device)
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model.to(device).eval()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document Dewarper Model")
    parser.add_argument('--seg_model', type=str, default='checkpoint/seg.pth',
                        help='Path to the segmentation model checkpoint')
    parser.add_argument('--rec_model', type=str, default='checkpoint/DocScanner-L.pth',
                        help='Path to the recognition model checkpoint')
    parser.add_argument('--distorted', type=str, default='output/distorted/',
                        help='Directory containing distorted images')
    parser.add_argument('--rectified', type=str, default='output/rectified/',
                        help='Directory to save rectified images')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use for processing (default: 0)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level (default: INFO)')
    return parser.parse_args()

def main():
    opt = parse_args()
    setup_logger(opt.log_level)
    device = get_device(opt.gpu_id)

    img_list = os.listdir(opt.distorted)
    if not os.path.exists(opt.rectified):
        os.makedirs(opt.rectified)

    model = Net().to(device)
    model.mask = reload_seg(model.mask, opt.seg_model, device)
    model.backbone = reload_rec(model.backbone, opt.rec_model, device)
    model.eval()

    logging.info(f"Processing {len(img_list)} images from {opt.distorted}")
    for img_name in tqdm(img_list):
        name = os.path.splitext(img_name)[0]
        img_path = os.path.join(opt.distorted, img_name)
        logging.info(f"Processing image: {img_path}")

        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.0
        h, w, _ = im_ori.shape
        img = cv2.resize(im_ori, (288, 288)).transpose(2, 0, 1)
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)

        with torch.no_grad():
            bm = model(img, threshold=0.5)
            bm = bm.cpu()

            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))

            bm0, bm1 = cv2.blur(bm0, (3, 3)), cv2.blur(bm1, (3, 3))
            label = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).to(device)
            out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float().to(device),
                                label, mode='bilinear', align_corners=True)
            
            cv2.imwrite(os.path.join(opt.rectified, f"{name}_rectified.png"),
                        (((out[0] * 255.0).permute(1, 2, 0).cpu().numpy())[:,:,::-1]).astype(np.uint8))
    logging.info(f"Rectified images saved to {opt.rectified}")

if __name__ == "__main__":
    main()