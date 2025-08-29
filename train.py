import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from ultralytics import YOLO

print("PyTorch sees", torch.cuda.device_count(), "GPU; cuda:0 =", torch.cuda.get_device_name(0))

model = YOLO("runs/train/weights/last.pt")
model.train(
    resume=True
)
