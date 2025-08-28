import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from ultralytics import YOLO

print("PyTorch sees", torch.cuda.device_count(), "GPU; cuda:0 =", torch.cuda.get_device_name(0))

model = YOLO("yolov10x.pt")
model.train(
    task="detect",
    data="PDCu/data.yaml",
    epochs=512,
    imgsz=1280,
    patience=64,
    device=0,         # 0 == visible GPU index (physical GPU 2)
    batch=4,
    workers=16,
    mosaic=0.0, mixup=0.0, degrees=0, translate=0.0, shear=0.0, scale=0.5,
    iou=0.8, rect=True,
    seed=42, box=64,
    cache=False,
    project="runs",
)
