import os
import cv2
import torch
import shutil
from tqdm.auto import tqdm
from pathlib import Path
from ultralytics import YOLO

# ================= CONFIG =================
INPUT_DIR = Path("pdc")                  # source images
OUTPUT_DIR = Path("cropped")             # cropped results
MODEL_CHECKPOINT = "checkpoint/nx_yolo.pt"
RESIZE_W = 640   # width for inference (smaller = less VRAM, e.g. 640, 960, 1280)
# ==========================================

# Load YOLO model
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_CHECKPOINT)

# Ensure output folder exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect valid images
img_files = [p for p in INPUT_DIR.glob("*.*")
             if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]]

skipped = 0

with tqdm(total=len(img_files), desc="Processing images") as pbar:
    for img_path in img_files:
        img_bgr = cv2.imread(str(img_path))
        orig_h, orig_w = img_bgr.shape[:2]

        # Resize image manually for inference
        scale = RESIZE_W / orig_w
        new_w, new_h = RESIZE_W, int(orig_h * scale)
        img_resized = cv2.resize(img_bgr, (new_w, new_h))

        # Run YOLO on resized image only (no imgsz override, avoids stride warning)
        results = model.predict(
            source=img_resized,
            device=device,
            verbose=False,
            save=False
        )[0]

        out_name = f"{img_path.stem}_cropped{img_path.suffix}"
        out_path = OUTPUT_DIR / out_name

        if len(results.boxes) == 0:
            # No detections → copy original
            shutil.copy(img_path, out_path)
            skipped += 1
        else:
            # Best detection (highest confidence is index 0)
            best_box = results.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = best_box[:4]

            # Scale back coords to original resolution
            scale_x = orig_w / new_w
            scale_y = orig_h / new_h
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

            # Clamp to valid region
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            if x2 > x1 and y2 > y1:
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    cv2.imwrite(str(out_path), crop)
                else:
                    shutil.copy(img_path, out_path)
                    skipped += 1
            else:
                shutil.copy(img_path, out_path)
                skipped += 1

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({"no_pred": skipped})

print(f"\n🎉 Cropped images saved in: {OUTPUT_DIR}")
print(f"📊 Total skipped (0 preds or invalid): {skipped} / {len(img_files)}")