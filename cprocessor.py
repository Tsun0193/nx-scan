import os
import argparse
import logging
import cv2
import numpy as np
import json
from pathlib import Path

from ultralytics import YOLO
from typing import Optional, Dict

MODEL_CHECKPOINT = "checkpoint/nx_yolo.pt"  # assume it's placed in same folder as .exe


def setup_logger(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def find_image_paths(data_dir: Path, exts: list[str]) -> list[Path]:
    image_paths = []
    for ext in exts:
        image_paths.extend(data_dir.rglob(f"*.{ext}"))
        image_paths.extend(data_dir.rglob(f"*.{ext.upper()}"))
    return image_paths


def load_model(checkpoint: Path) -> YOLO:
    model = YOLO(model=str(checkpoint))
    return model


def divider(img: np.ndarray, crop_frac=0.05, gutter_frac=0.005, blur_ksize=5) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    h, w = gray.shape
    half = w // 2
    span = int(w * crop_frac / 2)
    left, right = half - span, half + span

    _, dark_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    col_dark_ratio = dark_mask.sum(axis=0) / float(h)
    region = col_dark_ratio[left:right]

    win = max(1, int(w * gutter_frac))
    win = min(win, region.shape[0])
    kernel = np.ones(win, dtype=float) / win
    avg_dark = np.convolve(region, kernel, mode='valid')

    idx = int(np.argmax(avg_dark))
    return left + idx + win // 2


def process_image(model: YOLO, image_path: Path, output_dir: Path) -> Optional[Dict]:
    results = model.predict(str(image_path), verbose=False, device="cpu")
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    if boxes.shape[0] == 0:
        logging.warning(f"No objects detected in {image_path.name}")
        return None

    max_idx = confs.argmax()
    x1, y1, x2, y2 = map(int, boxes[max_idx])
    score = float(confs[max_idx])

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logging.warning(f"Failed to read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cropped = img_rgb[y1:y2, x1:x2]

    mid = divider(cropped)
    left_img = cropped[:, :mid]
    right_img = cropped[:, mid:]

    left_path = output_dir / f"{image_path.stem}_1L.jpg"
    right_path = output_dir / f"{image_path.stem}_2R.jpg"
    cv2.imwrite(str(left_path), cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(right_path), cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))

    result_json = output_dir / "results.json"
    if not result_json.exists():
        with open(result_json, "w") as f:
            json.dump([], f, indent=2)

    with open(result_json, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data = [d for d in data if d["original_image"] != str(image_path)]
        data.append({
            "original_image": str(image_path),
            "left_image": str(left_path),
            "right_image": str(right_path),
            "confidence": score
        })
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2)

    return data[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layout cropper (CPU version)")
    parser.add_argument('--data-dir', '-i', type=Path, required=True, help="Folder of images")
    parser.add_argument('output_dir', type=Path, help="Where to save cropped output")
    parser.add_argument('--checkpoint', '-c', type=Path, default=Path(MODEL_CHECKPOINT), help="YOLO .pt file path")
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(getattr(logging, args.log_level.upper()))

    model = load_model(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_image_paths(args.data_dir, ['jpg', 'jpeg', 'png'])
    if not image_paths:
        logging.error("No images found.")
        return

    for path in image_paths:
        try:
            result = process_image(model, path, args.output_dir)
            if result:
                logging.info(f"Processed: {result['original_image']}")
        except Exception as e:
            logging.error(f"Failed to process {path.name}: {e}")


if __name__ == "__main__":
    main()