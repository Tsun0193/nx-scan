import os
import argparse
import logging
import cv2
import numpy as np
import torch
import json
from pathlib import Path

from ultralytics import YOLO

# Default constants
MODEL_CHECKPOINT = "checkpoint/nx_yolo.pt"


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


def find_image_paths(data_dir: Path, exts: list[str]) -> list[Path]:
    """
    Recursively find image files with given extensions (case-insensitive).
    """
    image_paths: list[Path] = []
    for ext in exts:
        image_paths.extend(data_dir.rglob(f"*.{ext}"))
        image_paths.extend(data_dir.rglob(f"*.{ext.upper()}"))
    return image_paths


def load_model(checkpoint: Path, device: torch.device) -> YOLO:
    """
    Load the YOLO model from a checkpoint, on specified device.
    """
    model = YOLO(model=str(checkpoint)).to(device)
    return model


def divider(
    img: np.ndarray,
    crop_frac: float = 0.05,    # how far left/right of image‐center to search
    gutter_frac: float = 0.005,  # gutter width as fraction of image width
    blur_ksize: int = 5        # optional smoothing to reduce speckle
) -> int:
    """
    Detect the gutter by finding the darkest vertical strip
    in a central band of the page.
    """
    # 1) Grayscale & smooth
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    h, w = gray.shape
    half = w // 2
    span = int(w * crop_frac / 2)
    left, right = half - span, half + span

    # 2) Auto‐threshold (Otsu) to separate dark pixels
    _, dark_mask = cv2.threshold(
        gray, 0, 1,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    # dark_mask is 1 for gutter (dark), 0 for page (light)

    # 3) Per-column dark ratio (0–1)
    col_dark_ratio = dark_mask.sum(axis=0) / float(h)
    region = col_dark_ratio[left:right]

    # 4) Sliding window of gutter width
    win = max(1, int(w * gutter_frac))
    win = min(win, region.shape[0])
    kernel = np.ones(win, dtype=float) / win
    avg_dark = np.convolve(region, kernel, mode='valid')

    # 5) Pick the window with the highest dark ratio
    idx = int(np.argmax(avg_dark))
    split_x = left + idx + win // 2
    return split_x


def detect_layout(
    image_path: Path,
    result,
    output_dir: Path,
    crop_frac: float,
    gutter_frac: float,
    blur_ksize: int
):
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    if boxes.shape[0] == 0:
        logging.warning(f"No objects detected in {image_path.name}, skipping.")
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

    mid = divider(cropped, crop_frac=crop_frac,
                  gutter_frac=gutter_frac,
                  blur_ksize=blur_ksize)
    left_img = cropped[:, :mid]
    right_img = cropped[:, mid:]

    left_path = output_dir / f"{image_path.stem}_1L.jpg"
    right_path = output_dir / f"{image_path.stem}_2R.jpg"
    cv2.imwrite(str(left_path), cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(str(right_path), cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    results_json = output_dir / 'results.json'
    if not results_json.exists():
        with open(results_json, 'w') as f:
            json.dump([], f, indent=2)
    with open(results_json, 'r+') as f:
        data = json.load(f)
        data = [item for item in data if item['original_image'] != str(image_path)]
        f.seek(0)
        f.truncate()
        data.append({
            "original_image": str(image_path),
            "left_image": str(left_path),
            "right_image": str(right_path),
            "confidence": score
        })
        json.dump(data, f, indent=2)

    return {
        "original_image": str(image_path),
        "left_image": str(left_path),
        "right_image": str(right_path),
        "confidence": score
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Document layout cropper using YOLO object detection."
    )
    parser.add_argument(
        '--data-dir', '-i',
        type=Path,
        required=True,
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=Path,
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=Path,
        default=Path(MODEL_CHECKPOINT),
        help='YOLO model checkpoint'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU device ID (use -1 for CPU)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=1,
        help='Batch size for processing images (default: 1)'
    )
    parser.add_argument(
        '--crop_frac',
        type=float,
        default=0.05,
        help='Fraction of image width to search for gutter (default: 0.05)'
    )
    parser.add_argument(
        '--gutter_frac',
        type=float,
        default=0.005,
        help='Gutter width as fraction of image width (default: 0.005)'
    )
    parser.add_argument(
        '--blur_ksize',
        type=int,
        default=5,
        help='Kernel size for Gaussian blur (default: 5)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(getattr(logging, args.log_level))

    device = get_device(args.gpu_id)
    logging.info(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = args.output_dir / 'results.json' 
    if results.exists():
        with open(results, 'r') as f:
            try:
                existeds = json.load(f)
                proceeds = {
                    item['original_image'] for item in existeds
                }
            except json.JSONDecodeError:
                logging.warning("Failed to decode JSON from results file.")

    exts = ['png', 'jpg', 'jpeg']
    paths = find_image_paths(args.data_dir, exts)

    unprocessed = [p for p in paths if str(p) not in proceeds]
    skipped = len(paths) - len(unprocessed)
    if skipped > 0:
        logging.info(f"Skipping {skipped} already processed images.")
    paths = unprocessed

    if not paths:
        logging.error(f"No unprocessed images found in {args.data_dir} with extensions {exts}")
        return
    
    logging.info(f"Found {len(paths)} images to process")

    if args.batch == 1:
        for img_path in paths:
            try:
                result = model.predict(str(img_path), verbose=False)[0]
                out = detect_layout(
                    img_path, result, args.output_dir,
                    args.crop_frac, args.gutter_frac, args.blur_ksize
                )
                logging.info(f"Processed {img_path}: {out}")
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
    else:
        # batch inference
        sources = [str(p) for p in paths]
        try:
            batch_results = model.predict(
                sources,
                batch=args.batch,
                verbose=False
            )
            for img_path, result in zip(paths, batch_results):
                out = detect_layout(
                    img_path, result, args.output_dir,
                    args.crop_frac, args.gutter_frac, args.blur_ksize
                )
                logging.info(f"Processed {img_path}: {out}")
        except Exception as e:
            logging.error(f"Batch inference failed: {e}")


if __name__ == '__main__':
    main()