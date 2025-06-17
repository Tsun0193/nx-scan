import argparse
import logging
import cv2
import numpy as np
import torch
import shutil
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


def process_image(
    model: YOLO,
    image_path: Path,
    output_dir: Path = None,
    crop_frac: float = 0.05,
    gutter_frac: float = 0.005,
    blur_ksize: int = 5
) -> json:
    """
    Detect layout, crop the highest-confidence box (no padding),
    split into left/right pages, and save results.
    """
    subdir = output_dir / image_path.stem
    subdir.mkdir(parents=True, exist_ok=True)

    # Predict boxes
    results = model.predict(str(image_path), verbose=False)
    if not results or len(results[0].boxes) == 0:
        raise RuntimeError(f"No objects detected in {image_path}")

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    max_idx = confs.argmax()
    x1, y1, x2, y2 = map(int, boxes[max_idx])
    score = float(confs[max_idx])

    # Load and crop
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cropped = img_rgb[y1:y2, x1:x2]

    # Split pages
    mid = divider(cropped, crop_frac=crop_frac,
                  gutter_frac=gutter_frac,
                  blur_ksize=blur_ksize)
    left_img = cropped[:, :mid]
    right_img = cropped[:, mid:]

    # Save images
    left_path = subdir / "left.jpg"
    right_path = subdir / "right.jpg"
    cv2.imwrite(str(left_path), cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(right_path), cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))

    obj = {
        "left_image": str(left_path),
        "right_image": str(right_path),
        "confidence": score
    }
    # logging.info(f"Processed {image_path}: {obj}")
    return json.dumps(obj, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Document layout cropper using YOLO object detection."
    )
    parser.add_argument(
        'image_path',
        type=Path,
        help='Path to the input image'
    )
    parser.add_argument(
        'output_dir',
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
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(getattr(logging, args.log_level))

    device = get_device(args.gpu_id)
    logging.info(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = process_image(
            model,
            args.image_path,
            args.output_dir
        )
        # Print JSON to stdout
        print(json.dumps(result))
    except Exception as e:
        logging.error(e)
        exit(1)


if __name__ == '__main__':
    main()