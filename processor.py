import argparse
import logging
import cv2
import numpy as np
import torch
import shutil
from pathlib import Path
from tqdm.auto import tqdm

from doclayout_yolo import YOLOv10

# Default constants
MODEL_CHECKPOINT = "checkpoint/yolo11_x.pt"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "output"


def setup_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def get_device(use_cuda: bool = True) -> torch.device:
    """Select CPU or CUDA device."""
    if use_cuda and torch.cuda.is_available():
        return torch.device('cuda:2')
    return torch.device('cpu')


def load_model(checkpoint: Path, device: torch.device) -> YOLOv10:
    """
    Load the YOLOv10 model from a checkpoint, on specified device.
    """
    model = YOLOv10(model=str(checkpoint)).to(device)
    return model


def find_image_paths(data_dir: Path, exts: list[str]) -> list[Path]:
    """
    Recursively find image files with given extensions (case-insensitive).
    """
    image_paths: list[Path] = []
    for ext in exts:
        image_paths.extend(data_dir.rglob(f"*.{ext}"))
        image_paths.extend(data_dir.rglob(f"*.{ext.upper()}"))
    return image_paths


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
    # compute average over each window
    kernel = np.ones(win, dtype=float) / win
    avg_dark = np.convolve(region, kernel, mode='valid')

    # 5) Pick the window with the highest dark ratio
    idx = int(np.argmax(avg_dark))
    split_x = left + idx + win // 2
    return split_x


def process_image(
    model: YOLOv10,
    image_path: Path,
    output_dir: Path
) -> None:
    """
    Detect layout, crop out non-abandon bounding region,
    split into left/right pages, and save results.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image_rgb,
                            verbose=False)
    if not results:
        logging.warning(f"No objects detected in {image_path}")
        return

    # Compute bounding box extremes
    boxes = results[0].boxes.xyxy.data.cpu().numpy()
    max_area_box = boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
    x1, y1 = int(max_area_box[0]), int(max_area_box[1])
    x2, y2 = int(max_area_box[2]), int(max_area_box[3])

    cropped = image_rgb[y1:y2, x1:x2]
    mid = divider(cropped)
    pages = {'left': cropped[:, :mid], 'right': cropped[:, mid:]}

    # Prepare output subdirectory per image
    subdir = output_dir / image_path.stem
    subdir.mkdir(parents=True, exist_ok=True)

    original_file = subdir / "original.png"
    shutil.copy(image_path, original_file)

    for side, img in pages.items():
        out_file = subdir / f"{side}.png"
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_file), img_bgr)
        logging.info(f"Saved page: {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Document layout cropper using YOLOv10"
    )
    parser.add_argument(
        '--data-dir', '-i',
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help='Directory to save processed images'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=Path,
        default=Path(MODEL_CHECKPOINT),
        help='YOLOv10 model checkpoint path'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    level = getattr(logging, args.log_level)
    setup_logger(level)

    device = get_device(not args.no_cuda)
    logging.info(f"Using device: {device}")

    model = load_model(args.checkpoint, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exts = ['png', 'jpg', 'jpeg']
    paths = find_image_paths(args.data_dir, exts)
    if not paths:
        logging.warning(f"No images found in {args.data_dir} with extensions {exts}")
        return

    logging.info(f"Found {len(paths)} image(s) to process.")
    with tqdm(paths, desc="Processing images", unit='image') as pbar:
        for img_path in pbar:
            process_image(model, img_path, args.output_dir)


if __name__ == '__main__':
    main()