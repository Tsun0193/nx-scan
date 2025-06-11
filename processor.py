import argparse
import logging
import cv2
import torch
from pathlib import Path
from tqdm.auto import tqdm

from doclayout_yolo import YOLOv10

# Default constants
MODEL_CHECKPOINT = "checkpoint/doclayout_yolo_docstructbench_imgsz1024.pt"
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
        return torch.device('cuda')
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


def process_image(
    model: YOLOv10,
    image_path: Path,
    output_dir: Path,
    keep_classes: list[int]
) -> None:
    """
    Detect layout, crop out non-abandon bounding region,
    split into left/right pages, and save results.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    # Convert to RGB for model prediction
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image_rgb, classes=keep_classes)
    if not results:
        logging.warning(f"No objects detected in {image_path}")
        return

    # Compute bounding box extremes
    boxes = results[0].boxes.xyxy.data.cpu()
    x1, y1 = int(boxes[:,0].min()), int(boxes[:,1].min())
    x2, y2 = int(boxes[:,2].max()), int(boxes[:,3].max())

    # Crop and split pages
    cropped = image_rgb[y1:y2, x1:x2]
    mid = cropped.shape[1] // 2
    pages = {'left': cropped[:, :mid], 'right': cropped[:, mid:]}

    # Prepare output subdirectory per image
    subdir = output_dir / image_path.stem
    subdir.mkdir(parents=True, exist_ok=True)

    # Save pages
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
    names = model.names
    # Identify and filter out "abandon" class
    abandon_idx = next(idx for idx, name in names.items() if name == 'abandon')
    keep_ids = [idx for idx in names.keys() if idx != abandon_idx]

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find images recursively
    exts = ['png', 'jpg', 'jpeg']
    paths = find_image_paths(args.data_dir, exts)
    if not paths:
        logging.warning(f"No images found in {args.data_dir} with extensions {exts}")
        return

    logging.info(f"Found {len(paths)} image(s) to process.")
    with tqdm(paths, desc="Processing images", unit='image') as pbar:
        for img_path in pbar:
            process_image(model, img_path, args.output_dir, keep_ids)


if __name__ == '__main__':
    main()