import os
import argparse
import logging
import cv2
import numpy as np
import json
from pathlib import Path

from ultralytics import YOLO
from typing import Optional, Dict, List

MODEL_CHECKPOINT = "checkpoint/nx_yolo.pt"  # assume it's placed alongside the script


def setup_logger(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def find_image_paths(data_dir: Path, exts: List[str]) -> List[Path]:
    image_paths: List[Path] = []
    for ext in exts:
        image_paths.extend(data_dir.rglob(f"*.{ext}"))
        image_paths.extend(data_dir.rglob(f"*.{ext.upper()}"))
    return image_paths


def load_model(checkpoint: Path) -> YOLO:
    model = YOLO(model=str(checkpoint))
    return model


def divider(
    img: np.ndarray,
    crop_frac: float = 0.05,
    gutter_frac: float = 0.005,
    blur_ksize: int = 5
) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    h, w = gray.shape
    half = w // 2
    span = int(w * crop_frac / 2)
    left, right = half - span, half + span

    _, dark_mask = cv2.threshold(
        gray, 0, 1,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    col_dark_ratio = dark_mask.sum(axis=0) / float(h)
    region = col_dark_ratio[left:right]

    win = max(1, int(w * gutter_frac))
    win = min(win, region.shape[0])
    kernel = np.ones(win, dtype=float) / win
    avg_dark = np.convolve(region, kernel, mode='valid')

    idx = int(np.argmax(avg_dark))
    return left + idx + win // 2


def detect_layout(
    image_path: Path,
    result,
    output_dir: Path,
    crop_frac: float,
    gutter_frac: float,
    blur_ksize: int
) -> Optional[Dict]:
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

    mid = divider(cropped, crop_frac=crop_frac, gutter_frac=gutter_frac, blur_ksize=blur_ksize)
    left_img = cropped[:, :mid]
    right_img = cropped[:, mid:]

    left_path = output_dir / f"{image_path.stem}_1L.jpg"
    right_path = output_dir / f"{image_path.stem}_2R.jpg"
    cv2.imwrite(str(left_path), cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(right_path), cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))

    results_json = output_dir / 'results.json'
    if not results_json.exists():
        with open(results_json, 'w') as f:
            json.dump([], f, indent=2)

    with open(results_json, 'r+', encoding='utf-8') as f:
        data = json.load(f)
        # remove any previous entry for this image
        data = [d for d in data if d['original_image'] != str(image_path)]
        data.append({
            'original_image': str(image_path),
            'left_image': str(left_path),
            'right_image': str(right_path),
            'confidence': score
        })
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2)

    return {
        'original_image': str(image_path),
        'left_image': str(left_path),
        'right_image': str(right_path),
        'confidence': score
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layout cropper (CPU version)")
    parser.add_argument(
        '--data-dir', '-i', type=Path, required=True,
        help="Folder of images"
    )
    parser.add_argument(
        'output_dir', type=Path,
        help="Where to save cropped output"
    )
    parser.add_argument(
        '--checkpoint', '-c', type=Path,
        default=Path(MODEL_CHECKPOINT),
        help="YOLO .pt file path"
    )
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--batch-size', '-b', type=int, default=1,
        help='Batch size for inference (1 = per-image)'
    )
    parser.add_argument(
        '--crop-frac', type=float, default=0.05,
        help='Central search width fraction'
    )
    parser.add_argument(
        '--gutter-frac', type=float, default=0.005,
        help='Gutter width fraction'
    )
    parser.add_argument(
        '--blur-ksize', type=int, default=5,
        help='Gaussian blur kernel size'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(getattr(logging, args.log_level.upper(), logging.INFO))

    model = load_model(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results to skip re-processing
    processed = set()
    results_json = args.output_dir / 'results.json'
    if results_json.exists():
        with open(results_json, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                processed = {item['original_image'] for item in data}
            except json.JSONDecodeError:
                logging.warning("Corrupt results.json; reprocessing all images")

    exts = ['jpg', 'jpeg', 'png']
    all_paths = find_image_paths(args.data_dir, exts)
    to_process = [p for p in all_paths if str(p) not in processed]
    skipped = len(all_paths) - len(to_process)
    if skipped:
        logging.info(f"Skipping {skipped} already-processed images")

    if not to_process:
        logging.info("No new images to process.")
        return

    logging.info(f"Processing {len(to_process)} images with batch size {args.batch_size}")

    # Inference and layout
    if args.batch_size == 1:
        for img_path in to_process:
            try:
                result = model.predict(str(img_path), verbose=False, device="cpu")[0]
                out = detect_layout(
                    img_path, result, args.output_dir,
                    args.crop_frac, args.gutter_frac, args.blur_ksize
                )
                if out:
                    logging.info(f"Processed: {out['original_image']}")
            except Exception as e:
                logging.error(f"Failed processing {img_path.name}: {e}")
    else:
        sources = [str(p) for p in to_process]
        try:
            batch_results = model.predict(
                sources,
                batch=args.batch_size,
                verbose=False,
                device="cpu"
            )
            for img_path, res in zip(to_process, batch_results):
                out = detect_layout(
                    img_path, res, args.output_dir,
                    args.crop_frac, args.gutter_frac, args.blur_ksize
                )
                if out:
                    logging.info(f"Processed: {out['original_image']}")
        except Exception as e:
            logging.error(f"Batch inference failed: {e}")


if __name__ == "__main__":
    main()