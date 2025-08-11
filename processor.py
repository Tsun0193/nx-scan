import os
import argparse
import logging
import cv2
import numpy as np
import torch
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

# ---------------------------- Defaults ----------------------------

MODEL_CHECKPOINT = "checkpoint/nx_yolo.pt"


# ---------------------------- Logging -----------------------------

def setup_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


# ---------------------------- Device ------------------------------

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


# ---------------------------- Files -------------------------------

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


# ----------------------- IoU / NMS helpers ------------------------

def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    IoU between one box [x1,y1,x2,y2] and many boxes shape (N,4).
    """
    xA = np.maximum(box[0], boxes[:, 0])
    yA = np.maximum(box[1], boxes[:, 1])
    xB = np.minimum(box[2], boxes[:, 2])
    yB = np.minimum(box[3], boxes[:, 3])

    interW = np.maximum(0, xB - xA)
    interH = np.maximum(0, yB - yA)
    inter = interW * interH

    boxArea = (box[2] - box[0]) * (box[3] - box[1])
    boxesArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = boxArea + boxesArea - inter + 1e-9
    return inter / union


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """
    Simple NMS: keep highest score, suppress any with IoU > iou_thresh w.r.t. a kept box.
    Returns indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]  # high→low
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou_one_to_many(boxes[i], boxes[rest])
        rest = rest[ious <= iou_thresh]
        order = rest

    return keep


# ------------------------ Decision helpers ------------------------

def _area(b):
    return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))


def _centers_x(boxes):
    return (boxes[:, 0] + boxes[:, 2]) / 2.0


def _choose_two_as_spread(boxes: np.ndarray, w: int, min_center_gap_frac: float) -> tuple[int, int] | None:
    """
    Pick leftmost and rightmost boxes if their centers are sufficiently far apart.
    Returns (left_idx, right_idx) indices into boxes, else None.
    """
    if len(boxes) < 2:
        return None
    cx = _centers_x(boxes)
    left_idx = int(np.argmin(cx))
    right_idx = int(np.argmax(cx))
    center_gap = abs(cx[right_idx] - cx[left_idx])
    if center_gap >= min_center_gap_frac * w:
        # Enforce left<right order
        if cx[left_idx] <= cx[right_idx]:
            return left_idx, right_idx
        else:
            return right_idx, left_idx
    return None


# ------------------------- Core routine ---------------------------

def detect_layout(
    image_path: Path,
    result,
    output_dir: Path,
    iou_thresh: float,
    min_area_frac: float,
    min_center_gap_frac: float
):
    """
    Binary decision:
      - SINGLE PAGE → write {stem}_1S.ext
      - DOUBLE PAGE → write {stem}_1L.ext and {stem}_2R.ext

    JSON schema kept backward compatible:
      Single: {"original_image","single_image","confidence","split":None}
      Double: {"original_image","left_image","right_image","confidence","split":None}
    """
    # 0) Prep output_dir & results.json
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / 'results.json'
    if not results_json.exists():
        results_json.write_text("[]")

    # 1) Read image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logging.warning(f"Failed to read image: {image_path}, copying as single page.")
        single_path = output_dir / f"{image_path.stem}_1S{image_path.suffix}"
        shutil.copy2(str(image_path), str(single_path))
        record = {
            "original_image": str(image_path),
            "single_image": str(single_path),
            "confidence": None,
            "split": None
        }
        with open(results_json, 'r+') as f:
            data = json.load(f)
            data = [r for r in data if r.get('original_image') != str(image_path)]
            data.append(record); f.seek(0); f.truncate(); json.dump(data, f, indent=2)
        return record

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    img_area = float(W * H)

    # 2) All detections
    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4), dtype=np.float32)
    confs = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.zeros((0,), dtype=np.float32)

    # 3) No detections → full image as single page
    if boxes.shape[0] == 0:
        single_path = output_dir / f"{image_path.stem}_1S{image_path.suffix}"
        cv2.imwrite(str(single_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        record = {
            "original_image": str(image_path),
            "single_image": str(single_path),
            "confidence": None,
            "split": None
        }
        with open(results_json, 'r+') as f:
            data = json.load(f)
            data = [r for r in data if r.get('original_image') != str(image_path)]
            data.append(record); f.seek(0); f.truncate(); json.dump(data, f, indent=2)
        return record

    # 4) NMS and area filtering
    keep = nms_indices(boxes, confs, iou_thresh=iou_thresh)
    boxes = boxes[keep]; confs = confs[keep]
    # filter tiny boxes
    if len(boxes) > 0:
        areas = np.array([_area(b) for b in boxes], dtype=np.float64)
        keep_area = areas >= (min_area_frac * img_area)
        boxes = boxes[keep_area]; confs = confs[keep_area]

    # 5) Decide: single vs double page
    two = _choose_two_as_spread(boxes, W, min_center_gap_frac=min_center_gap_frac)

    if two is None:
        # SINGLE PAGE
        # pick the best single region if any remain; else fallback to full image
        if len(boxes) > 0:
            idx = int(np.argmax(confs))
            x1, y1, x2, y2 = map(int, boxes[idx])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                crop = img_rgb[y1:y2, x1:x2]
            else:
                crop = img_rgb
            conf_val = float(confs[idx])
        else:
            crop = img_rgb
            conf_val = None

        out_path = output_dir / f"{image_path.stem}_1S{image_path.suffix}"
        cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        record = {
            "original_image": str(image_path),
            "single_image": str(out_path),
            "confidence": conf_val,
            "split": None
        }
        with open(results_json, 'r+') as f:
            data = json.load(f)
            data = [r for r in data if r.get('original_image') != str(image_path)]
            data.append(record); f.seek(0); f.truncate(); json.dump(data, f, indent=2)
        return record

    # DOUBLE PAGE
    li, ri = two
    cx = _centers_x(boxes)
    if cx[li] > cx[ri]:
        li, ri = ri, li

    (lx1, ly1, lx2, ly2) = map(int, boxes[li])
    (rx1, ry1, rx2, ry2) = map(int, boxes[ri])

    lx1, ly1 = max(0, lx1), max(0, ly1)
    lx2, ly2 = min(W, lx2), min(H, ly2)
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(W, rx2), min(H, ry2)

    left_img  = img_rgb[ly1:ly2, lx1:lx2] if (lx2 > lx1 and ly2 > ly1) else img_rgb
    right_img = img_rgb[ry1:ry2, rx1:rx2] if (rx2 > rx1 and ry2 > ry1) else img_rgb

    left_path  = output_dir / f"{image_path.stem}_1L{image_path.suffix}"
    right_path = output_dir / f"{image_path.stem}_2R{image_path.suffix}"

    cv2.imwrite(str(left_path),  cv2.cvtColor(left_img,  cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(str(right_path), cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    conf_pair = float(min(confs[li], confs[ri])) if (len(confs) > 1) else float(confs[li]) if len(confs) else None
    record = {
        "original_image": str(image_path),
        "left_image": str(left_path),
        "right_image": str(right_path),
        "confidence": conf_pair,   # conservative: min of the two
        "split": None              # no gutter split
    }
    with open(results_json, 'r+') as f:
        data = json.load(f)
        data = [r for r in data if r.get('original_image') != str(image_path)]
        data.append(record); f.seek(0); f.truncate(); json.dump(data, f, indent=2)
    return record


# --------------------------- CLI / Main ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect single vs double page and export crops."
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
        required=True,
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
        '--iou-thresh',
        type=float,
        default=0.36,
        help='IoU threshold for suppression (NMS). Detections with IoU > thresh are suppressed (keep higher conf).'
    )
    parser.add_argument(
        '--min-area-frac',
        type=float,
        default=0.10,
        help='Ignore detections smaller than this fraction of full image area (default: 0.10)'
    )
    parser.add_argument(
        '--min-center-gap-frac',
        type=float,
        default=0.20,
        help='Min horizontal center gap (fraction of width) to call it a double-page (default: 0.20)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(getattr(logging, args.log_level))

    device = get_device(args.gpu_id)
    logging.info(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load already-processed list (if present)
    results = args.output_dir / 'results.json'
    proceeds = set()
    if results.exists():
        try:
            with open(results, 'r') as f:
                existeds = json.load(f)
                proceeds = {item.get('original_image') for item in existeds if 'original_image' in item}
        except json.JSONDecodeError:
            logging.warning("Failed to decode JSON from results file; starting fresh.")

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
                    args.iou_thresh, args.min_area_frac, args.min_center_gap_frac
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
                    args.iou_thresh, args.min_area_frac, args.min_center_gap_frac
                )
                logging.info(f"Processed {img_path}: {out}")
        except Exception as e:
            logging.error(f"Batch inference failed: {e}")


if __name__ == '__main__':
    main()
