import os
import argparse
import logging
import cv2
import numpy as np
import torch
import json
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from ultralytics import YOLO

# Default constants
MODEL_CHECKPOINT   = "checkpoint/nx_yolo.pt"
DIVIDER_CHECKPOINT = "checkpoint/nx_divider.pt"


def setup_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def get_device(gpu_id: int = 0) -> torch.device:
    if gpu_id is None or gpu_id < 0:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    logging.warning('CUDA not available, falling back to CPU')
    return torch.device('cpu')


def find_image_paths(data_dir: Path, exts: List[str]) -> List[Path]:
    image_paths: List[Path] = []
    for ext in exts:
        image_paths.extend(data_dir.rglob(f"*.{ext}"))
        image_paths.extend(data_dir.rglob(f"*.{ext.upper()}"))
    return image_paths


def load_models(det_checkpoint: Path, divider_checkpoint: Path, device: torch.device) -> Tuple[YOLO, YOLO]:
    det_model = YOLO(model=str(det_checkpoint)).to(device)
    divider_model = YOLO(model=str(divider_checkpoint)).to(device)
    return det_model, divider_model


def _save_result_json(results_json: Path, record: dict, image_key: str = 'original_image'):
    results_json.parent.mkdir(parents=True, exist_ok=True)
    if not results_json.exists():
        results_json.write_text("[]")
    with open(results_json, 'r+') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
        data = [r for r in data if r.get(image_key) != record.get(image_key)]
        data.append(record)
        f.seek(0); f.truncate()
        json.dump(data, f, indent=2)


def _get_names_from_result(res) -> Dict[int, str]:
    names = getattr(res, "names", None)
    if names is None:
        try:
            names = res.model.names
        except Exception:
            names = {}
    # ensure keys are ints
    return {int(k): v for k, v in (names.items() if isinstance(names, dict) else {})}


def pick_center_box(result, img_w: int, img_h: int, center_frac: float = 0.25,
                    min_conf: float = 0.0, class_id: Optional[int] = None) -> Optional[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.data is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy().reshape(-1)
    cls  = boxes.cls.detach().cpu().numpy().astype(int).reshape(-1)

    idxs = np.arange(len(conf))
    if class_id is not None:
        idxs = idxs[cls == class_id]
    idxs = idxs[conf[idxs] >= min_conf]
    if idxs.size == 0:
        return None

    cx_img, cy_img = img_w / 2.0, img_h / 2.0
    centers_x = (xyxy[idxs, 0] + xyxy[idxs, 2]) / 2.0
    centers_y = (xyxy[idxs, 1] + xyxy[idxs, 3]) / 2.0

    in_center = (np.abs(centers_x - cx_img) <= center_frac * img_w) & \
                (np.abs(centers_y - cy_img) <= center_frac * img_h)
    cand = idxs[in_center]
    if cand.size == 0:
        d = np.hypot(centers_x - cx_img, centers_y - cy_img)
        order = np.lexsort((-conf[idxs], d))  # distance asc, conf desc
        i = idxs[order[0]]
    else:
        d = np.hypot(centers_x[in_center] - cx_img, centers_y[in_center] - cy_img)
        conf_c = conf[cand]
        order = np.lexsort((d, -conf_c))     # conf desc, distance asc
        i = cand[order[0]]

    x1, y1, x2, y2 = xyxy[i]
    names = _get_names_from_result(result)
    return {
        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
        "conf": float(conf[i]),
        "cls_id": int(cls[i]),
        "cls_name": names.get(int(cls[i])),
        "center_x": float((x1 + x2) / 2.0),
        "center_y": float((y1 + y2) / 2.0),
    }


def divider_with_yolo(
    img_rgb: np.ndarray,
    divider_model: YOLO,
    center_frac: float = 0.25,
    min_conf: float = 0.0,
    class_id: Optional[int] = None
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """Return (mid, pick_dict). mid is x-center of chosen divider box in CROPPED coords."""
    H, W = img_rgb.shape[:2]
    res = divider_model.predict(img_rgb, 
                                conf=0.001,
                                iou=0.9,
                                verbose=False)[0]
    pick = pick_center_box(res, img_w=W, img_h=H, center_frac=center_frac,
                           min_conf=min_conf, class_id=class_id)
    if pick is None:
        return None, None
    x1, _, x2, _ = pick["xyxy"]
    mid = int(round((x1 + x2) / 2.0))
    return mid, pick


def divider_heuristic(
    img_rgb: np.ndarray,
    crop_frac: float = 0.05,
    gutter_frac: float = 0.005,
    blur_ksize: int = 5,
    contrast_threshold: float = 1.2,
    refine_right_frac: float = 0.01
) -> Optional[Dict[str, int]]:
    """
    Old image-processing gutter detector + rightward refinement.
    Returns dict with keys: mid_initial, mid_refined, mid
    (all in CROPPED coords), or None if nothing strong is found.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    H, W = gray.shape
    cx = W // 2
    span = max(1, int(W * crop_frac * 0.5))
    left = max(0, cx - span)
    right = min(W, cx + span)
    if right <= left:
        return None

    _, dark_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    col_dark_ratio = dark_mask.sum(axis=0) / float(H)

    win = max(1, int(W * gutter_frac))
    win = min(win, right - left)
    if win <= 0:
        return None

    region = col_dark_ratio[left:right]
    kernel = np.ones(win, dtype=float) / win
    avg_dark = np.convolve(region, kernel, mode='valid')
    if avg_dark.size == 0:
        return None

    max_dark = float(np.max(avg_dark))
    mean_dark = float(np.mean(avg_dark))
    if max_dark < contrast_threshold * (mean_dark + 1e-8):
        return None

    idx = int(np.argmax(avg_dark))
    mid0 = left + idx + win // 2

    # Rightward refinement up to 1% width: pick lightest column
    max_step = max(1, int(W * refine_right_frac))
    r_end = min(W - 1, mid0 + max_step)
    seg = col_dark_ratio[mid0:r_end + 1]
    if seg.size > 0:
        off = int(np.argmin(seg))
        mid = mid0 + off
    else:
        mid = mid0

    return {"mid_initial": int(mid0), "mid_refined": int(mid), "mid": int(mid)}


def detect_layout(
    image_path: Path,
    result,
    output_dir: Path,
    divider_model: YOLO,
    center_frac: float,
    divider_min_conf: float,
    divider_class: Optional[int],
    crop_frac: float,
    gutter_frac: float,
    blur_ksize: int,
    contrast_threshold: float,
    refine_right_frac: float
):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json = output_dir / 'results.json'

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logging.warning(f"Failed to read image: {image_path}, copying as single page.")
        single_path = output_dir / f"{image_path.stem}_1S{image_path.suffix}"
        try:
            shutil.copy2(str(image_path), str(single_path))
        except Exception:
            pass
        record = {
            "original_image": str(image_path),
            "image_shape": None,
            "mode": "single_read_fail",
            "detector": None,
            "divider": None,
            "single_image": str(single_path),
        }
        _save_result_json(results_json, record)
        return record

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # Portrait → single page
    if H > W:
        single_path = output_dir / f"{image_path.stem}_1S{image_path.suffix}"
        cv2.imwrite(str(single_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        record = {
            "original_image": str(image_path),
            "image_shape": [int(H), int(W)],
            "mode": "single_portrait",
            "detector": {
                "xyxy": [0, 0, int(W), int(H)],
                "conf": None, "cls_id": None, "cls_name": None
            },
            "divider": None,
            "single_image": str(single_path),
        }
        _save_result_json(results_json, record)
        return record

    # Main detector → crop region
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    names = _get_names_from_result(result)

    if boxes.shape[0] == 0:
        # operate on full image
        det_xyxy = [0, 0, int(W), int(H)]
        det_conf = None
        det_cls  = None
        det_name = None
        cropped = img_rgb
        y1 = x1 = 0  # for global coord math later
    else:
        max_idx = int(confs.argmax())
        x1, y1, x2, y2 = map(int, boxes[max_idx])
        det_xyxy = [int(x1), int(y1), int(x2), int(y2)]
        det_conf = float(confs[max_idx])
        det_cls  = int(result.boxes.cls.cpu().numpy()[max_idx])
        det_name = names.get(det_cls)
        cropped = img_rgb[y1:y2, x1:x2]

    ch, cw = cropped.shape[:2]
    if ch > cw:
        single_path = output_dir / f"{image_path.stem}_1S{image_path.suffix}"
        cv2.imwrite(str(single_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        record = {
            "original_image": str(image_path),
            "image_shape": [int(H), int(W)],
            "crop_shape": [int(ch), int(cw)],
            "mode": "single_cropped_portrait",
            "detector": {
                "xyxy": det_xyxy, "conf": det_conf, "cls_id": det_cls, "cls_name": det_name
            },
            "divider": None,
            "single_image": str(single_path),
        }
        _save_result_json(results_json, record)
        return record

    # Try YOLO divider
    mid, pick = divider_with_yolo(
        cropped,
        divider_model=divider_model,
        center_frac=center_frac,
        min_conf=divider_min_conf,
        class_id=divider_class
    )

    divider_info: Dict[str, Any]
    if mid is not None and pick is not None:
        divider_info = {
            "method": "yolo",
            "xyxy": [float(v) for v in pick["xyxy"]],     # CROPPED coords
            "conf": float(pick["conf"]),
            "cls_id": int(pick["cls_id"]),
            "cls_name": pick.get("cls_name"),
            "center_x": float(pick["center_x"]),
            "center_y": float(pick["center_y"]),
            "mid": int(mid),                              # CROPPED coord
            "mid_global": int(x1 + mid)                   # ORIGINAL coord
        }
    else:
        # Heuristic fallback
        hres = divider_heuristic(
            cropped,
            crop_frac=crop_frac,
            gutter_frac=gutter_frac,
            blur_ksize=blur_ksize,
            contrast_threshold=contrast_threshold,
            refine_right_frac=refine_right_frac
        )
        if hres is not None:
            mid = int(hres["mid"])
            divider_info = {
                "method": "heuristic",
                "mid_initial": int(hres["mid_initial"]),
                "mid_refined": int(hres["mid_refined"]),
                "mid": int(mid),
                "mid_global": int(x1 + mid)
            }
        else:
            # Final fallback: midpoint
            mid = cw // 2
            divider_info = {
                "method": "midpoint",
                "mid": int(mid),
                "mid_global": int(x1 + mid)
            }

    # Save crops
    left_img  = cropped[:, :mid]
    right_img = cropped[:, mid:]
    left_path = output_dir / f"{image_path.stem}_1L{image_path.suffix}"
    right_path= output_dir / f"{image_path.stem}_2R{image_path.suffix}"

    cv2.imwrite(str(left_path),  cv2.cvtColor(left_img,  cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(str(right_path), cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    record = {
        "original_image": str(image_path),
        "image_shape": [int(H), int(W)],
        "crop_shape": [int(ch), int(cw)],
        "mode": "split",
        "detector": {
            "xyxy": det_xyxy, "conf": det_conf, "cls_id": det_cls, "cls_name": det_name
        },
        "divider": divider_info,
        "left_image": str(left_path),
        "right_image": str(right_path),
    }
    _save_result_json(results_json, record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Document layout cropper using YOLO object detection + divider model + heuristic fallback."
    )
    parser.add_argument('--data-dir', '-i', type=Path, required=True, help='Input directory containing images')
    parser.add_argument('--output_dir', '-o', type=Path, help='Directory to save outputs (default: <data-dir>/nx_out)')
    parser.add_argument('--checkpoint', '-c', type=Path, default=Path(MODEL_CHECKPOINT),
                        help='YOLO model checkpoint for main detector')
    parser.add_argument('--divider_checkpoint', '-dc', type=Path, default=Path(DIVIDER_CHECKPOINT),
                        help='YOLO model checkpoint for divider detection')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID (use -1 for CPU)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level')
    parser.add_argument('--batch', '-b', type=int, default=1, help='Batch size for processing images (default: 1)')

    # Divider YOLO tuning
    parser.add_argument('--center_frac', type=float, default=0.25,
                        help='Half-size of the center window as W/H fraction for divider picking (default: 0.25)')
    parser.add_argument('--divider_min_conf', type=float, default=0.0,
                        help='Minimum confidence for divider candidates (default: 0.0)')
    parser.add_argument('--divider_class', type=int, default=None,
                        help='Restrict divider to a specific class id (default: None = any class)')

    # Heuristic fallback tuning
    parser.add_argument('--crop_frac', type=float, default=0.05,
                        help='Fraction of width around center to search for gutter (default: 0.05)')
    parser.add_argument('--gutter_frac', type=float, default=0.005,
                        help='Gutter width as fraction of image width (default: 0.005)')
    parser.add_argument('--blur_ksize', type=int, default=5,
                        help='Kernel size for Gaussian blur (default: 5)')
    parser.add_argument('--contrast_threshold', type=float, default=1.2,
                        help='Require max-dark >= threshold * mean-dark to accept gutter (default: 1.2)')
    parser.add_argument('--refine_right_frac', type=float, default=0.01,
                        help='Max fraction of width to move RIGHT to the lightest column (default: 0.01)')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(getattr(logging, args.log_level))

    if args.output_dir is None:
        args.output_dir = args.data_dir / "nx_out"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.gpu_id)
    logging.info(f"Using device: {device}")

    det_model, divider_model = load_models(args.checkpoint, args.divider_checkpoint, device)

    results_json = args.output_dir / 'results.json'
    proceeds = set()
    if results_json.exists():
        try:
            with open(results_json, 'r') as f:
                existeds = json.load(f)
            proceeds = {item.get('original_image') for item in existeds if 'original_image' in item}
            logging.info(f"Found {len(proceeds)} entries in results.json, will skip them.")
        except json.JSONDecodeError:
            logging.warning("Failed to decode JSON from results file; proceeding without skip list.")

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
                result = det_model.predict(str(img_path), verbose=False)[0]
                out = detect_layout(
                    img_path, result, args.output_dir,
                    divider_model,
                    args.center_frac, args.divider_min_conf, args.divider_class,
                    args.crop_frac, args.gutter_frac, args.blur_ksize, args.contrast_threshold, args.refine_right_frac
                )
                logging.info(f"Processed {img_path}: {out}")
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
    else:
        sources = [str(p) for p in paths]
        try:
            batch_results = det_model.predict(sources, batch=args.batch, verbose=False)
            for img_path, result in zip(paths, batch_results):
                out = detect_layout(
                    img_path, result, args.output_dir,
                    divider_model,
                    args.center_frac, args.divider_min_conf, args.divider_class,  # typo fix below if needed
                    args.crop_frac, args.gutter_frac, args.blur_ksize, args.contrast_threshold, args.refine_right_frac
                )
                logging.info(f"Processed {img_path}: {out}")
        except Exception as e:
            logging.error(f"Batch inference failed: {e}")


if __name__ == '__main__':
    main()