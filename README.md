# NxScan

A command-line tool to detect and crop a single document image using a custom YOLO model, split it into left/right pages, and output a JSON with paths and confidence score.

## Features

* Loads a custom-trained YOLO model checkpoint to detect the document region.
* Crops the highest-confidence bounding box.
* Splits the cropped image into left and right pages based on a dark gutter detection algorithm.
* Saves the resulting images as `*_1L.jpg` and `*_2R.jpg` in the specified output folder.
* Maintains a `results.json` file with entries for each processed image.
* Skips re-processing of images already listed in `results.json`.

## Prerequisites

* Python 3.8+
* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* OpenCV (`cv2`)
* NumPy
* PyTorch

Install dependencies via pip:

```bash
pip install ultralytics opencv-python numpy torch
```

Ensure you have a YOLO checkpoint trained for document detection (e.g. `checkpoint/nx_yolo.pt`), or pass your own model path using `--checkpoint`.

## Usage
<OUTPUT_DIR> [options]
```

### Positional Arguments
```bash
python processor.py -i <DATA_DIR> -o 

* `<DATA_DIR>`: Path to the folder containing input images (e.g., `data/`).
* `<OUTPUT_DIR>`: Directory where outputs and `results.json` will be saved.

### Options

| Flag                   | Type   | Default                 | Description                                      |
| ---------------------- | ------ | ----------------------- | ------------------------------------------------ |
| `-c`, `--checkpoint`   | Path   | `checkpoint/nx_yolo.pt` | Path to your trained YOLO model checkpoint.      |
| `--gpu-id`             | Int    | `0`                     | GPU device ID to use; set to `-1` for CPU only.  |
| `--log-level`          | String | `INFO`                  | Logging level (`DEBUG`, `INFO`, `WARNING`, etc). |
| `-b`, `--batch`        | Int    | `1`                     | Batch size for inference.                        |
| `--crop-frac`          | Float  | `0.05`                  | Fraction of image width to search for gutter.    |
| `--gutter-frac`        | Float  | `0.005`                 | Width of gutter as fraction of image width.      |
| `--blur-ksize`         | Int    | `5`                     | Kernel size for Gaussian blur.                   |

### Example

Process a folder of images with a trained model:

```bash
python processor.py \
  -i data/pages \
  output \
  --checkpoint checkpoint/nx_yolo.pt \
  --gpu-id 0 \
  --batch 4
```

Processed image outputs:

```
output/
├── image1_1L.jpg
├── image1_2R.jpg
├── image2_1L.jpg
├── image2_2R.jpg
└── results.json
```

Contents of `results.json`:

```json
[
  {
    "original_image": "data/pages/image1.jpg",
    "left_image": "output/image1_1L.jpg",
    "right_image": "output/image1_2R.jpg",
    "confidence": 0.9543
  }
]
```

## Logging

Use `--log-level DEBUG` for detailed output (e.g., bounding box info, split position).

## License

GNU General Public License v3.0 (GPL-3.0)