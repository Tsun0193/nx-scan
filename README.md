# Single Image Processor

A command-line tool to detect and crop a single document image using YOLO, split it into left/right pages, and output a JSON with paths and confidence score.

## Features

* Loads a pretrained YOLO model checkpoint to detect the document region.
* Crops the highest-confidence bounding box.
* Splits the cropped image into left and right pages based on a dark gutter detection algorithm.
* Saves the resulting images as `left.jpg` and `right.jpg` in a subdirectory under the specified output folder.
* Prints a JSON object to stdout containing:

  * `left_image`: path to the left page image
  * `right_image`: path to the right page image
  * `confidence`: detection confidence score

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

Ensure you have a YOLO checkpoint at `checkpoint/nx_yolo.pt`, or pass your own with `--checkpoint`.

## Usage

```bash
python single_image_processor.py <IMAGE_PATH> <OUTPUT_DIR> [options]
```

### Positional Arguments

* `<IMAGE_PATH>`: Path to the input image file (e.g., `data/my_doc.jpg`).
* `<OUTPUT_DIR>`: Directory where outputs will be saved.

### Options

| Flag                 | Type   | Default                 | Description                                      |
| -------------------- | ------ | ----------------------- | ------------------------------------------------ |
| `-c`, `--checkpoint` | Path   | `checkpoint/nx_yolo.pt` | Path to the YOLO model checkpoint.               |
| `--gpu-id`           | Int    | `0`                     | GPU device ID to use; set to `-1` for CPU only.  |
| `--log-level`        | String | `INFO`                  | Logging level (`DEBUG`, `INFO`, `WARNING`, etc). |

### Example

Process a single image on CPU:

```bash
python single_image_processor.py \
  data/sample.jpg \         # Path to your input image
  output \                  # Output directory
  --gpu-id 2                # Use GPU device 2
```

Output (stdout):

```json
{
  "left_image": "output/sample/left.jpg",
  "right_image": "output/sample/right.jpg",
  "confidence": 0.95
}
```

The images will be saved under:

```
output/
└── sample/
    ├── left.jpg
    └── right.jpg
```

## Logging

By default logs are printed at `INFO` level. Use `--log-level DEBUG` for more detailed output.

## License

GNU General Public License v3.0 (GPL-3.0)
