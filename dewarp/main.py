import os
import torch
import yaml
import gdown
import logging
import argparse
import contextlib
import io
from pytorch_lightning import LightningModule
from tqdm.auto import tqdm
from pathlib import Path
from geotr.inv3d_model.models.geotr.geotr_template import LitGeoTrTemplate


def setup_logger(level: int = logging.INFO) -> None:
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_from_checkpoint(model_class: type[LightningModule], 
                         checkpoint_file: Path) -> LightningModule:
    """
    Load a LightningModule model from checkpoint, suppressing terminal upgrade logs.
    """
    if not checkpoint_file.is_file():
        raise ValueError(f"Checkpoint not found at: {checkpoint_file.resolve()}")

    # Suppress stdout temporarily
    with contextlib.redirect_stderr(io.StringIO()):
        model = model_class.load_from_checkpoint(str(checkpoint_file))

    return model


def load_model(model_class: type[LightningModule],
               model_name: str = "geotr_template@inv3d",
               yaml_path: str = "models.yaml",
               device: str = "cuda") -> LightningModule:
    try:
        with open(yaml_path, "r") as file:
            models = yaml.safe_load(file)

        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in {yaml_path}.")

        url = models[model_name]
        logging.info(f"Loading model: {model_name} from {url}")

        if os.path.exists(f"checkpoint/{model_name.replace('/', '_')}.ckpt"):
            logging.info(f"Using cached checkpoint for {model_name}")
            ckpt_path = Path(f"checkpoint/{model_name.replace('/', '_')}.ckpt")
        else:
            logging.info(f"Downloading checkpoint for {model_name} from {url}")
            ckpt_path = Path(
                gdown.cached_download(
                    url=url,
                    path=f"checkpoint/{model_name.replace('/', '_')}.ckpt"
                )
            )

        model = load_from_checkpoint(model_class, ckpt_path)
        model.to(torch.device(device))
        model.eval()
        return model

    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Load and run GeoTrTemplate model.")
    parser.add_argument(
        "--data_dir", '-i',
        type=Path,
        required=True,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output_dir", '-o',
        type=Path,
        required=True,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="geotr_template@inv3d",
        help="Name of the model to load from models.yaml"
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="models.yaml",
        help="Path to the YAML file containing model configurations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (e.g., 'cuda' or 'cpu')"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level"
    )
    return parser.parse_args()

def main():
    """
    Main function to execute the dewarping process.
    """
    args = parse_args()
    setup_logger(getattr(logging, args.log_level))
    logging.info("Starting dewarping process...")
    model = load_model(
        model_class=LitGeoTrTemplate,
        model_name="geotr_template@inv3d",
        yaml_path="models.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    if model is None:
        logging.error("Failed to load the model. Exiting.")
        return
    logging.info("Model loaded successfully.")

if __name__ == "__main__":
    main()