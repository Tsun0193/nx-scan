import torch
import yaml
import gdown
import logging
from pytorch_lightning import LightningModule
from pathlib import Path


def setup_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_from_checkpoint(model_class: type[LightningModule], 
                         checkpoint_file: Path) -> LightningModule:
    """
    Load a LightningModule model from checkpoint.

    Args:
        model_class (type): The model class to instantiate.
        checkpoint_file (Path): Path to the checkpoint file.

    Returns:
        LightningModule: The loaded model.
    """
    if not checkpoint_file.is_file():
        raise ValueError(f"Checkpoint not found at: {checkpoint_file.resolve()}")
    return model_class.load_from_checkpoint(str(checkpoint_file))


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

        # Download checkpoint (cached)
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
    

def main():
    """
    Main function to execute the dewarping process.
    """
    setup_logger(logging.DEBUG)  # Set to DEBUG for detailed output
    logging.info("Starting dewarping process...")
    model = load_model(
        model_class=LightningModule,  # Replace with your actual model class
        model_name="geotr_template@inv3d",
        yaml_path="models.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Model loaded: {model is not None}")

if __name__ == "__main__":
    main()