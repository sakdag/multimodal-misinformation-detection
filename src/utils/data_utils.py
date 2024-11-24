import os
import pandas as pd
from PIL import Image
from typing import Dict, Any
from pathlib import Path
from src.utils.path_utils import get_project_root

# Constants
PROJECT_ROOT = get_project_root()
PREPROCESSED_DIR = PROJECT_ROOT / "data/preprocessed"


def get_preprocessed_data(dataset: str = "train") -> pd.DataFrame:
    """
    Load the preprocessed data for the specified dataset.

    Args:
        dataset (str): Either 'train' or 'test'. Defaults to 'train'.

    Returns:
        pd.DataFrame: A DataFrame containing the preprocessed data.
    """
    csv_path = PREPROCESSED_DIR / f"{dataset}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset CSV not found: {csv_path}")

    return pd.read_csv(csv_path)


def load_images_for_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load the claim and evidence images for a given row of data.

    Args:
        row (Dict[str, Any]): A dictionary representing a row of preprocessed data.

    Returns:
        Dict[str, Any]: A dictionary containing the original row with loaded images added.
    """
    result = row.copy()  # Copy the original row to avoid modifying the input
    claim_image_path = row.get("claim_image")
    evidence_image_path = row.get("evidence_image")

    if claim_image_path and os.path.exists(claim_image_path):
        try:
            result["claim_image"] = Image.open(claim_image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load claim image from {claim_image_path}: {e}")
            result["claim_image"] = None
    else:
        result["claim_image"] = None

    if evidence_image_path and os.path.exists(evidence_image_path):
        try:
            result["evidence_image"] = Image.open(evidence_image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load evidence image from {evidence_image_path}: {e}")
            result["evidence_image"] = None
    else:
        result["evidence_image"] = None

    return result
