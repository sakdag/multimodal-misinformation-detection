import os
import pandas as pd
from pathlib import Path
from src.utils.path_utils import get_project_root

# Constants
PROJECT_ROOT = get_project_root()
RAW_DIR = PROJECT_ROOT / "data/raw/factify"
PROCESSED_DIR = PROJECT_ROOT / "data/preprocessed"
IMAGES_DIR = RAW_DIR / "extracted/images"


def ensure_directories():
    """Ensure that necessary directories exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)  # Create 'data/preprocessed'


def preprocess_csv(dataset: str):
    """
    Preprocess the given dataset CSV (train or test).

    Args:
        dataset (str): The dataset name ('train' or 'test').
    """
    # Paths
    ensure_directories()

    csv_path = RAW_DIR / f"extracted/{dataset}.csv"
    processed_csv_path = PROCESSED_DIR / f"{dataset}.csv"
    images_folder = IMAGES_DIR / dataset

    if not csv_path.exists():
        print(f"Dataset CSV not found: {csv_path}")
        return

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Rename columns
    df.rename(
        columns={
            "Id": "id",
            "claim_image": "claim_image",
            "claim": "claim",
            "claim_ocr": "claim_ocr",
            "document_image": "evidence_image",
            "document": "evidence",
            "document_ocr": "evidence_ocr",
            "Category": "category",
        },
        inplace=True,
    )

    # Update file paths for images
    def update_image_path(row, column_name):
        """Update the image path if it exists, else leave as None."""
        image_file = row[column_name]
        file_id = row["id"]
        if column_name == "claim_image":
            file_path = images_folder / f"{file_id}_claim.jpg"
        elif column_name == "evidence_image":
            file_path = images_folder / f"{file_id}_document.jpg"
        else:
            return None

        # Check if the file exists
        if file_path.exists():
            # Use the relative path starting from "/data/.."
            return str(file_path.relative_to(PROJECT_ROOT))
        return None

    df["claim_image"] = df.apply(
        lambda row: update_image_path(row, "claim_image"), axis=1
    )
    df["evidence_image"] = df.apply(
        lambda row: update_image_path(row, "evidence_image"), axis=1
    )

    # Save the processed CSV
    df.to_csv(processed_csv_path, index=False)
    print(f"Processed {dataset}.csv saved to {processed_csv_path}")


def main():
    for dataset in ["train", "test"]:
        preprocess_csv(dataset)


if __name__ == "__main__":
    main()
