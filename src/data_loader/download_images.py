import os
import argparse
import pandas as pd
import requests
import json
import io
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from PIL import Image
from src.utils.path_utils import get_project_root

# Constants
PROJECT_ROOT = get_project_root()
EXTRACTION_DIR = str(PROJECT_ROOT / "data/raw/factify/extracted")
IMAGES_DIR = os.path.join(EXTRACTION_DIR, "images")


def ensure_directories(images_folder):
    """Ensure the image directory exists."""
    os.makedirs(images_folder, exist_ok=True)


def download_image(url, save_path):
    """Download a single image."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Raise an error for HTTP issues
        img = Image.open(io.BytesIO(response.content))
        img = img.convert("RGB")  # Ensure the image is in RGB format
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
        return False


def process_image(row, images_folder, stats, dataset_name):
    """Process claim and document image downloads."""
    file_id = str(row["Id"])
    category = row.get("Category", "Unknown")
    claim_image_url = row.get("claim_image", "")
    document_image_url = row.get("document_image", "")

    # Ensure category stats exist
    stats["categories"][category]["total_claim"] += 1
    stats["categories"][category]["total_document"] += 1

    # Download claim image
    if claim_image_url:
        success = download_image(
            claim_image_url, os.path.join(images_folder, f"{file_id}_claim.jpg")
        )
        if success:
            stats["successful_claim"] += 1
            stats["categories"][category]["successful_claim"] += 1

    # Download document image
    if document_image_url:
        success = download_image(
            document_image_url, os.path.join(images_folder, f"{file_id}_document.jpg")
        )
        if success:
            stats["successful_document"] += 1
            stats["categories"][category]["successful_document"] += 1


def download_images(dataset, use_threading):
    """Download images for the specified dataset (train or test)."""
    csv_path = os.path.join(EXTRACTION_DIR, f"{dataset}.csv")
    images_folder = os.path.join(IMAGES_DIR, dataset)
    stats_file_path = os.path.join(
        EXTRACTION_DIR, f"{dataset}_image_download_stats.json"
    )
    ensure_directories(images_folder)

    if not os.path.exists(csv_path):
        print(f"CSV file not found for {dataset}: {csv_path}")
        return

    stats = {
        "successful_claim": 0,
        "successful_document": 0,
        "categories": defaultdict(
            lambda: {
                "total_claim": 0,
                "successful_claim": 0,
                "total_document": 0,
                "successful_document": 0,
            }
        ),
    }

    df = pd.read_csv(csv_path)

    if use_threading:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_image, row, images_folder, stats, dataset)
                for _, row in df.iterrows()
            ]
            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Downloading {dataset} images",
            ):
                pass
    else:
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Downloading {dataset} images"
        ):
            process_image(row, images_folder, stats, dataset)

    with open(stats_file_path, "w") as stats_file:
        json.dump(stats, stats_file, indent=4)
    print(f"Image download stats saved to {stats_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Download images for Factify dataset.")
    parser.add_argument(
        "--dataset",
        choices=["train", "test"],
        help="Specify which dataset to download images for (train or test). If not specified, both will be downloaded.",
    )
    parser.add_argument(
        "--use-threading",
        action="store_true",
        default=True,
        help="Enable threading for image downloads (default: True).",
    )
    args = parser.parse_args()

    if args.dataset:
        # Run for the specified dataset
        download_images(args.dataset, args.use_threading)
    else:
        # Run for both train and test if no dataset is specified
        print("No dataset specified. Downloading images for both train and test...")
        for dataset in ["train", "test"]:
            download_images(dataset, args.use_threading)


if __name__ == "__main__":
    main()
