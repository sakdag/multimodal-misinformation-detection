from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import zipfile
import pandas as pd
import requests
from getpass import getpass
import gdown
import shutil
import io
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

from src.utils.path_utils import get_project_root

# Define constants
PROJECT_ROOT: Path = get_project_root()
ZIP_FILE_PATH: str = str(PROJECT_ROOT / "data/raw/factify/factify_data.zip")
EXTRACTION_DIR: str = str(PROJECT_ROOT / "data/raw/factify/extracted")
TEMP_EXTRACTION_DIR: str = str(PROJECT_ROOT / "data/raw/factify/public_folder")
IMAGES_DIR: str = os.path.join(EXTRACTION_DIR, "images")
TRAIN_IMAGES_DIR: str = os.path.join(IMAGES_DIR, "train")
TEST_IMAGES_DIR: str = os.path.join(IMAGES_DIR, "test")
GDRIVE_FILE_URL: str = (
    "https://drive.google.com/uc?id=1ig7XEYU1UKDHrHgDYgqiARWvNdswgFEX"
)
STATS_FILE_PATH: str = os.path.join(EXTRACTION_DIR, "image_download_stats.json")


def ensure_directories() -> None:
    """Ensure that all necessary directories exist."""
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)


def download_zip() -> None:
    """Download the ZIP file from Google Drive if it does not already exist."""
    if os.path.exists(ZIP_FILE_PATH):
        print(f"Zip file already exists at {ZIP_FILE_PATH}. Skipping download...")
        return
    print("Downloading zip file from Google Drive...")
    gdown.download(GDRIVE_FILE_URL, ZIP_FILE_PATH, quiet=False)
    print(f"Downloaded zip file to {ZIP_FILE_PATH}")


def extract_zip() -> None:
    """Extract the ZIP file, rename the extracted folder, and handle CSV renaming."""
    train_csv_path: str = os.path.join(EXTRACTION_DIR, "train.csv")
    if os.path.exists(train_csv_path):
        print(f"{train_csv_path} already exists. Skipping extraction...")
        return
    print("Extracting zip file...")
    # Get password for the zip file
    password: str = getpass("Enter the password for the zip file: ")
    with zipfile.ZipFile(ZIP_FILE_PATH, "r") as zip_ref:
        try:
            zip_ref.extractall(
                str(PROJECT_ROOT / "data/raw/factify/"), pwd=password.encode()
            )
            print(f"Extracted files to temporary folder: {TEMP_EXTRACTION_DIR}")
        except RuntimeError:
            print("Incorrect password. Exiting...")
            exit(1)

    # Remove existing extracted directory if it exists
    if os.path.exists(EXTRACTION_DIR):
        shutil.rmtree(EXTRACTION_DIR)
        print(f"Removed existing directory: {EXTRACTION_DIR}")

    # Rename extracted folder
    if os.path.exists(TEMP_EXTRACTION_DIR):
        os.rename(TEMP_EXTRACTION_DIR, EXTRACTION_DIR)
        print(f"Renamed folder {TEMP_EXTRACTION_DIR} to {EXTRACTION_DIR}")

    # Rename val.csv to test.csv
    val_csv_path: str = os.path.join(EXTRACTION_DIR, "val.csv")
    test_csv_path: str = os.path.join(EXTRACTION_DIR, "test.csv")
    if os.path.exists(val_csv_path):
        os.rename(val_csv_path, test_csv_path)
        print(f"Renamed {val_csv_path} to {test_csv_path}")


from pathlib import Path


def process_image_download(
    row, images_folder: str, stats: dict, dataset_name: str
) -> None:
    """Process a single image download."""
    file_id: str = str(row["Id"])
    category: str = row.get("Category", "Unknown")
    claim_image_url: str = row.get("claim_image", "")
    document_image_url: str = row.get("document_image", "")

    # Ensure the folder exists before saving the images
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    # Update overall stats and category stats
    stats[dataset_name]["total_images"] += 2
    stats[dataset_name]["categories"][category]["total"] += 2

    # Download claim image
    if claim_image_url:
        success = download_image(
            claim_image_url, os.path.join(images_folder, f"{file_id}_claim.jpg")
        )
        if success:
            stats[dataset_name]["successful_downloads"] += 1
            stats[dataset_name]["categories"][category]["successful"] += 1
        else:
            stats[dataset_name]["failed_downloads"] += 1
            stats[dataset_name]["categories"][category]["failed"] += 1

    # Download document image
    if document_image_url:
        success = download_image(
            document_image_url, os.path.join(images_folder, f"{file_id}_document.jpg")
        )
        if success:
            stats[dataset_name]["successful_downloads"] += 1
            stats[dataset_name]["categories"][category]["successful"] += 1
        else:
            stats[dataset_name]["failed_downloads"] += 1
            stats[dataset_name]["categories"][category]["failed"] += 1


def download_images(
    csv_path: str, images_folder: str, stats: dict, use_threading: bool = True
) -> None:
    """
    Download images from URLs listed in the CSV file and update download stats.

    Args:
        csv_path (str): Path to the CSV file.
        images_folder (str): Folder where images will be saved.
        stats (dict): Dictionary to store download statistics.
        use_threading (bool): Whether to use threading for concurrent downloads.
    """
    print(f"Processing images for {csv_path}")
    df: pd.DataFrame = pd.read_csv(csv_path)
    dataset_name: str = "train" if "train" in csv_path else "test"
    stats[dataset_name] = {
        "total_images": 0,
        "successful_downloads": 0,
        "failed_downloads": 0,
        "categories": defaultdict(lambda: {"total": 0, "successful": 0, "failed": 0}),
    }

    if use_threading:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    process_image_download, row, images_folder, stats, dataset_name
                )
                for _, row in df.iterrows()
            ]

            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Downloading {dataset_name} images",
            ):
                pass
    else:
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Downloading {dataset_name} images"
        ):
            process_image_download(row, images_folder, stats, dataset_name)


def download_image(url: str, save_path: str) -> bool:
    """Download a single image from the given URL."""
    headers: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
        )
    }
    try:
        response: requests.Response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Raise an error for HTTP issues
        img: Image.Image = Image.open(io.BytesIO(response.content))
        img = img.convert("RGB")  # Ensure the image is in RGB format
        img.save(save_path)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from {url}: {e}")
    except Exception as e:
        print(f"Failed to process image from {url}: {e}")
    return False


def save_stats(stats: dict) -> None:
    """Save the image download statistics to a JSON file."""
    with open(STATS_FILE_PATH, "w") as stats_file:
        json.dump(stats, stats_file, indent=4)
    print(f"Image download stats saved to {STATS_FILE_PATH}")


def main() -> None:
    """Main function to orchestrate the workflow."""
    ensure_directories()

    # Download the zip file if it doesn't exist
    download_zip()

    # Extract the zip file and rename as needed
    extract_zip()

    # Initialize stats dictionary
    stats: dict = {}

    # Set threading preference
    use_threading = True  # Change to False to disable threading

    # Download images for train and test
    train_csv_path: str = os.path.join(EXTRACTION_DIR, "train.csv")
    test_csv_path: str = os.path.join(EXTRACTION_DIR, "test.csv")
    if os.path.exists(train_csv_path):
        download_images(
            train_csv_path, TRAIN_IMAGES_DIR, stats, use_threading=use_threading
        )
    else:
        print(f"{train_csv_path} not found.")
    if os.path.exists(test_csv_path):
        download_images(
            test_csv_path, TEST_IMAGES_DIR, stats, use_threading=use_threading
        )
    else:
        print(f"{test_csv_path} not found.")

    # Save stats to JSON
    save_stats(stats)


if __name__ == "__main__":
    main()
