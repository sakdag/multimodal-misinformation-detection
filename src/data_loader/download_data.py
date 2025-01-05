import os
import zipfile
import gdown
from getpass import getpass
import shutil
from src.utils.path_utils import get_project_root

# Constants
PROJECT_ROOT = get_project_root()
ZIP_FILE_PATH = str(PROJECT_ROOT / "data/raw/factify/factify_data.zip")
EXTRACTION_DIR = str(PROJECT_ROOT / "data/raw/factify/extracted")
# Change to str(PROJECT_ROOT / "data/raw/factify/public_folder") if running for Factify1
TEMP_EXTRACTION_DIR = str(PROJECT_ROOT / "data/raw/factify/factify2")
# Factify1
# GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1ig7XEYU1UKDHrHgDYgqiARWvNdswgFEX"
# Factify2
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1i7cM3KyG1_Ue5TtBBp38raFQqBxHeGDK"


def ensure_directories():
    """Ensure necessary directories exist."""
    os.makedirs(os.path.dirname(ZIP_FILE_PATH), exist_ok=True)


def download_zip():
    """Download the ZIP file if it doesn't already exist."""
    if os.path.exists(ZIP_FILE_PATH):
        print(f"Zip file already exists at {ZIP_FILE_PATH}. Skipping download...")
        return
    print("Downloading zip file from Google Drive...")
    gdown.download(GDRIVE_FILE_URL, ZIP_FILE_PATH, quiet=False)
    print(f"Downloaded zip file to {ZIP_FILE_PATH}")


def extract_zip():
    """Extract the ZIP file and handle folder and file renaming."""
    train_csv_path = os.path.join(EXTRACTION_DIR, "train.csv")
    if os.path.exists(train_csv_path):
        print(f"{train_csv_path} already exists. Skipping extraction...")
        return
    print("Extracting zip file...")
    # Get password for the zip file
    password = getpass("Enter the password for the zip file: ")
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
    val_csv_path = os.path.join(EXTRACTION_DIR, "val.csv")
    test_csv_path = os.path.join(EXTRACTION_DIR, "test.csv")
    if os.path.exists(val_csv_path):
        os.rename(val_csv_path, test_csv_path)
        print(f"Renamed {val_csv_path} to {test_csv_path}")


def main():
    ensure_directories()
    download_zip()
    extract_zip()


if __name__ == "__main__":
    main()
