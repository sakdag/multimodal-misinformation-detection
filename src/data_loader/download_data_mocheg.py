import os
import requests
import tarfile
from tqdm import tqdm

DATA_URL: str = (
    "http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/dataset/latest_dataset/mocheg_with_tweet_2023_03.tar.gz"
)
RAW_DATA_DIR: str = "data/raw"
ARCHIVE_NAME: str = "mocheg_with_tweet_2023_03.tar.gz"
CHUNK_SIZE: int = 16 * 1024 * 1024  # 16 MB

# Ensure the raw data directory exists
os.makedirs(RAW_DATA_DIR, exist_ok=True)
archive_path: str = os.path.join(RAW_DATA_DIR, ARCHIVE_NAME)


def check_disk_space(required_space_gb: int) -> bool:
    """Check if there is enough free disk space."""
    stat = os.statvfs(RAW_DATA_DIR)
    free_space_gb: float = (stat.f_bavail * stat.f_frsize) / (1024**3)
    return free_space_gb > required_space_gb


def download_data() -> None:
    """Download the data if not already present and extract it."""
    # Check if the data file already exists
    if os.path.exists(archive_path):
        print(f"Data already downloaded at {archive_path}. Skipping download.")
        return

    # Ensure enough disk space (approximate)
    required_space_gb: int = 80  # Adjust based on expected file size + extraction space
    if not check_disk_space(required_space_gb):
        print(f"Not enough disk space. At least {required_space_gb} GB required.")
        return

    # Download the data in larger chunks
    print(f"Downloading data from {DATA_URL}...")
    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status()  # Ensure the URL is accessible

    total_size: int = int(response.headers.get("content-length", 0))
    with open(archive_path, "wb") as file, tqdm(
        desc=ARCHIVE_NAME,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

    print(f"Download completed: {archive_path}")

    # Extract the tar.gz file
    extract_data(archive_path)


def extract_data(archive_path: str) -> None:
    """Extract the downloaded tar.gz file."""
    print(f"Extracting data from {archive_path}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=RAW_DATA_DIR)
    print(f"Data extracted to {RAW_DATA_DIR}")


if __name__ == "__main__":
    download_data()
