"""
Utility functions for downloading and extracting surrogate model data from Zenodo.
"""

import os
import zipfile
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('redback_surrogates.download')

# Zenodo record information for Type II surrogate model
ZENODO_RECORD_ID = "15575033"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

# File information
SURROGATE_FILES = {
    "TypeII_surrogate_Sarin+2025.zip": {
        "url": f"{ZENODO_BASE_URL}/TypeII_surrogate_Sarin+2025.zip",
        "md5": "ca6be4fc2e391df23d20a750da86c45e"
    }
}


def get_surrogate_data_dir() -> Path:
    """
    Get the directory where surrogate data should be stored.
    Creates the directory if it doesn't exist.

    Returns:
        Path: Path to the surrogate data directory
    """
    dirname = os.path.dirname(__file__)
    dir = Path(dirname)

    # Create the surrogate_data directory in the user's home directory
    data_dir = dir / "surrogate_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


def calculate_md5(filepath: Path) -> str:
    """
    Calculate the MD5 hash of a file.

    Args:
        filepath: Path to the file

    Returns:
        str: MD5 hash as a hexadecimal string
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url: str, target_path: Path) -> bool:
    """
    Download a file from a URL with progress bar.

    Args:
        url: URL to download from
        target_path: Path where the file should be saved

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Make a streaming GET request
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total file size
        total_size = int(response.headers.get('content-length', 0))

        # Create progress bar
        progress_bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=f"Downloading {target_path.name}"
        )

        # Write the file
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()
        return True

    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        if target_path.exists():
            target_path.unlink()
        return False


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """
    Extract a ZIP file with progress tracking.

    Args:
        zip_path: Path to the ZIP file
        extract_dir: Directory to extract files to

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Create extraction directory if it doesn't exist
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size for progress bar
            total_size = sum(item.file_size for item in zip_ref.infolist())
            extracted_size = 0

            with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"Extracting {zip_path.name}"
            ) as progress_bar:
                for file_info in zip_ref.infolist():
                    zip_ref.extract(file_info, extract_dir)
                    extracted_size += file_info.file_size
                    progress_bar.update(file_info.file_size)

        return True

    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False


def download_surrogate_data(force_download: bool = False) -> Optional[Path]:
    """
    Download and extract surrogate data from Zenodo.

    Args:
        force_download: If True, re-download even if files already exist

    Returns:
        Optional[Path]: Path to the surrogate data directory if successful, None otherwise
    """
    # Get the surrogate data directory
    data_dir = get_surrogate_data_dir()

    # Track overall success
    success = True

    for filename, file_info in SURROGATE_FILES.items():
        file_path = data_dir / filename

        # Check if the file already exists
        if file_path.exists() and not force_download:
            logger.info(f"File {filename} already exists")

            # If MD5 hash is specified, verify the file
            if "md5" in file_info:
                logger.info("Verifying file integrity...")
                file_hash = calculate_md5(file_path)
                if file_hash == file_info["md5"]:
                    logger.info(f"Hash matches for {filename}, skipping download")
                    continue
                else:
                    logger.warning(f"Hash mismatch for {filename}, re-downloading")
                    file_path.unlink()
            else:
                # If no hash is specified, assume file is valid
                logger.info(f"No hash to check for {filename}, using existing file")
                continue

        # Download the file
        logger.info(f"Downloading {filename} from Zenodo...")
        if not download_file(file_info["url"], file_path):
            logger.error(f"Failed to download {filename}")
            success = False
            continue

        # Verify hash if provided
        if "md5" in file_info:
            file_hash = calculate_md5(file_path)
            if file_hash != file_info["md5"]:
                logger.error(f"Hash mismatch for downloaded file {filename}")
                file_path.unlink()
                success = False
                continue
            else:
                logger.info(f"Hash verification successful for {filename}")

        # Extract if it's a zip file
        if filename.endswith(".zip"):
            logger.info(f"Extracting {filename}...")
            if not extract_zip(file_path, data_dir):
                logger.error(f"Failed to extract {filename}")
                success = False
                continue
            else:
                logger.info(f"Successfully extracted {filename}")

    if success:
        logger.info(f"Successfully downloaded and extracted surrogate data to {data_dir}")
        return data_dir
    else:
        logger.error("Some errors occurred during download or extraction")
        return None


def get_surrogate_file_path(relative_path: str) -> Optional[Path]:
    """
    Get the path to a specific file in the surrogate data directory.

    Args:
        relative_path: Relative path of the file within the surrogate data directory

    Returns:
        Optional[Path]: Full path to the file if it exists, None otherwise
    """
    data_dir = get_surrogate_data_dir()
    file_path = data_dir / relative_path

    if not file_path.exists():
        logger.warning(f"Surrogate file {relative_path} not found")
        # Try to download data if file doesn't exist
        download_surrogate_data()

        # Check again after download attempt
        if not file_path.exists():
            logger.error(f"Surrogate file {relative_path} still not found after download attempt")
            return None

    return file_path


def list_surrogate_files(subdir: Optional[str] = None) -> List[Path]:
    """
    List all files in the surrogate data directory.

    Args:
        subdir: Optional subdirectory to list files from

    Returns:
        List[Path]: List of file paths
    """
    data_dir = get_surrogate_data_dir()

    if subdir:
        search_dir = data_dir / subdir
    else:
        search_dir = data_dir

    if not search_dir.exists():
        logger.warning(f"Directory {search_dir} does not exist")
        return []

    # Only return files, not directories
    return [p for p in search_dir.glob("**/*") if p.is_file()]


def get_md5_hash(file_path: str) -> str:
    """
    Utility function to calculate and print MD5 hash of a file.
    Useful for getting the hash of the downloaded file to add to SURROGATE_FILES.

    Args:
        file_path: Path to the file

    Returns:
        str: MD5 hash value
    """
    path = Path(file_path)
    if not path.exists():
        print(f"File {file_path} does not exist")
        return ""

    hash_value = calculate_md5(path)
    print(f"MD5 hash for {path.name}: {hash_value}")
    return hash_value