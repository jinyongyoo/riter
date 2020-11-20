import os
import pathlib
import shutil
import tempfile
import zipfile

import filelock
import requests
import tqdm

from .misc import logger


def path_in_cache(file_path):
    try:
        os.makedirs(RITER_CACHE_DIR)
    except FileExistsError:  # cache path exists
        pass
    return os.path.join(RITER_CACHE_DIR, file_path)


def s3_url(uri):
    return "https://riter.s3.amazonaws.com/" + uri


def download_if_needed(file_name, folder=False):
    """File name will be saved as `.cache/riter/<file_name>`. If it
    doesn't exist on disk, the file will be downloaded.
    Args:
        file_name (str): path to folder or file in cache
    Returns:
        str: path to the downloaded folder or file on disk
    """
    cache_dest_path = path_in_cache(file_name)
    os.makedirs(os.path.dirname(cache_dest_path), exist_ok=True)
    # Use a lock to prevent concurrent downloads.
    cache_dest_lock_path = cache_dest_path + ".lock"
    cache_file_lock = filelock.FileLock(cache_dest_lock_path)
    cache_file_lock.acquire()
    # Check if already downloaded.
    if os.path.exists(cache_dest_path):
        cache_file_lock.release()
        return cache_dest_path
    # If the file isn't found yet, download the zip file to the cache.
    downloaded_file = tempfile.NamedTemporaryFile(dir=RITER_CACHE_DIR, delete=False)
    http_get(file_name, downloaded_file)
    # Move or unzip the file.
    downloaded_file.close()
    if folder:
        unzip_file(downloaded_file.name, cache_dest_path)
    else:
        logger.info(f"Copying '{downloaded_file.name}' to '{cache_dest_path}'.")
        shutil.copyfile(downloaded_file.name, cache_dest_path)
    cache_file_lock.release()
    # Remove the temporary file.
    os.remove(downloaded_file.name)
    logger.info(f"Successfully saved '{file_name}' to cache.")
    return cache_dest_path


def unzip_file(path_to_zip_file, unzipped_folder_path):
    """Unzips a .zip file to folder path."""
    logger.info(f"Unzipping file '{path_to_zip_file}' to '{unzipped_folder_path}'.")
    enclosing_unzipped_path = pathlib.Path(unzipped_folder_path).parent
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(enclosing_unzipped_path)


def http_get(file_name, out_file, proxies=None):
    """Get contents of a URL and save to a file.
    https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    """
    folder_s3_url = s3_url(file_name)
    logger.info(f"Downloading '{folder_s3_url}'.")
    req = requests.get(folder_s3_url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if req.status_code == 403:  # Not found on AWS
        raise Exception(f"Could not find '{file_name}' on server.")
    progress = tqdm.tqdm(unit="B", unit_scale=True, total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            out_file.write(chunk)
    progress.close()


RITER_CACHE_DIR = os.environ.get(
    "RITER_CACHE_DIR", os.path.expanduser("~/.cache/riter")
)
