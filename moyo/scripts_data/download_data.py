import argparse
import json
import os
import os.path as op
import sys
import traceback
import warnings
from glob import glob
from hashlib import sha256

import requests
from clint.textui import progress
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def download(url_files_dict, post_data):
    # Download urls
    logger.warning(
        f"Note: Download Images? = {args.download_images}. To get images, rerun with -i flag in the bash script")
    session = requests.Session()

    for url_tag, url_file in url_files_dict.items():
        if not op.exists(url_file):
            raise FileNotFoundError(f"File {url_file} not found")

        if args.download_images == "False":
            if "image" in url_tag:
                continue
        # Read the URLs from the file
        with open(url_file, "r") as f:
            urls = f.readlines()
        # Strip newline characters from the URLs
        urls = [url.strip() for url in urls]

        # Loop through the URLs and download the files
        logger.info(f"Start downloading from {url_file}\n")

        pbar = tqdm(urls)
        for url in pbar:
            # Get the filename from the URL
            filepath = url.split("=")[-1]
            if op.exists(op.join(args.out_dir, filepath)):
                logger.info(f"File {filepath} already exists. Skipping...")
                continue

            pbar.set_description(f"Downloading {url}")
            # Make a POST request with the username and password
            response = session.post(
                url,
                data=post_data,
                stream=True,
                verify=False,
                allow_redirects=True,
            )

            if response.status_code == 401:
                logger.warning(
                    f"Authentication failed for URLs in {url_file}. Username/password correct?"
                )
                sys.exit(1)


            # Write the contents of the response to a file
            out_p = op.join(args.out_dir, filepath)
            os.makedirs(op.dirname(out_p), exist_ok=True)
            total_length = int(response.headers.get("content-length"))
            with open(out_p, "wb") as f:
                for chunk in progress.bar(
                        response.iter_content(chunk_size=1024*1024*100),
                        expected_size=(total_length / (1024*1024*100)) + 1,
                ):
                    if chunk:
                        f.write(chunk)
                        f.flush()

def verify_checksum(out_dir):
    logger.info("Verifying checksums for downloaded files")

    fnames = glob(op.join(out_dir, "**/*"), recursive=True)
    pbar = tqdm(fnames)

    with open("./moyo/bash/assets/checksum.json", "r") as f:
        gt_checksum = json.load(f)

    hash_dict = {}
    for fname in pbar:
        if op.isdir(fname):
            continue
        if ".zip" not in fname:
            continue

        try:
            with open(fname, "rb") as f:
                pbar.set_description(f"Reading {fname}")
                data = f.read()
                hashcode = sha256(data).hexdigest()
                key = fname.replace("./", "")

                hash_dict[key] = hashcode
                if hashcode != gt_checksum[key]:
                    print(f"Error: {fname} has different checksum!")
                else:
                    pbar.set_description(f"Hashcode of {fname} is correct!")
                    # print(f'Hashcode of {fname} is correct!')
        except:
            print(f"Error processing {fname}")
            traceback.print_exc()
            continue

    out_p = op.join(out_dir, "checksum.json")
    with open(out_p, "w") as f:
        json.dump(hash_dict, f, indent=4, sort_keys=True)
    print(f"Checksum file saved to {out_p}!")

def unzip(url_files_dict):
    for url_tag, url_file in url_files_dict.items():
        if not op.exists(url_file):
            raise FileNotFoundError(f"File {url_file} not found")

        if args.download_images == "False":
            logger.warning("Note: Skipping image download. To get images, rerun with -i flag in the bash script")
            if "image" in url_tag:
                continue
        # Read the URLs from the file
        with open(url_file, "r") as f:
            urls = f.readlines()
        # Strip newline characters from the URLs
        urls = [url.strip() for url in urls]

        # Loop through the URLs and download the files
        logger.info(f"Start downloading from {url_file}")
        pbar = tqdm(urls)
        for url in pbar:
            pbar.set_description(f"Unzipping {url[-40:]}")

            # Get the filename from the URL
            filepath = url.split("=")[-1]

            # Write the contents of the response to a file
            out_p = op.join(args.out_dir, filepath)
            if out_p.endswith(".zip"):
                logger.info(f"Unzipping {out_p}")
                os.system(f"unzip -o {out_p} -d {args.out_dir}")


def delete_zip(url_files_dict):
    for url_tag, url_file in url_files_dict.items():
        if not op.exists(url_file):
            raise FileNotFoundError(f"File {url_file} not found")

        if args.download_images == "False":
            logger.warning("Note: Skipping image download. To get images, rerun with -i flag in the bash script")
            if "image" in url_tag:
                continue
        # Read the URLs from the file
        with open(url_file, "r") as f:
            urls = f.readlines()
        # Strip newline characters from the URLs
        urls = [url.strip() for url in urls]

        # Loop through the URLs and download the files
        logger.info(f"Start downloading from {url_file}")
        pbar = tqdm(urls)
        for url in pbar:
            pbar.set_description(f"Unzipping {url[-40:]}")

            # Get the filename from the URL
            filepath = url.split("=")[-1]

            # Write the contents of the response to a file
            out_p = op.join(args.out_dir, filepath)
            os.makedirs(op.dirname(out_p), exist_ok=True)
            logger.info(f"Unzipping {out_p}")
            os.system(f"unzip -o {out_p} -d {args.out_dir}")


def download_data(args):
    # Define the username and password
    flag = "MOYO"

    username = args.username
    password = args.password
    password_fake = "*" * len(password)

    logger.info(f"Username: {username}")
    logger.info(f"Password: {password_fake}")

    post_data = {"username": username, "password": password}

    # Get URL paths
    url_files_dict = {
        "cameras_urls": op.join(args.url_dir, "cameras.txt"),
        "image_train_urls": op.join(args.url_dir, "images_train.txt"),
        "image_val_urls": op.join(args.url_dir, "images_val.txt"),
        "com_urls": op.join(args.url_dir, "coms.txt"),
        "pressure_urls": op.join(args.url_dir, "pressures.txt"),
        "smplx_fits_urls": op.join(args.url_dir, "smplx_fits.txt"),
    }

    # Download urls
    download(url_files_dict, post_data)
    # Verify checksums
    verify_checksum(args.out_dir)

    # Unzip files
    if args.unzip == "True":
        unzip(url_files_dict)
    # Delete zip files
    if args.delete == "True":
        delete_zip(url_files_dict)
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from a list of URLs")
    parser.add_argument(
        "--url_dir",
        type=str,
        help="Path to file directory containing URLs",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Path to folder to store downloaded files",
        required=True,
    )
    parser.add_argument(
        "--username", type=str, help="Username for the MOYO data download, register at moyo.is.mpg.de", required=True
    )
    parser.add_argument(
        "--password", type=str, help="Password for the MOYO data download, register at moyo.is.mpg.de", required=True
    )
    parser.add_argument(
        "--download_images", type=str, help="Whether to download the images", default="True"
    )
    parser.add_argument(
        "--unzip", type=str, help="Whether to unzip the downloaded zip files", default="True"
    )
    parser.add_argument(
        "--delete", type=str, help="Whether to delete the downloaded zip files", default="True"
    )

    args = parser.parse_args()

    download_data(args)
