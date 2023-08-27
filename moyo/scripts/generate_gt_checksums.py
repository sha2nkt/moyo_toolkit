import argparse
import glob
import hashlib
import json
import os

from tqdm import tqdm


def calculate_checksum(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            hasher.update(byte_block)
    return hasher.hexdigest()


def generate_checksums(data_dir):
    checksums = {}
    zip_files = glob.glob(os.path.join(data_dir, '**/*.zip'), recursive=True)

    for file_path in tqdm(zip_files):
        checksum = calculate_checksum(file_path)
        checksums[file_path] = checksum

    with open('./moyo/bash/assets/checksum.json', 'w') as json_file:
        json.dump(checksums, json_file, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Generate checksums for zip files in a folder recursively.')
    parser.add_argument('--data_dir', help='Path to the folder containing zip files')
    args = parser.parse_args()

    generate_checksums(args.data_dir)


if __name__ == '__main__':
    main()
