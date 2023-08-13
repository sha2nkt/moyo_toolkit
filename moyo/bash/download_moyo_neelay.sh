#!/bin/bash


# Copyright (C) 2023, Max Planck Institute for Intelligent Systems, Tuebingen, Germany
#
# Script to download the MOYO dataset (https://moyo.is.tue.mpg.de/). The script resumes download from last state if interrupted.
# 
# Requirements: wget (`apt install wget`)
#
# Usage: 
#   1. Register at https://moyo.is.tue.mpg.de/ with `email` and `password`

#   2. Read and accept license conditions at https://moyo.is.tue.mpg.de/license.html

#   3. Run this script with the provided urls_paths.txt file to download data:
#       `bash download_moyo.sh urls_paths.txt`
#
#   4. Input the `email` and `password` you used for registration when prompted


# Return URL encoded version of input string
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}";done; echo; }

function download_file {
    local USERNAME=$1
    local PASSWORD=$2 
    local url=$3
    local filename=$4
    wget --post-data "username=$USERNAME&password=$PASSWORD" $url -O $filename --no-check-certificate --continue &> /dev/null
    return $?
}

# Optional flags
UNZIP_FLAG=false
DELETE_FLAG=false

while getopts "t:ud" opt; do
    case "$opt" in
        t) urls_paths_txt_file="$OPTARG" ;;
        u) UNZIP_FLAG=true ;;
        d) DELETE_FLAG=true ;;
    esac
done

if [ -z "$urls_paths_txt_file" ]; then
    echo -e "\nUsage: $0 -t <urls_paths_txt_file> [-u] [-d]"
    exit 1
fi

if [ ! -f "$urls_paths_txt_file" ]; then
    echo -e "\nError: .txt file '$urls_paths_txt_file' not found."
    exit 1
fi

echo -e "\nMOYO dataset download script." 1>&2

echo -e "\nURLS_PATHS file provided: $urls_paths_txt_file" 1>&2

if $DELETE_FLAG && ! $UNZIP_FLAG; then
    echo -e "\nError: -d flag can only be used with -u flag."
    exit 1
fi

if $UNZIP_FLAG; then
    echo -e "\nUnzip flag set. Downloaded zip files will be unzipped." 1>&2
fi

if $DELETE_FLAG; then
    echo -e "\nDelete flag set. Downloaded zip files will be deleted after being unzipped." 1>&2
fi

echo -e "\n" 1>&2
read -p "Enter your MOYO website email address: " USERNAME
read -s -p "Enter your MOYO website password: " PASSWORD
USERNAME=$(urle $USERNAME)
PASSWORD=$(urle $PASSWORD)
echo -e "\n" 1>&2

# Create an array to store the paths of the files being downloaded
zip_files=()

# Read URL and filepath pairs from the urls_paths_txt_file and download the files
while IFS=", " read -r url filepath || [[ -n "$url" ]]; do
    echo -e "\nDownloading file from URL: $url to path: $filepath"
    mkdir -p "$(dirname "$filepath")"
    download_file "$USERNAME" "$PASSWORD" "$url" "$filepath"
    zip_files+=("$filepath")
done < "$urls_paths_txt_file"

# Unzip downloaded files if the -u flag is provided
if [ "$UNZIP_FLAG" = true ]; then
    for zip_file in "${zip_files[@]}"; do
        echo -e "\nUnzipping file: $zip_file to directory: $(dirname "$zip_file")/$(basename "$zip_file" .zip)"
        unzip -q "$zip_file" -d "$(dirname "$zip_file")"
    done
fi

# Delete downloaded files if the -d flag is provided
if [ "$DELETE_FLAG" = true ]; then
    for file in "${zip_files[@]}"; do
        echo -e "\nDeleting file: $file"
        rm -f "$file"
    done
fi