#!/bin/bash
set -e

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
# urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Optional flags
DOWNLOAD_IMAGES=False
UNZIP_FLAG=False
DELETE_FLAG=False

# Command line arguments
# take optfolder path from command line
while getopts "o:iud" opt; do
  case "$opt" in
  o) OUT_DIR="$OPTARG" ;;
  i) DOWNLOAD_IMAGES=True ;;
  u) UNZIP_FLAG=True ;;
  d) DELETE_FLAG=True ;;
  esac
done

echo -e "\nMOYO dataset download script." 1>&2
echo -e "\nUsage: bash $0 -o <out_folder> [-i] [-u] [-d]" 1>&2
echo -e "-i: Download full resolution images" 1>&2
echo -e "-u: Unzip downloaded files" 1>&2
echo -e "-d: Delete downloaded zip files after unzipping" 1>&2


echo -e "\n" 1>&2
read -p "Enter your MOYO website email address: " USERNAME
read -s -p "Enter your MOYO website password: " PASSWORD
#USERNAME=$(urle $USERNAME)
#PASSWORD=$(urle $PASSWORD)
echo -e "\n" 1>&2

echo $OUT_DIR
echo -e "Download Images: $DOWNLOAD_IMAGES" 1>&2
python $PWD/moyo/scripts_data/download_data.py --url_dir $PWD/moyo/bash/assets/urls/ \
                                     --out_dir $OUT_DIR \
                                     --username $USERNAME \
                                     --password $PASSWORD \
                                     --download_images $DOWNLOAD_IMAGES \
                                     --unzip $UNZIP_FLAG \
                                     --delete $DELETE_FLAG \


