## MOYO 🧘🏻‍♀️: A dataset containing complex yoga poses, multi-view videos, SMPL-X meshes, pressure and body center of mass

<p align="center"> 
    <img src="moyo/docs/moyo_logo_transparent_cropped.png" alt="Image" width="450
" height="100" />
</p>

[ [Project Page](https://ipman.is.tue.mpg.de) ][ [MOYO Dataset](https://moyo.is.tue.mpg.de) ][ [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tripathi_3D_Human_Pose_Estimation_via_Intuitive_Physics_CVPR_2023_paper.pdf) ][ [Video](https://www.youtube.com/watch?v=eZTtLUMnGIg) ][ [Register MoYo Account](https://moyo.is.tue.mpg.de/register.php) ]

<p align="center">    
    <img src="moyo/docs/moyo_offer-crop.png" alt="Image" width="80%"/>
</p>

## News :triangular_flag_on_post:

- [2023/12/23] AMASS format data released :confetti_ball: (thanks [Giorgio Becherini](https://ps.is.mpg.de/person/gbecherini) and [Neelay Shah](https://neelays.github.io/)). Please check the [AMASS](#downloading-the-dataset-in-amass-format) section for more details. 

This is a repository for download, preprocessing, visualizing, running evaluations on the MOYO dataset.

Our dataset provides a challenging new benchmark; it has extreme poses, strong self-occlusion, and
significant body-ground and self-contact.

<p align="center">
    <img src="moyo/docs/montage_broad.jpg" alt="Image" width="100%"/>
</p>

### Getting started

Get a copy of the code:

```bash
git clone https://github.com/sha2nkt/moyo.git
```

General Requirements:

- Python 3.9

[//]: # (- torch 1.13.0)

[//]: # (- CUDA 11.6 &#40;check `nvcc --version`&#41;)

[//]: # (- pytorch3d 0.7.3)
Install the environment:

```bash
ENV_NAME=moyo_p39
conda create -n $ENV_NAME python=3.9
conda activate $ENV_NAME
pip install .
conda install -c conda-forge ezc3d
```

### Downloading the dataset

MOYO provides the following data:

1. Data with SMPLX with hand markers and pressure, but no coms **[recommended]**
    - `20220923_20220926_with_hands/images [741G]`: Full 2K-resolution images
    - `20220923_20220926_with_hands/cameras.zip [1.7M]`: Camera parameters for the 8 IOI RGB cameras
    - `20220923_20220926_with_hands/mosh.zip [1.3G]`: SMPL-X fits with hand markers
    - `20220923_20220926_with_hands/mosh_smpl.zip [1.3G]`: SMPL fits
    - `20220923_20220926_with_hands/pressure.zip [298M]`: Pressure mat data
    - `20220923_20220926_with_hands/vicon.zip [257M]`: Raw marker data from Vicon
2. Data with SMPLX without hand markers, but includes both coms and pressure
    - `20221004_with_com/images [635G]`: Full 2K-resolution images
    - `20221004_with_com/cameras.zip [840K]`: Camera parameters for the 8 IOI RGB cameras
    - `20221004_with_com/mosh.zip [1,1G]`: SMPL-X fits *without* hand markers
    - `20221004_with_com/mosh_smpl.zip [1,1G]`: SMPL fits
    - `20221004_with_com/pressure.zip [517M]`: Pressure mat data
    - `20221004_with_com/coms.md [489M]`: Center of mass data from Vicon plug-in gait
  
Note: The SMPL fits are obtained from the MOYO SMPL-X fits using the [SMPLX-to-SMPL conversion script](https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md#smpl-x-to-smpl). 

⚠️ Register accounts on [MOYO](https://moyo.is.tue.mpg.de/register.php), and then use your username and password when
prompted.

The following command downloads the full dataset to `./data/` minus the images and unzips them (-u flag).

```bash
bash ./moyo/bash/download_moyo.sh -o ./data/ -u
```

If you additionally want to download the images, you can run the following command:

```bash
bash ./moyo/bash/download_moyo.sh -o ./data/ -u -i
```

The following command downloads the full dataset to `./data/` (including images), unzips the downloaded zips and deletes
the zip files to save space. This will take a while but will give you a fully usable dataset.

```bash 
bash ./moyo/bash/download_moyo.sh -o ./data/ -u -i -d
```

### Downloading the dataset in AMASS format

MOYO provides the following AMASS formats:
- SMPLH_FEMALE
- SMPLH_NEUTRAL
- SMPLX_FEMALE
- SMPLX_NEUTRAL

⚠️ Register accounts on [MOYO](https://moyo.is.tue.mpg.de/register.php), and then use your username and password when
prompted.

The following command downloads the full dataset to ./data/ minus the images and unzips them (-u flag).
```bash
bash ./moyo/bash/download_moyo.sh -o ./data/ -u -a <AMASS_FORMAT>
```

The following command downloads the full dataset to `./data/` (including images), unzips the downloaded zips and deletes
the zip files to save space. This will take a while but will give you a fully usable dataset.

```bash 
bash ./moyo/bash/download_moyo.sh -o ./data/ -u -d -a <AMASS_FORMAT>
```

Replace the ```<AMASS_FORMAT>``` with the split name you want to download: ```SMPLH_FEMALE```, ```SMPLH_NEUTRAL```, ```SMPLX_FEMALE``` or ```SMPLX_NEUTRAL```.

### Projecting marker on the image

We include a simple script to project vicon markers on the RGB images using the provided camera parameters. A similar
approach can be used to project the full mesh.

```bash
python scripts/ioi_vicon_frame_sync.py --img_folder ../data/moyo/20220923_20220926_with_hands/images/ --c3d_folder ../data/moyo/20220923_20220926_with_hands/vicon --cam_folder_first ../data/moyo/20220923_20220926_with_hands/cameras/20220923/220923_Afternoon_PROCESSED_CAMERA_PARAMS/cameras_param.json --cam_folder_second ../data/moyo/20220923_20220926_with_hands/cameras/20220926/220926_Morning_PROCESSED_CAMERA_PARAMS/cameras_param.json --output_dir ../data/moyo_images_mocap_projected --frame_offset 1 --split val 
```

<p align="left">    
    <img src="assets/220923_yogi_body_hands_03596_Tree_Pose_or_Vrksasana_-a_YOGI_Cam_01_0190_markers.png" alt="Image" width="40%"/>
</p>

To visualize the exact the pressure mat markers alignment with respect to the subject, we provide a blender file
in `assets/mat_marker_configuration.blend`.

### Biomechanical Evaluation

We provide evaluation scripts to run evaluations fro estimated pressure and com w.r.t groud truth as reported in our
paper.

1. Pressure Evaluation

```bash
python eval/pressure_map_evaluation.py --img_folder ../data/moyo/20220923_20220926_with_hands/images/val/ --pp_folder ../data/moyo/20220923_20220926_with_hands/mosh/val/ --pressure_xml_folder ../data/moyo/20220923_20220926_with_hands/pressure/val/xml --pressure_csv_folder ../data/moyo/20220923_20220926_with_hands/pressure/val/single_csv
```

If you would like to visualize, per frame results, please add the `--save_outputs` flag.

2. COM Evaluation

```bash
python eval/com_evaluation.py --img_folder ../data/moyo/20221004_with_com/images/val/ --pp_folder ../data/moyo//20221004_with_com/mosh/val/ --nexus_com_c3d_folder ../data/moyo//20221004_with_com/com/val
```

If you would like to visualize, per frame results, please add the `--save_outputs` flag.

The above implementation is not optimized for speed. We will be releasing a faster version soon.

### Citation

If you found this code helpful, please consider citing our work:

```bibtex
@inproceedings{tripathi2023ipman,
    title = {{3D} Human Pose Estimation via Intuitive Physics},
    author = {Tripathi, Shashank and M{\"u}ller, Lea and Huang, Chun-Hao P. and Taheri Omid
    and Black, Michael J. and Tzionas, Dimitrios},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition (CVPR)},
    month = {June},
    year = {2023}
}
```

### License

See [LICENSE](LICENSE).

### Acknowledgments

Constructing the MOYO dataset is a huge effort. The authors deeply thank Tsvetelina Alexiadis, Taylor McConnell, Claudia
Gallatz, Markus Höschle, Senya Polikovsky, Camilo Mendoza, Yasemin Fincan, Leyre Sanchez and Matvey Safroshkin for data
collection, Giorgio Becherini for MoSh++, Joachim Tesch and Nikos Athanasiou for visualizations, Zicong Fan, Vasselis
Choutas and all of Perceiving Systems for fruitful discussions. This work was funded by the International Max Planck
Research School for Intelligent Systems (IMPRS-IS) and in part by the German Federal Ministry of Education and
Research (BMBF), Tübingen AI Center, FKZ: 01IS18039B.".

We would also like to extend a special thanks to [Giorgio Becherini](https://ps.is.mpg.de/person/gbecherini) and [Neelay Shah](https://neelays.github.io/) for helping with the release of the AMASS version of the MOYO dataset.  

### Contact

For technical questions, please create an issue. For other questions, please contact `ipman@tue.mpg.de`.

For commercial licensing, please contact `ps-licensing@tue.mpg.de`.
