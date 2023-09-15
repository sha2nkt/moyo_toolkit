"""
Script to get 2D projects from mosh_fits.

python scripts/ioi_vicon_frame_sync.py --img_folder ../data/moyo/20220923_20220926_with_hands/images/ --c3d_folder ../data/moyo/20220923_20220926_with_hands/vicon --cam_folder_first ../data/moyo/20220923_20220926_with_hands/cameras/20220923/220923_Afternoon_PROCESSED_CAMERA_PARAMS/cameras_param.json --cam_folder_second ../data/moyo/20220923_20220926_with_hands/cameras/20220926/220926_Morning_PROCESSED_CAMERA_PARAMS/cameras_param.json --output_dir ../data/moyo_images_mocap_projected --frame_offset 1 --split val"""

import argparse
import glob
import json
import os
import os.path as osp

import cv2
import ipdb
import numpy as np
import torch
import trimesh
from ezc3d import c3d as ezc3d
from tqdm import tqdm

from moyo.utils.constants import frame_select_dict_combined as frame_select_dict
from moyo.utils.misc import colors, copy2cpu as c2c


class Opt:
    def __init__(self):
        self.pose_track = False
        self.tracking = False
        self.showbox = False


def visualize_mesh(bm_output, faces, frame_id=0, display=False):
    imw, imh = 1600, 1600

    body_mesh = trimesh.Trimesh(vertices=c2c(bm_output.vertices[frame_id]),
                                faces=faces,
                                vertex_colors=np.tile(colors['grey'], (6890, 1)),
                                process=False,
                                maintain_order=True)

    if display:
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)

    return body_mesh


def project2d(j3d, cam_params, downsample_factor=1.0):
    """
    Project 3D points to 2D
    Args:
        j3d: (N, 3) 3D joints
        cam_params: dict
        downsample_factor: resize factor

    Returns:
        j2d : 2D joint locations
    """
    mm2m = 1000
    j3d = torch.tensor(j3d, dtype=torch.float32)

    # cam intrinsics
    f = cam_params['focal'] * downsample_factor
    cx = cam_params['princpt'][0] * downsample_factor
    cy = cam_params['princpt'][1] * downsample_factor

    # cam extrinsics
    R = torch.tensor(cam_params['rotation'])
    t = -torch.mm(R, torch.tensor(cam_params['position'])[:, None]).squeeze()  # t= -RC

    # cam matrix
    K = torch.tensor([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]]).to(j3d.device)

    Rt = torch.cat([R, t[:, None]], dim=1).to(j3d.device)

    # apply extrinsics
    bs = j3d.shape[0]
    j3d_cam = torch.bmm(Rt[None, :, :].expand(bs, -1, -1), j3d[:, :, None])
    j2d = torch.bmm(K[None, :].expand(bs, -1, -1), j3d_cam)
    j2d = j2d / j2d[:, [-1]]
    return j2d[:, :-1, :].squeeze()


def visualize_on_img(j2d, img_name, out_dir):
    # Visualize the joints
    pose_name = img_name.split('/')[-2]
    img_num = img_name.split('/')[-1]
    img = cv2.imread(img_name)
    fname = img_name.split('/')[-1]
    os.makedirs(out_dir, exist_ok=True)
    ext = osp.splitext(fname)[1]
    for n in range(j2d.shape[0]):
        # check if nan
        if np.any(np.isnan(j2d[n, :])):
            continue
        cor_x, cor_y = int(j2d[n, 0]), int(j2d[n, 1])
        cv2.circle(img, (cor_x, cor_y), 1, (0, 255, 0), 5)

    out_img_path = osp.join(out_dir, fname).replace(f'{ext}', '_markers.png')
    cv2.imwrite(out_img_path, img)
    print(f'{out_img_path} is saved')


def main(img_folder, c3d_folder, model_folder, output_dir, cam_folders, frame_offset, split, downsample_factor):
    # presented poses
    c3d_folder = os.path.join(c3d_folder, split, 'c3d')
    c3d_names = os.listdir(c3d_folder)
    c3d_names = [os.path.splitext(x)[0] for x in c3d_names if x[0] != '.' and '.c3d' in x]
    c3d_names = sorted(c3d_names)

    img_folder = os.path.join(img_folder, split)
    # list all folders in img_folder
    img_pose_folders = os.listdir(img_folder)
    img_pose_folders = [item for item in img_pose_folders if os.path.isdir(os.path.join(img_folder, item))]

    for i, c3d_name in enumerate(tqdm(c3d_names)):
        print(f'Processing {i}: {c3d_name}')
        # load images, loop through and read smplx and keypoints
        pose_name = '-'.join('_'.join(c3d_name.split('_')[5:]).split('-')[:-1])
        if pose_name == "Happy_Baby_Pose":
            pose_name = "Happy_Baby_Pose_or_Ananda_Balasana_"  # this name was changed during mocap

        # get soma fit
        try:
            c3d_path = glob.glob(osp.join(c3d_folder, f'{c3d_name}.c3d'))[0]
        except:
            print(f'{c3d_folder}/{c3d_name}_stageii.pkl does not exist. SKIPPING!!!')
            continue

        c3d = ezc3d(c3d_path)

        markers3d = c3d['data']['points'].transpose(2, 1,
                                                    0)  # Frames x NumPoints x 4 (x,y,z,1) in homogenous coordinates

        try:
            c3d_var = '_'.join(c3d_name.split('_')[5:])
            selected_frame = frame_select_dict[c3d_var]
        except:
            print(f'{c3d_var} does not exist in frame_selection_dict. SKIPPING!!!')
            continue

        j3d = markers3d[selected_frame]

        # load images belonging to presented pose
        img_dir = [dir for dir in img_pose_folders if c3d_var in dir]

        if len(img_dir) == 0:
            print(f'{pose_name} does not exist in {img_folder}. SKIPPING!!!')
            continue
        if len(img_dir) > 1:
            print(f'{pose_name} has more than one folder. Choose one:')
            for i, dir in enumerate(img_dir):
                print(f'{i}. {dir}')
            select_idx = int(input())
            if select_idx > len(img_dir) - 1:
                print(f'{select_idx} is out of range. SKIPPING!!!')
                continue
            img_dir = img_dir[select_idx]
        if len(img_dir) == 1:
            img_dir = img_dir[0]

        img_dir = osp.join(img_folder, img_dir)
        image_paths = glob.glob(osp.join(img_dir, '*/*.jpg'))

        # select a particular image frame
        selected_img_frame = (selected_frame // 2) + frame_offset
        image_paths = [path for path in image_paths if
                       int(selected_img_frame) == int(osp.splitext(path)[0].split('_')[-1])]

        # only take Cam_01
        image_paths = [path for path in image_paths if 'Cam_01' in path]

        if len(image_paths) == 0:
            print(f'No images found for {img_dir}')

        out_dir = osp.join(output_dir, pose_name)

        for img_path in image_paths:
            img_name = osp.basename(img_path)
            sessionid = img_name.split('_')[0]
            if '220923' in sessionid:
                cam_path = cam_folders['220923']
            elif '220926' in sessionid:
                cam_path = cam_folders['220926']
            else:
                print(f'Invalid {sessionid}. Check!!!')
                ipdb.set_trace()
            # Load camera matrix from json file
            with open(cam_path, 'rb') as fp:
                cameras = json.load(fp)

            name_splits = img_name.split('_')
            cam_num = int(name_splits[name_splits.index('Cam') + 1])
            cam_id = f'cam_{cam_num}'

            cam_params = cameras[cam_id]
            j2d = project2d(j3d, cam_params, downsample_factor=downsample_factor)

            # convert to openpose format
            visualize_on_img(j2d.cpu().numpy(), img_path, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', required=True,
                        help='folder_containing_ioi_data')
    parser.add_argument('--c3d_folder', required=True,
                        help='folder containing raw c3d mocap files')
    parser.add_argument('--cam_folder_first', required=True,
                        help='folder containing camera matrix for first session 210727')
    parser.add_argument('--cam_folder_second', required=True,
                        help='folder containing camera matrix for second session 211117')
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'],
                        help='split to process')
    parser.add_argument('--model_folder', required=False, default='/ps/project/common/smplifyx/models/',
                        help='path to SMPL/SMPL-X model folder')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    parser.add_argument('--downsample_factor', type=float, default=0.5,
                        help='Downsample factor for the images. Release images are downsampled by 0.5')
    parser.add_argument('--frame_offset', type=int,
                        help='To get the frame offset, divide the vicon fnum by 2 and add this offset')

    args = parser.parse_args()

    cam_folders = {'220923': args.cam_folder_first, '220926': args.cam_folder_second}

    main(args.img_folder, args.c3d_folder, args.model_folder, args.output_dir, cam_folders, args.frame_offset,
         args.split, args.downsample_factor)
