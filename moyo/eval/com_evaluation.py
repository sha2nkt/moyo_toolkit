"""
Usage python eval/com_evaluation.py --img_folder ../data/moyo/20221004_with_com/images/val/ --pp_folder ../data/moyo//20221004_with_com/mosh/val/ --nexus_com_c3d_folder ../data/moyo//20221004_with_com/com/val

"""

import argparse
import copy
import glob
import json
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import c3d

from tqdm import tqdm
import pickle as pkl

from moyo.utils.misc import colors, copy2cpu as c2c

import smplx
from moyo.utils.geometry import matrix_to_axis_angle, ea2rm, axis_angle_to_matrix
from moyo.utils.constants import frame_select_dict_nexus as frame_select_dict
from moyo.utils.mesh_utils import smplx_breakdown
from moyo.utils.biomech_eval_utils import BiomechanicalEvaluator

VICON_FPS = 60
PRESSURE_MAP_FPS = 60
IOI_FPS = 30


def read_keypoints(keypoint_fn):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(
            person_data['pose_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])

        left_hand_keyp = np.array(
            person_data['hand_left_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])

        right_hand_keyp = np.array(
            person_data['hand_right_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])

        face_keypoints = np.array(
            person_data['face_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

        contour_keyps = np.array(
            person_data['face_keypoints_2d'],
            dtype=np.float32).reshape([-1, 3])[:17, :]

        body_keypoints = np.concatenate([body_keypoints, left_hand_keyp,
                                         right_hand_keyp, face_keypoints, contour_keyps], axis=0)

        keypoints.append(body_keypoints)
    return keypoints


def select_keypoints(keypoints, img):
    img_center = np.array(img.shape[:2]) / 2  # height, width
    # select keypoints closest to image center weighted by inverse confidence
    if len(keypoints) > 1:
        kpselect = np.inf * np.ones(len(keypoints))
        # major joints openpose
        op_to_12 = [9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7]
        for idx, personkpts in enumerate(keypoints):
            kpdist = personkpts[op_to_12, :2] - img_center
            kpdist = np.linalg.norm(kpdist, axis=1)
            kpconf = np.dot(kpdist, (- personkpts[op_to_12, 2] + 1))
            kpselect[idx] = kpconf
        kpselidx = np.argmin(kpselect)
    elif len(keypoints) == 1:
        kpselidx = 0
    else:
        keypoints = None
    keypoints = np.stack(keypoints[kpselidx:kpselidx + 1])
    return keypoints


def visualize_mesh(bm_output, faces, frame_id=0, display=False):
    imw, imh = 1600, 1600

    body_mesh = trimesh.Trimesh(vertices=c2c(bm_output.vertices[frame_id]),
                                faces=faces,
                                vertex_colors=np.tile(colors['grey'], (6890, 1)),
                                process=False,
                                maintain_order=True)
    return body_mesh


def get_trans_offset(pelvis, smplx_params, trans, body_model):
    '''
    SOMA fits are in AMASS format (xy plane is ground) but SMPLify-XMC uses Pyrender format (xz plane is ground). global_orient is corrected by 270 rotation along x-axis to match the two formats.
    While the ground plane is z=0 in soma, since global_orient is wrt pelvis, this would mean the ground plane is not at y=0 after rotation.
    Fix: distance of pelvis to ground in preserved. Use it to push the mesh up after rotation such that the ground plane is at y=0.
    Args:
        pelvis: pelvis joint before rotation
    '''
    bs = trans.shape[0]
    init_root_orient = smplx_params['global_orient']
    pelvis_height = pelvis[:, 2]  # this is always conserved

    new_smplx_params = copy.deepcopy(smplx_params)
    # Rotate AMASS root orient to smplify-xmc format
    R_init = axis_angle_to_matrix(init_root_orient)
    R1 = ea2rm(torch.tensor([[np.radians(270)]]), torch.tensor([[np.radians(0)]]),
               torch.tensor([[np.radians(0)]])).float().to(R_init.device)
    R = torch.bmm(R1.expand(bs, -1, -1), R_init)
    new_smplx_params['global_orient'] = matrix_to_axis_angle(R)

    # posed body with hand, with global orient
    body_model_output = body_model(
        global_orient=new_smplx_params['global_orient'],
        body_pose=new_smplx_params['body_pose'])

    new_pelvis = body_model_output.joints[:, 0]
    new_ground_plane_height = new_pelvis[:, 1] - pelvis_height
    trans_offset = -new_ground_plane_height
    return trans_offset


def smplx_to_mesh(body_params, model_folder, model_type, gender='neutral'):
    with torch.no_grad():
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        num_betas = 10

        smplx_params = smplx_breakdown(body_params, device)

        trans = torch.from_numpy(body_params['trans']).float().to(device)

        betas = torch.from_numpy(body_params['betas']).float().to(device).unsqueeze(0)
        betas = betas[:, :num_betas]  # only using 10 betas

        body_model_params = dict(model_path=model_folder,
                                 model_type=model_type,
                                 gender=gender,
                                 v_template=smplx_params['v_template'],
                                 # joint_mapper=joint_mapper,
                                 batch_size=trans.shape[0],
                                 create_global_orient=True,
                                 create_body_pose=True,
                                 create_betas=True,
                                 num_betas=num_betas,
                                 create_left_hand_pose=True,
                                 create_right_hand_pose=True,
                                 create_expression=True,
                                 create_jaw_pose=True,
                                 create_leye_pose=True,
                                 create_reye_pose=True,
                                 create_transl=True,
                                 use_pca=False,
                                 flat_hand_mean=True,
                                 dtype=torch.float32)

        body_model = smplx.create(**body_model_params).to(device)
        body_model_output = body_model(transl=trans,
                                       global_orient=smplx_params['global_orient'],
                                       body_pose=smplx_params['body_pose'],
                                       left_hand_pose=smplx_params['left_hand_pose'],
                                       right_hand_pose=smplx_params['right_hand_pose'])

        pelvis = body_model_output.joints[:, 0]
        trans_offset = get_trans_offset(pelvis, smplx_params, trans, body_model)
        zero_vec = torch.zeros_like(trans_offset)
        transl = torch.stack([zero_vec, trans_offset, zero_vec], dim=-1)

    return body_model_output, smplx_params, transl, body_model.faces

def extract_com_from_c3d(c3d_file):
    with open(c3d_file, 'rb') as f:
        data = c3d.Reader(f)  # data contains the header of the c3d files, info such as the label name
        com_idx = np.where(data.point_labels == 'CentreOfMass                  ')
        com_idx_floor = np.where(data.point_labels == 'CentreOfMassFloor             ')
        coms, coms_floor = [], []
        for frame in data.read_frames():  # call read_frames() to iterate across the marker positions
            frame = frame[1][:, :3]  # this contains the marker positions
            coms.append(frame[com_idx, :])
            coms_floor.append(frame[com_idx_floor, :])

    coms = np.concatenate(coms, axis=0)
    coms_floor = np.concatenate(coms_floor, axis=0)
    return coms, coms_floor



def yoga82_extra(img_folder, pp_folder, nexus_com_c3d_folder, model_folder,
                 model_type='smplx', subsampling_factor=1, save_outputs=False):
    imgpaths_ = []
    imgnames_ = []
    subject_ids_ = []
    relpaths_ = []
    heights_ = []
    weights_ = []
    genders_ = []
    gt_coms_ = []
    pred_coms_ = []
    com_errors_ = []
    pp_body_poses_ = []
    pp_left_hand_poses_ = []
    pp_right_hand_poses_ = []
    pp_global_orients_ = []
    pp_transls_ = []

    # presented poses
    pp_names = os.listdir(pp_folder)
    pp_names = [os.path.splitext(x)[0] for x in pp_names if x[0] != '.' and '.pkl' in x]

    all_pose_names = [pp_name.replace('_stageii', '') for pp_name in pp_names]
    all_pose_names = sorted(all_pose_names)

    pp_pose_cache = {}

    no_images_found = set()
    no_pp_found = set()

    smplx_faces = smplx.create(model_path=model_folder, model_type=model_type).faces
    smplx_faces = torch.tensor(smplx_faces.astype(np.int64), dtype=torch.long)
    com_evaluator = BiomechanicalEvaluator(faces=smplx_faces)

    for i, pname in enumerate(tqdm(sorted(all_pose_names))):
        if 'female_stagei' in pname: # do not analyze female stage1 file
            continue
        # get all possible variations
        pp_pkl = osp.join(pp_folder, pname + '_stageii.pkl')
        pp_params = pkl.load(open(pp_pkl, 'rb'))
        # read height, weight, gender meta
        height = None
        weight = None
        gender = 'female'
        subj_id = 'unknown'
        pp_body_model_output, smplx_params, transls, faces = smplx_to_mesh(pp_params, model_folder,
                                                                           model_type, gender=gender)
        # get pressure xml
        com_c3d = osp.join(nexus_com_c3d_folder, pname + '.c3d')
        gt_coms, gt_coms_floor = extract_com_from_c3d(com_c3d)

        # pick selected frame
        selected_frame = frame_select_dict['_'.join(pname.split('_')[6:])]

        # get all images for this pose
        img_dir = Path(osp.join(img_folder, pname))
        for cam in range(1, 7):
            cam_name = Path('YOGI_Cam_' + str(cam).zfill(2))
            img_dir_cam = Path.joinpath(img_dir, cam_name)
            types = ('*.jpg', '*.png')  # the tuple of file types
            images = []
            for files in types:
                images.extend(glob.glob(osp.join(img_dir_cam, files)))
            if len(images) == 0:
                print(f'No images found for {pname}')
                no_images_found.add(pname)

            # subsample frames
            images.sort()

            # folders to log outputs
            if cam == 1 and save_outputs:
                out_img_dir = osp.join(pp_folder, 'images_small', pname, cam_name)
                os.makedirs(out_img_dir, exist_ok=True)
                out_obj_dir = osp.join(pp_folder, 'meshes', pname, cam_name)
                os.makedirs(out_obj_dir, exist_ok=True)

            for img_path in images[
                            :-1:subsampling_factor]:  # sometimes, the last frame is missing in pressure or vicon data
                # get frame correspondences
                img_fnum = int(img_path.split('_')[-1].split('.')[0])
                vicon_fnum = img_fnum * (VICON_FPS // IOI_FPS)

                # only run on selected frames
                if img_fnum != selected_frame:
                    continue

                # load image
                img_fn, _ = osp.splitext(osp.split(img_path)[1])
                relpath = osp.dirname(img_path).replace(str(img_dir), '').strip('/')

                if cam == 1 and save_outputs:
                    img = cv2.imread(img_path).astype(np.float32)
                    # resize image to half size
                    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                    cv2.imwrite(osp.join(out_img_dir, f'{pname}_{vicon_fnum}.jpg'), img)
                    print('[[Image Saved]]\n', osp.join(out_img_dir, f'{pname}_{vicon_fnum}.jpg'))

                transl = transls[vicon_fnum]
                pp_mesh = visualize_mesh(pp_body_model_output, faces, frame_id=vicon_fnum)
                pp_verts = pp_mesh.vertices
                # pp_verts = pp_body_model_output.vertices[vicon_fnum]
                vertices = torch.tensor(pp_verts).unsqueeze(0)
                if cam == 1 and save_outputs:
                    out_obj = osp.join(out_obj_dir, f'{pname}_{vicon_fnum}.obj')
                    pp_mesh.export(out_obj)
                    print('[[Mesh]]\n', out_obj)

                # get com for current frame
                gt_com = gt_coms[vicon_fnum, :]
                gt_com = gt_com/1000.0 # convert to meters
                print('[[GT COM]]\n', gt_com)

                # get com_error
                pred_com, com_error = com_evaluator.evaluate_com(gt_com, vertices)
                pred_com = pred_com.numpy()
                com_error = com_error.squeeze().numpy()

                body_pose = smplx_params['body_pose'].squeeze().cpu()
                left_hand_pose = smplx_params['left_hand_pose'].squeeze().cpu()
                right_hand_pose = smplx_params['right_hand_pose'].squeeze().cpu()
                global_orient = smplx_params['global_orient'].squeeze().cpu()
                transl = transls.squeeze().cpu()

                # add to dict if keypoints were detected
                imgpaths_.append(img_path)
                imgnames_.append(img_fn)
                subject_ids_.append(subj_id)
                relpaths_.append(relpath)
                heights_.append(height)
                weights_.append(weight)
                genders_.append(gender)
                gt_coms_.append(gt_com)
                pred_coms_.append(pred_com)
                com_errors_.append(com_error)
                pp_body_poses_.append(np.array(body_pose))
                pp_left_hand_poses_.append(np.array(left_hand_pose))
                pp_right_hand_poses_.append(np.array(right_hand_pose))
                pp_global_orients_.append(np.array(global_orient))
                pp_transls_.append(np.array(transl))

    # get mean com errro
    print('Mean COM error: ', np.mean(com_errors_))
    print(f'Folders with no images: \n {no_images_found}')
    print(f'Folders with no presented poses: \n {no_pp_found}')

    # save as db file
    outfile = osp.join('../data', 'dbs_biomech', 'newmoyo_gt_50_nexus_221004.npz')
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    np.savez(outfile,
             imgpath=imgpaths_,
             subject_id=subject_ids_,
             imgname=imgnames_,
             relpath=relpaths_,
             height=heights_,
             weight=weights_,
             gender=genders_,
             gt_com=gt_coms_,
             pred_com=pred_coms_,
             com_error=com_errors_,
             body_pose=pp_body_poses_,
             left_hand_pose=pp_left_hand_poses_,
             right_hand_pose=pp_right_hand_poses_,
             global_orient=pp_global_orients_,
             transl=pp_transls_
             )

    # Average com_errors_
    com_errors_ = np.array(com_errors_)
    print('Average COM error: ', np.mean(com_errors_))


if __name__ == "__main__":
    """
    yoga-82 dir /ps/scratch/ps_shared/4Paul/Yoga-82
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', required=True,
                        help='folder containing yoga 82 images')
    parser.add_argument('--nexus_com_c3d_folder', required=True,
                        help='folder containing GT pressure CSVs')
    parser.add_argument('--pp_folder', required=True,
                        help='presented poses folder')
    parser.add_argument('--model_folder', required=False, default='/ps/project/common/smplifyx/models/',
                        help='path to SMPL/SMPL-X model folder')
    parser.add_argument('--save_outputs', action='store_true', default=False,
                         help='if enabled, saves outputs to disk, but slows down eval')

    args = parser.parse_args()
    yoga82_extra(args.img_folder, args.pp_folder, args.nexus_com_c3d_folder, args.model_folder)