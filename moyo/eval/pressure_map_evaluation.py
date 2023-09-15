"""
Usage
python tests/biomechanical_evaluations/pressure_map_evaluation.py --img_folder /ps/data/YOGI/YOGI_220923_03596/IOI/20220923_Yogi_capture_PNG_MH/ --pp_folder /is/cluster/scratch/gbecherini/mosh/YOGI_2/YOGI_2_def-vt_type-smplx_grab_yogi_head-corr-no_wt-fingers-15000.0_wt-feet-300.0_wt-head-10000.0_seed-100_OFFSET/220923/03596 --pressure_xml_folder /ps/project/vicondata/ViconDataCaptures/OfficialCaptures/YOGI_2/pressure/220923/xml/ --pressure_csv_folder /ps/project/vicondata/ViconDataCaptures/OfficialCaptures/YOGI_2/pressure/220923/single_csv/YOGI/03596/

"""

import argparse
import copy
import glob
import json
import os
import os.path as osp

import cv2
import numpy as np
import torch
import trimesh

from tqdm import tqdm
import pickle as pkl
from scipy import stats


import sys

from moyo.utils.misc import colors, copy2cpu as c2c


import smplx
from moyo.utils.geometry import matrix_to_axis_angle, ea2rm, axis_angle_to_matrix
from moyo.utils.constants import frame_select_dict_combined as frame_select_dict
from moyo.utils.biomech_eval_utils import BiomechanicalEvaluator
from moyo.utils.pressure_mat_utils import PressureMat
from psbody.mesh import Mesh

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

    if display:
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)

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

def smplx_breakdown(bdata, device):
    num_frames = len(bdata['trans'])

    bdata['poses'] = bdata['fullpose']

    global_orient = torch.from_numpy(bdata['poses'][:, :3]).float().to(device)
    body_pose = torch.from_numpy(bdata['poses'][:, 3:66]).float().to(device)
    jaw_pose = torch.from_numpy(bdata['poses'][:, 66:69]).float().to(device)
    leye_pose = torch.from_numpy(bdata['poses'][:, 69:72]).float().to(device)
    reye_pose = torch.from_numpy(bdata['poses'][:, 72:75]).float().to(device)
    left_hand_pose = torch.from_numpy(bdata['poses'][:, 75:120]).float().to(device)
    right_hand_pose = torch.from_numpy(bdata['poses'][:, 120:]).float().to(device)

    # if bdata['v_template_fname'] == '/ps/project/amass/MOCAP/PS_MoCaps/YOGI/subject_data/subject_v_template/210727_03596/smplx_female.ply':
    v_template_fname = '/is/cluster/scratch/stripathi/pycharm_remote/yogi/data/v_templates/220923_yogi_03596_minimal_simple_female/mesh.ply'
    # else:
    #     print("Something weird happening with V_template loading!!!!")
    #     import ipdb; ipdb.set_trace()
    v_template = Mesh(filename=v_template_fname)

    body_params = {'global_orient': global_orient, 'body_pose': body_pose,
                  'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                  'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                   'v_template': torch.Tensor(v_template.v).to(device),}
    return body_params


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

def yoga82_extra(img_folder, pp_folder, pressure_xml_folder, pressure_csv_folder, model_folder, save_outputs=False,
                 model_type='smplx', subsampling_factor=1):
    imgpaths_ = []
    imgnames_ = []
    subject_ids_ = []
    relpaths_ = []
    heights_ = []
    weights_ = []
    genders_ = []
    pressures_ = []
    pressure_heatmaps_ = []
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
    pressure_evaluator = BiomechanicalEvaluator(faces=smplx_faces)

    for i, pname in enumerate(tqdm(all_pose_names)):
        print(pname)
        if 'Cow_Face_Pose_or_Gomukhasana_-c' in pname: # pressure mat was broken for this pose
            continue
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
        pressure_xml = osp.join(pressure_xml_folder, pname + '.xml')
        pressure_csv = osp.join(pressure_csv_folder, pname + '.csv')
        pressure_mat_c3d = pressure_xml.replace('/xml/', '/pressure_mat_c3d/').replace('.xml', '.c3d')
        # initialize pressure mat
        pressure_mat = PressureMat(pressure_csv, pressure_xml, pressure_mat_c3d)

        # pick selected frame
        selected_frame = frame_select_dict['_'.join(pname.split('_')[5:])]

        # get all images for this pose
        img_dir = osp.join(img_folder, pname)
        for cam in [1]:
            cam_name = 'YOGI_Cam_' + str(cam).zfill(2)
            img_dir_cam = osp.join(img_dir, cam_name)
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
                out_pressure_heatmap_dir = osp.join(pp_folder, 'pressure_heatmaps', pname, cam_name)
                os.makedirs(out_pressure_heatmap_dir, exist_ok=True)
                out_obj_dir = osp.join(pp_folder, 'meshes', pname, cam_name)
                os.makedirs(out_obj_dir, exist_ok=True)

            for img_path in images[
                            :-1:subsampling_factor]:  # sometimes, the last frame is missing in pressure or vicon data
                # get frame correspondences
                img_fnum = int(img_path.split('_')[-1].split('.')[0])

                #only run on selected frames
                if img_fnum != selected_frame:
                    continue
                vicon_fnum = img_fnum * (VICON_FPS // IOI_FPS)
                pressure_fnum = img_fnum * (PRESSURE_MAP_FPS // IOI_FPS)
                assert vicon_fnum == pressure_fnum

                # load image
                img_fn, _ = osp.splitext(osp.split(img_path)[1])
                relpath = osp.dirname(img_path).replace(img_dir, '').strip('/')

                if cam == 1 and save_outputs:
                    img = cv2.imread(img_path).astype(np.float32)
                    # resize image to half size
                    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                    cv2.imwrite(osp.join(out_img_dir, f'{pname}_{pressure_fnum}.jpg'), img)
                    print('[[Image Saved]]\n', osp.join(out_img_dir, f'{pname}_{pressure_fnum}.jpg'))
                if cam != 1:
                    continue

                # get pressure for current frame
                # increase_heatmap size before saving
                gt_pressure = pressure_mat.gt_pressures[pressure_fnum]
                gt_heatmap = pressure_mat.gt_heatmaps[pressure_fnum]
                gt_cop_relative = pressure_mat.gt_cops_relative[pressure_fnum]
                if np.isnan(gt_cop_relative).any():
                    continue
                # gt_cop_metric = pressure_mat.gt_cops_metric[pressure_fnum]
                if cam == 1 and save_outputs:
                    pressure_name = str(img_fnum).zfill(4) + '.png'
                    heatmap_large = cv2.resize(gt_heatmap, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
                    # add gt_cop point to heatmap
                    cv2.drawMarker(heatmap_large, (
                        int(gt_cop_relative[0] / pressure_mat.mat_size_metric[0] * heatmap_large.shape[1]),
                        int(gt_cop_relative[1] / pressure_mat.mat_size_metric[1] * heatmap_large.shape[0])),
                                   [0, 255, 0],
                                   markerType=cv2.MARKER_STAR, markerSize=40, thickness=5)
                    # heatmap_large = cv2.flip(heatmap_large, 0) # to have better visualization wrt to blender mat configuration
                    cv2.imwrite(os.path.join(out_pressure_heatmap_dir, f'{pname}_{pressure_name}'),
                                heatmap_large)
                    print('[[GT Pressure Heatmap Saved]]\n',
                          os.path.join(out_pressure_heatmap_dir, f'{pname}_{pressure_name}'))



                transl = transls[vicon_fnum]
                pp_mesh = visualize_mesh(pp_body_model_output, faces, frame_id=img_fnum)
                pp_verts = pp_mesh.vertices
                # pp_verts = pp_body_model_output.vertices[img_fnum]
                vertices = torch.tensor(pp_verts).unsqueeze(0)
                if cam == 1 and save_outputs:
                    mesh_name = str(img_fnum).zfill(4) + '.obj'
                    out_obj = osp.join(out_obj_dir, f'{pname}_{mesh_name}')
                    pp_mesh.export(out_obj)
                    print('[[Mesh]]\n', out_obj)


                # get per-vertex pressure
                pressure_evaluator.evaluate_pressure(gt_pressure,
                                                   gt_cop_relative,
                                                   vertices,
                                                   pressure_mat.mat_size,
                                                   pressure_mat.mat_size_metric,
                                                   pressure_mat.marker_positions_metric)
                if cam == 1 and save_outputs:
                    # iou first visualization
                    pred_heatmap = pressure_evaluator.iou_first.pred_heatmaps[-1]
                    pred_cop_relative = pressure_evaluator.iou_first.pred_cops_relative[-1]
                    cop_error = pressure_evaluator.iou_first.cop_errors[-1]
                    iou = pressure_evaluator.iou_first.ious[-1]
                    frame_diff = pressure_evaluator.iou_first.frame_diffs[-1]
                    best_cop_w = pressure_evaluator.iou_first.best_cop_w[-1]
                    best_cop_k = pressure_evaluator.iou_first.best_cop_k[-1]

                    pred_heatmap_large = cv2.resize(pred_heatmap, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
                    cv2.drawMarker(pred_heatmap_large, (
                        int(pred_cop_relative[0] / pressure_mat.mat_size_metric[0] * pred_heatmap_large.shape[1]),
                        int(pred_cop_relative[1] / pressure_mat.mat_size_metric[1] * pred_heatmap_large.shape[0])),
                                   [0, 255, 255],
                                   markerType=cv2.MARKER_STAR, markerSize=40, thickness=5)
                    cv2.putText(pred_heatmap_large, f'{cop_error*1000:.2f} mm', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
                    cv2.putText(pred_heatmap_large, f'{iou:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255,255, 0), 2)
                    cv2.putText(pred_heatmap_large, f'{frame_diff:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.putText(pred_heatmap_large, f'cop_w: {best_cop_w:.2f}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.putText(pred_heatmap_large, f'cop_k: {best_cop_k:.2f}', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.imwrite(os.path.join(out_pressure_heatmap_dir, f'{pname}_{pressure_fnum}_pred_iou_first.png'),
                                pred_heatmap_large)
                    print('[[PRED Pressure Heatmap Saved]]\n',
                          os.path.join(out_pressure_heatmap_dir, f'{pname}_{pressure_fnum}_pred_iou_first.png'))

                    # cop error first visualizations
                    pred_heatmap = pressure_evaluator.cop_error_first.pred_heatmaps[-1]
                    pred_cop_relative = pressure_evaluator.cop_error_first.pred_cops_relative[-1]
                    cop_error = pressure_evaluator.cop_error_first.cop_errors[-1]
                    iou = pressure_evaluator.cop_error_first.ious[-1]
                    frame_diff = pressure_evaluator.cop_error_first.frame_diffs[-1]
                    best_cop_w = pressure_evaluator.cop_error_first.best_cop_w[-1]
                    best_cop_k = pressure_evaluator.cop_error_first.best_cop_k[-1]

                    pred_heatmap_large = cv2.resize(pred_heatmap, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
                    cv2.drawMarker(pred_heatmap_large, (
                        int(pred_cop_relative[0] / pressure_mat.mat_size_metric[0] * pred_heatmap_large.shape[1]),
                        int(pred_cop_relative[1] / pressure_mat.mat_size_metric[1] * pred_heatmap_large.shape[0])),
                                   [0, 255, 255],
                                   markerType=cv2.MARKER_STAR, markerSize=40, thickness=5)
                    cv2.putText(pred_heatmap_large, f'{cop_error * 1000:.2f} mm', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
                    cv2.putText(pred_heatmap_large, f'{iou:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.putText(pred_heatmap_large, f'{frame_diff:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.putText(pred_heatmap_large, f'cop_w: {best_cop_w:.2f}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.putText(pred_heatmap_large, f'cop_k: {best_cop_k:.2f}', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.imwrite(os.path.join(out_pressure_heatmap_dir, f'{pname}_{pressure_fnum}_pred_cop_error_first.png'),
                                pred_heatmap_large)
                    print('[[PRED Pressure Heatmap Saved]]\n',
                          os.path.join(out_pressure_heatmap_dir, f'{pname}_{pressure_fnum}_pred_cop_error_first.png'))

                    # frame_idd first visualizations
                    pred_heatmap = pressure_evaluator.frame_diff_first.pred_heatmaps[-1]
                    pred_cop_relative = pressure_evaluator.frame_diff_first.pred_cops_relative[-1]
                    cop_error = pressure_evaluator.frame_diff_first.cop_errors[-1]
                    iou = pressure_evaluator.frame_diff_first.ious[-1]
                    frame_diff = pressure_evaluator.frame_diff_first.frame_diffs[-1]
                    best_cop_w = pressure_evaluator.frame_diff_first.best_cop_w[-1]
                    best_cop_k = pressure_evaluator.frame_diff_first.best_cop_k[-1]

                    pred_heatmap_large = cv2.resize(pred_heatmap, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
                    cv2.drawMarker(pred_heatmap_large, (
                        int(pred_cop_relative[0] / pressure_mat.mat_size_metric[0] * pred_heatmap_large.shape[1]),
                        int(pred_cop_relative[1] / pressure_mat.mat_size_metric[1] * pred_heatmap_large.shape[0])),
                                   [0, 255, 255],
                                   markerType=cv2.MARKER_STAR, markerSize=40, thickness=5)
                    cv2.putText(pred_heatmap_large, f'{cop_error * 1000:.2f} mm', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2)
                    cv2.putText(pred_heatmap_large, f'{iou:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.putText(pred_heatmap_large, f'{frame_diff:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.putText(pred_heatmap_large, f'cop_w: {best_cop_w:.2f}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.putText(pred_heatmap_large, f'cop_k: {best_cop_k:.2f}', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 0), 2)
                    cv2.imwrite(
                        os.path.join(out_pressure_heatmap_dir, f'{pname}_{pressure_fnum}_pred_frame_diff_first.png'),
                        pred_heatmap_large)
                    print('[[PRED Pressure Heatmap Saved]]\n',
                          os.path.join(out_pressure_heatmap_dir, f'{pname}_{pressure_fnum}_pred_frame_diff_first.png'))



                body_pose = smplx_params['body_pose'][vicon_fnum].squeeze().cpu()
                left_hand_pose = smplx_params['left_hand_pose'][vicon_fnum].squeeze().cpu()
                right_hand_pose = smplx_params['right_hand_pose'][vicon_fnum].squeeze().cpu()
                global_orient = smplx_params['global_orient'][vicon_fnum].squeeze().cpu()
                transl = transl.squeeze().cpu()

                # add to dict if keypoints were detected
                imgpaths_.append(img_path)
                imgnames_.append(img_fn)
                subject_ids_.append(subj_id)
                relpaths_.append(relpath)
                heights_.append(height)
                weights_.append(weight)
                genders_.append(gender)
                pressures_.append(gt_pressure)
                pressure_heatmaps_.append(gt_heatmap)
                pp_body_poses_.append(np.array(body_pose))
                pp_left_hand_poses_.append(np.array(left_hand_pose))
                pp_right_hand_poses_.append(np.array(right_hand_pose))
                pp_global_orients_.append(np.array(global_orient))
                pp_transls_.append(np.array(transl))

    print('Mean COP error (iou first): ', np.mean(pressure_evaluator.iou_first.cop_errors))
    print('Mean Pressure IOU (iou first): ', np.mean(pressure_evaluator.iou_first.ious))
    print('Mean Pressure Frame Diff (iou first): ', np.mean(pressure_evaluator.iou_first.frame_diffs))
    # print mode of cop_w and cop_k
    print('Mode of cop_w (iou first): ', stats.mode(pressure_evaluator.iou_first.best_cop_w))
    print('Mode of cop_k (iou first): ', stats.mode(pressure_evaluator.iou_first.best_cop_k))
    print('---------------------------------------------')
    print('Mean COP error (cop error first): ', np.mean(pressure_evaluator.cop_error_first.cop_errors))
    print('Mean Pressure IOU (cop error first): ', np.mean(pressure_evaluator.cop_error_first.ious))
    print('Mean Pressure Frame Diff (cop error first): ', np.mean(pressure_evaluator.cop_error_first.frame_diffs))
    # print mode of cop_w and cop_k
    print('Mode of cop_w (cop error first): ', stats.mode(pressure_evaluator.cop_error_first.best_cop_w))
    print('Mode of cop_k (cop error first): ', stats.mode(pressure_evaluator.cop_error_first.best_cop_k))
    print('---------------------------------------------')
    print('Mean COP error (frame diff first): ', np.mean(pressure_evaluator.frame_diff_first.cop_errors))
    print('Mean Pressure IOU (frame diff first): ', np.mean(pressure_evaluator.frame_diff_first.ious))
    print('Mean Pressure Frame Diff (frame diff first): ', np.mean(pressure_evaluator.frame_diff_first.frame_diffs))
    # print mode of cop_w and cop_k
    print('Mode of cop_w (frame diff first): ', stats.mode(pressure_evaluator.frame_diff_first.best_cop_w))
    print('Mode of cop_k (frame diff first): ', stats.mode(pressure_evaluator.frame_diff_first.best_cop_k))
    print('---------------------------------------------')
    print(f'Folders with no images: \n {no_images_found}')
    print(f'Folders with no presented poses: \n {no_pp_found}')

    # save as db file
    outfile = osp.join('../data', 'dbs_biomech', 'newmoyo_gt_first_day_vicon_220923.npz')
    os.makedirs(osp.dirname(outfile), exist_ok=True)
    np.savez(outfile,
             imgpath=imgpaths_,
             subject_id=subject_ids_,
             imgname=imgnames_,
             relpath=relpaths_,
             height=heights_,
             weight=weights_,
             gender=genders_,
             pressure=pressures_,
             pressure_heatmap=pressure_heatmaps_,
             body_pose=pp_body_poses_,
             left_hand_pose=pp_left_hand_poses_,
             right_hand_pose=pp_right_hand_poses_,
             global_orient=pp_global_orients_,
             transl=pp_transls_
             )


if __name__ == "__main__":
    """
    yoga-82 dir /ps/scratch/ps_shared/4Paul/Yoga-82
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', required=True,
                        help='folder containing yoga 82 images')
    # parser.add_argument('--openpose_folder', required=True,
    #                     help='folder containing yoga 82 keypoints')
    parser.add_argument('--pressure_xml_folder', required=True,
                        help='folder containing GT pressure XMLs')
    parser.add_argument('--pressure_csv_folder', required=True,
                        help='folder containing GT pressure CSVs')
    parser.add_argument('--pp_folder', required=True,
                        help='presented poses folder')
    # parser.add_argument('--essentials_folder', required=True,
    #                     help='geodesics, ... essentials for smplify-xmc')
    parser.add_argument('--model_folder', required=False, default='/ps/project/common/smplifyx/models/',
                        help='path to SMPL/SMPL-X model folder')
    parser.add_argument('--save_outputs', action='store_true', default=False,
                        help='if enabled, saves outputs to disk, but slows down eval')

    args = parser.parse_args()
    yoga82_extra(args.img_folder, args.pp_folder, args.pressure_xml_folder, args.pressure_csv_folder, args.model_folder, args.save_outputs)