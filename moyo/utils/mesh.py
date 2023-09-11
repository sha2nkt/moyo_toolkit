import torch
import trimesh

from moyo.utils.constants import MOYO_V_TEMPLATE


def smplx_breakdown(bdata, select_fnum, device):
    num_frames = len(bdata['trans'])

    bdata['poses'] = bdata['fullpose']

    global_orient = torch.from_numpy(bdata['poses'][[select_fnum], :3]).float().to(device)
    body_pose = torch.from_numpy(bdata['poses'][[select_fnum], 3:66]).float().to(device)
    jaw_pose = torch.from_numpy(bdata['poses'][[select_fnum], 66:69]).float().to(device)
    leye_pose = torch.from_numpy(bdata['poses'][[select_fnum], 69:72]).float().to(device)
    reye_pose = torch.from_numpy(bdata['poses'][[select_fnum], 72:75]).float().to(device)
    left_hand_pose = torch.from_numpy(bdata['poses'][[select_fnum], 75:120]).float().to(device)
    right_hand_pose = torch.from_numpy(bdata['poses'][[select_fnum], 120:]).float().to(device)

    v_template = trimesh.load(MOYO_V_TEMPLATE, process=False)

    body_params = {'global_orient': global_orient, 'body_pose': body_pose,
                  'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                  'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                   'v_template': torch.Tensor(np.repeat(v_template.v[None], repeats=1, axis=0)).to(device),}
    return body_params