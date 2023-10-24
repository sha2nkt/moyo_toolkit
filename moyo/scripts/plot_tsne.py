# Usage python plot_tsne.py --geo

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
from tqdm import tqdm
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.transforms import so3_relative_angle
# from smplx import MANO
import sys
from amass_loader import AMASSDataset
from moyo65_loader import MOYO65Dataset
from datasets.base_dataset import BaseDataset

sys.path = ["."] + sys.path
# import src.utils.analysis_utils as ana_utils
# import src.utils.dataloader as loader

sys.path = ["."] + sys.path

def reform_data_geo(poses):
    batch_size = poses.shape[0]
    num_rots = 21
    poses = torch.FloatTensor(poses)
    poses = poses.view(batch_size, num_rots, 3)
    poses = axis_angle_to_matrix(poses)
    zero_poses = torch.zeros(batch_size, num_rots, 3)
    zero_poses = axis_angle_to_matrix(zero_poses)
    geo_dist = so3_relative_angle(
        zero_poses.view(-1, 3, 3), poses.view(-1, 3, 3)
    )  # .view( batch_size, num_rots, 3, 3)
    geo_dist = geo_dist.view(batch_size, num_rots)
    return geo_dist


def plot_tsne(args):
    n = args.n_samples
    # 10% ARCTIC; 10% HO3D

    # Load Moyo65 poses
    moyo65_db = MOYO65Dataset()
    moyo65_poses = np.concatenate(moyo65_db.db['body_pose']) # smplx only contains 21 body joints + 1 root

    # Load AMASS poses
    fraction = 0.001
    amass_db = AMASSDataset(seqlen=10) # seqlen doesn't matter
    amass_poses = amass_db.db['theta'][:, 3:66] # only take the pose parameters, ignore last 2 finger joints. last 10 are shape params
    # randomly sample fraction of the poses
    amass_poses = amass_poses[np.random.choice(amass_poses.shape[0], int(fraction * amass_poses.shape[0]), replace=False), :]

    # Load H36M poses
    fraction = 0.001
    h36m_db = BaseDataset(options=None, dataset='h36m')
    h36m_poses = h36m_db.data['pose'][:, 3:66] # only take the pose parameters, ignore last 2 finger joints
    h36m_poses = h36m_poses[
                  np.random.choice(h36m_poses.shape[0], int(fraction * h36m_poses.shape[0]), replace=False), :]

    # Load MPI-INF-3DHP poses
    fraction = 0.01
    mpi_db = BaseDataset(options=None, dataset='mpi-inf-3dhp')
    mpi_poses = mpi_db.data['pose'][:, 3:66]  # only take the pose parameters, ignore last 2 finger joints
    mpi_poses = mpi_poses[
                 np.random.choice(mpi_poses.shape[0], int(fraction * mpi_poses.shape[0]), replace=False), :]

    # Load AGORA poses
    fraction = 0.01
    agora_poses = []
    for i in range(1, 6):
        agora_db = BaseDataset(options=None, dataset=f'agora{i}')
        agora_poses.append(agora_db.data['pose'][:, 3:66])  # only take the pose parameters, ignore last 2 finger joints
    agora_poses = np.concatenate(agora_poses)
    agora_poses = agora_poses[
                    np.random.choice(agora_poses.shape[0], int(fraction * agora_poses.shape[0]), replace=False), :]
    if args.geodesic:
        print('*** Using geodesic angle ***')
        moyo65_poses = reform_data_geo(moyo65_poses)
        amass_poses = reform_data_geo(amass_poses)
        h36m_poses = reform_data_geo(h36m_poses)
        mpi_poses = reform_data_geo(mpi_poses)
        agora_poses = reform_data_geo(agora_poses)


    # aohmr = ana_utils.random_sample(loader.get_aohmr_data(seq=False), 10 * n)
    # dexycb = ana_utils.random_sample(loader.get_dexycb_data(seq=False), n)
    # ho3d = ana_utils.random_sample(loader.get_ho3d_data_alex(seq=False), n)
    # hoi4d = ana_utils.random_sample(loader.get_hoi4d_data(), n)
    #
    # aohmr = ana_utils.reform_data(aohmr)
    # dexycb = ana_utils.reform_data(dexycb)
    # ho3d = ana_utils.reform_data(ho3d)
    # hoi4d = ana_utils.reform_data(hoi4d)

    # num_examples = [len(aohmr), len(dexycb), len(ho3d), len(hoi4d)]
    # print(f"ARCTIC: {len(aohmr)}")
    # print(f"DexYCB: {len(dexycb)}")
    # print(f"HO3Dv2: {len(ho3d)}")
    # print(f"HOI4D: {len(hoi4d)}")
    #
    # X = np.concatenate([aohmr, dexycb, ho3d, hoi4d])

    num_examples = [len(moyo65_poses), len(amass_poses), len(h36m_poses), len(mpi_poses), len(agora_poses)]
    X = np.concatenate([moyo65_poses, amass_poses, h36m_poses, mpi_poses, agora_poses])

    # pca = PCA(n_components=10)
    # X = pca.fit_transform(X)

    init = args.init
    cluster_algo = TSNE(
        n_components=2,
        init=init,
        learning_rate="auto",
        metric=args.metric,
        perplexity=50.0,
        # n_iter=500,
        # n_iter_without_progress=150,
        n_jobs=16,
        # random_state=0,
        verbose=True,
    )
    X_emb = cluster_algo.fit_transform(X)
    X_emb = torch.FloatTensor(X_emb)
    X_emb_moyo65, X_emb_amass, X_emb_h36m, X_emb_mpi, X_emb_agora = torch.split(
        X_emb,
        num_examples,
        dim=0,
    )

    X_emb_moyo65 = X_emb_moyo65.numpy()
    X_emb_amass = X_emb_amass.numpy()
    X_emb_h36m = X_emb_h36m.numpy()
    X_emb_mpi = X_emb_mpi.numpy()
    X_emb_agora = X_emb_agora.numpy()

    # X_emb_ho3d = X_emb_ho3d.numpy()
    # X_emb_hoi4d = X_emb_hoi4d.numpy()
    alpha = None
    s = 10.0

    plt.figure(figsize=(7, 7))

    plt.scatter(
        X_emb_h36m[:, 0],
        X_emb_h36m[:, 1],
        marker="D",
        c="#FF8E00",
        label="Human 3.6M",
        alpha=alpha,
        s=s,
    )

    plt.scatter(
        X_emb_amass[:, 0],
        X_emb_amass[:, 1],
        marker="^",
        c="#F0D31D",
        label="AMASS",
        alpha=alpha,
        s=s,
    )

    plt.scatter(
        X_emb_mpi[:, 0],
        X_emb_mpi[:, 1],
        marker="s",
        c="#06FF00",
        label="MPI-INF-3DHP",
        alpha=alpha,
        s=s,
    )


    plt.scatter(
        X_emb_agora[:, 0],
        X_emb_agora[:, 1],
        marker="*",
        c="#FF1700",
        label="AGORA",
        alpha=alpha,
        s=s,
    )

    plt.scatter(
        X_emb_moyo65[:, 0],
        X_emb_moyo65[:, 1],
        c="#4D37F0",
        label="MOYO65",
        # alpha=0.3,
        alpha=alpha,
        s=s,
    )

    # plt.scatter(
    #     X_emb_hoi4d[:, 0],
    #     X_emb_hoi4d[:, 1],
    #     c="#F02A11",
    #     label="HOI4D",
    #     alpha=alpha,
    #     s=s,
    # )
    #
    # plt.scatter(
    #     X_emb_aohmr[:, 0],
    #     X_emb_aohmr[:, 1],
    #     # marker="x",
    #     c="k",
    #     label="ARCTIC",
    #     alpha=0.3,
    #     # alpha=alpha,
    #     s=s,
    # )
    plt.legend()
    plt.axis("off")

    # folder = f"tsne_plots/n={n}/init={init}/metric={args.metric}"
    folder = f"tsne_plots/"
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "x_emb.npy"), X_emb)
    plt.savefig(
        os.path.join(folder, "tsne.png"), bbox_inches="tight", tight_layout=True
    )
    plt.savefig(os.path.join(folder, "tsne.pdf"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_samples", type=int, default=1000)
    parser.add_argument("-i", "--init", type=str, default="pca")
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("-s", "--seq", action="store_true")
    parser.add_argument("-m", "--mode", type=str, default="consecutive")
    parser.add_argument("-geo", "--geodesic", action="store_true")
    args = parser.parse_args()

    plot_tsne(args)
