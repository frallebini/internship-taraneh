from pathlib import Path
from random import randint

import h5py
import numpy as np
import torch
from pycarus.geometry.pcd import get_o3d_pcd_from_tensor, voxelize_pcd
from pycarus.learning.models.siren import SIREN

from utils.var import unflatten_mlp_params

import open3d as o3d  # isort: skip


dset_root = Path("/path/to/vox/inrs")
mlps_paths = sorted(list(dset_root.glob("*.h5")), key=lambda p: int(p.stem))

while True:
    idx = randint(0, len(mlps_paths) - 1)
    print("Index:", idx)

    with h5py.File(mlps_paths[idx], "r") as f:
        pcd = torch.from_numpy(np.array(f.get("pcd")))
        params = torch.from_numpy(np.array(f.get("params")))

    mlp = SIREN(3, 512, 4, 1)
    mlp.load_state_dict(unflatten_mlp_params(params, mlp.state_dict()))
    mlp = mlp.cuda()

    vgrid, centroids = voxelize_pcd(pcd, 64, -1, 1)

    vgrid_gt = centroids[vgrid == 1]
    vgrid_gt_o3d = get_o3d_pcd_from_tensor(vgrid_gt)

    vgrid_pred = torch.sigmoid(mlp(centroids.cuda())[0].squeeze(-1))
    vgrid_pred = centroids[vgrid_pred > 0.4]
    vgrid_pred_o3d = get_o3d_pcd_from_tensor(vgrid_pred).translate((2, 0, 0))

    o3d.visualization.draw_geometries([vgrid_gt_o3d, vgrid_pred_o3d])