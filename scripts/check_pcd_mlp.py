from pathlib import Path
from random import randint
from typing import List

import h5py
import numpy as np
import torch
from pycarus.geometry.pcd import get_o3d_pcd_from_tensor, sample_pcds_from_udfs
from pycarus.learning.models.siren import SIREN
from torch import Tensor

from utils.var import unflatten_mlp_params

import open3d as o3d  # isort: skip


dset_root = Path("/path/to/pcd/inrs")
mlps_paths = sorted(list(dset_root.glob("*.h5")), key=lambda p: int(p.stem))

while True:
    idx = randint(0, len(mlps_paths) - 1)
    print("Index:", idx)

    with h5py.File(mlps_paths[idx], "r") as f:
        pcd = torch.from_numpy(np.array(f.get("pcd")))
        params = torch.from_numpy(np.array(f.get("params")))

    gt_pcd = pcd.cuda()

    mlp = SIREN(3, 512, 4, 1)
    mlp.load_state_dict(unflatten_mlp_params(params, mlp.state_dict()))
    mlp = mlp.cuda()

    def udfs_func(coords: Tensor, indices: List[int]) -> Tensor:
        pred = torch.sigmoid(mlp(coords)[0])
        pred = pred.squeeze(-1)
        pred = 1 - pred
        pred *= 0.1
        return pred

    pred_pcd = sample_pcds_from_udfs(udfs_func, 1, 4096, (-1, 1), 0.05, 0.02, 8192, 5)[0]

    gt_o3d = get_o3d_pcd_from_tensor(gt_pcd)
    pred_pcd_o3d = get_o3d_pcd_from_tensor(pred_pcd).translate((2, 0, 0))

    o3d.visualization.draw_geometries([gt_o3d, pred_pcd_o3d])

    # with h5py.File(mlps_paths[idx], "r") as f:
    #     incomplete = torch.from_numpy(np.array(f.get("incomplete")))
    #     params_incomplete = torch.from_numpy(np.array(f.get("params_incomplete")))
    #     complete = torch.from_numpy(np.array(f.get("complete")))
    #     params_complete = torch.from_numpy(np.array(f.get("params_complete")))

    # mlp_inc = SIREN(3, 512, 4, 1)
    # mlp_inc.load_state_dict(unflatten_mlp_params(params_incomplete, mlp_inc.state_dict()))
    # mlp_inc = mlp_inc.cuda()

    # mlp_compl = SIREN(3, 512, 4, 1)
    # mlp_compl.load_state_dict(unflatten_mlp_params(params_complete, mlp_compl.state_dict()))
    # mlp_compl = mlp_compl.cuda()

    # def udfs_func_inc(coords: Tensor, indices: List[int]) -> Tensor:
    #     pred = torch.sigmoid(mlp_inc(coords)[0])
    #     pred = pred.squeeze(-1)
    #     pred = 1 - pred
    #     pred *= 0.1
    #     return pred

    # pred_inc = sample_pcds_from_udfs(udfs_func_inc, 1, 4096, (-1, 1), 0.05, 0.02, 8192, 5)[0]

    # def udfs_func_compl(coords: Tensor) -> Tensor:
    #     pred = torch.sigmoid(mlp_compl(coords)[0])
    #     pred = pred.squeeze(-1)
    #     pred = 1 - pred
    #     pred *= 0.1
    #     return pred

    # pred_compl = sample_pcds_from_udfs(udfs_func_compl, 1, 4096, (-1, 1), 0.05, 0.02, 8192, 5)[0]

    # inc_o3d = get_o3d_pcd_from_tensor(incomplete)
    # pred_inc_o3d = get_o3d_pcd_from_tensor(pred_inc).translate((2, 0, 0))
    # compl_o3d = get_o3d_pcd_from_tensor(complete).translate((4, 0, 0))
    # pred_compl_o3d = get_o3d_pcd_from_tensor(pred_compl).translate((6, 0, 0))

    # o3d.visualization.draw_geometries([inc_o3d, pred_inc_o3d, compl_o3d, pred_compl_o3d])