from pathlib import Path
from random import randint

import h5py
import numpy as np
import torch
from pycarus.geometry.mesh import get_o3d_mesh_from_tensors, marching_cubes
from pycarus.learning.models.siren import SIREN
from torch import Tensor

from utils.var import unflatten_mlp_params

import open3d as o3d  # isort: skip


dset_root = Path("/path/to/inrs")
mlps_paths = sorted(list(dset_root.glob("*.h5")), key=lambda p: int(p.stem))

while True:
    idx = randint(0, len(mlps_paths) - 1)
    print("Index:", idx)

    # with h5py.File(mlps_paths[idx], "r") as f:
    #     vertices = torch.from_numpy(np.array(f.get("vertices")))
    #     num_vertices = torch.from_numpy(np.array(f.get("num_vertices")))
    #     triangles = torch.from_numpy(np.array(f.get("triangles")))
    #     num_triangles = torch.from_numpy(np.array(f.get("num_triangles")))
    #     params = torch.from_numpy(np.array(f.get("params")))

    # gt_vertices = vertices[:num_vertices]
    # gt_triangles = triangles[:num_triangles]

    # mlp = SIREN(3, 512, 4, 1)
    # mlp.load_state_dict(unflatten_mlp_params(params, mlp.state_dict()))
    # mlp = mlp.cuda()

    # def levelset_func(coords: Tensor) -> Tensor:
    #     pred = mlp(coords)[0].squeeze(-1)
    #     pred = torch.sigmoid(pred)
    #     pred *= 0.2
    #     pred -= 0.1
    #     return pred

    # pred_v, pred_t = marching_cubes(levelset_func, coords_range=(-1, 1), resolution=128)

    # gt_mesh_o3d = get_o3d_mesh_from_tensors(gt_vertices, gt_triangles)
    # pred_mesh_o3d = get_o3d_mesh_from_tensors(pred_v, pred_t).translate((2, 0, 0))

    # o3d.visualization.draw_geometries([gt_mesh_o3d, pred_mesh_o3d])

    with h5py.File(mlps_paths[idx], "r") as f:
        params = torch.from_numpy(np.array(f.get("params")))
        cls = str(np.array(f.get("class"), dtype=str))

    print("class:", cls)

    mlp = SIREN(3, 512, 4, 1)
    mlp.load_state_dict(unflatten_mlp_params(params, mlp.state_dict()))
    mlp = mlp.cuda()

    def levelset_func(coords: Tensor) -> Tensor:
        pred = mlp(coords)[0].squeeze(-1)
        pred = torch.sigmoid(pred)
        pred *= 0.2
        pred -= 0.1
        return pred

    pred_v, pred_t = marching_cubes(
        levelset_func,
        coords_range=(-1, 1),
        resolution=128,
        level=0.01,
    )

    pred_mesh_o3d = get_o3d_mesh_from_tensors(pred_v, pred_t)

    o3d.visualization.draw_geometries([pred_mesh_o3d])