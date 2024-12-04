import imageio
import math
import numpy as np
import random
import sys
import torch
import torch.nn.functional as F
import tqdm

from pathlib import Path

from datasets.nerf_synthetic import SubjectLoader
from datasets.utils import Rays
from nerfacc import ContractionType, OccupancyGrid
from radiance_fields.ngp_nerf2vec import NGPradianceField
from utils import render_image


def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, torch.cos(phi), -torch.sin(phi), 0],
        [0, torch.sin(phi), torch.cos(phi), 0],
        [0, 0, 0, 1]
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [torch.cos(theta), 0, -torch.sin(theta), 0],
        [0, 1, 0, 0],
        [torch.sin(theta), 0, torch.cos(theta), 0],
        [0, 0, 0, 1]
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi
    and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.from_numpy(np.array([
        [-1, 0, 0, 0], 
        [ 0, 0, 1, 0], 
        [ 0, 1, 0, 0], 
        [ 0, 0, 0, 1]
    ], dtype=np.float32)) @ c2w  
    
    return c2w


def create_video(
    width,
    height,
    device,
    focal,
    radiance_field,
    occupancy_grid,
    scene_aabb,
    near_plane,
    far_plane,
    render_step_size,
    render_bkgd,
    cone_angle,
    alpha_thre,
    test_chunk_size,
    path,
    OPENGL_CAMERA=True):

    rgb_frames = []

    # Iterate over different theta value and generate scenes.
    max_images = 20
    array = np.linspace(-30.0, 30.0, max_images//2, endpoint=False)
    array = np.append(array, np.linspace(30.0, -30.0, max_images//2, endpoint=False))
    
    for index, theta in tqdm.tqdm(enumerate(np.linspace(0.0, 360.0, max_images, endpoint=False))):

        # Get the camera to world matrix.
        c2w = pose_spherical(torch.tensor(theta), torch.tensor(array[index]), torch.tensor(1.0))
        c2w = c2w.to(device)

        x, y = torch.meshgrid(
            torch.arange(width, device=device),
            torch.arange(height, device=device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        K = torch.tensor([
            [focal, 0, width / 2.0],
            [0, focal, height / 2.0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)  # (3, 3)

        camera_dirs = F.pad(
            torch.stack([
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5) / K[1, 1] * (-1.0 if OPENGL_CAMERA else 1.0),
            ], dim=-1),
            (0, 1),
            value=(-1.0 if OPENGL_CAMERA else 1.0)
        )  # [num_rays, 3]
        camera_dirs.to(device)

        directions = (camera_dirs[:, None, :] * c2w[:3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

        origins = torch.reshape(origins, (height, width, 3))
        viewdirs = torch.reshape(viewdirs, (height, width, 3))
        
        rays = Rays(origins=origins, viewdirs=viewdirs)
        # render
        rgb, acc, depth, n_rendering_samples = render_image(
            radiance_field=radiance_field,
            occupancy_grid=occupancy_grid,
            rays=rays,
            scene_aabb=scene_aabb,
            # rendering options
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )

        numpy_image = (rgb.cpu().numpy() * 255).astype(np.uint8)
        rgb_frames.append(numpy_image)

    imageio.mimwrite(path, rgb_frames, fps=30, quality=8, macro_block_size=None)
    

if __name__ == "__main__":
    subject_id = "02691156/1a9b552befd6306cc8f2d5fe7449af61"

    n_hidden_layers = 3
    n_neurons = 64
    coordinate_encoding = "Frequency"
    encoding_size = 24
    mlp = "FullyFusedMLP"
    activation = "ReLU"
    
    data_root = "shapenet_render"
    video_path = f"out.mp4"
    device = "cuda"

    aabb = [-0.7,-0.7,-0.7,0.7,0.7,0.7]
    scene_aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    alpha_thre = 0.0
    cone_angle = 0.0
    contraction_type = ContractionType.AABB
    far_plane = None
    grid_resolution = 128
    near_plane = None
    render_n_samples = 1024
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / render_n_samples
    ).item()
    target_sample_batch_size = 1 << 18
    test_chunk_size = 8192
    train_dataset_kwargs = {}
    unbounded = False

    train_dataset = SubjectLoader(
        subject_id=subject_id,
        root_fp=data_root,
        split="train",
        num_rays=target_sample_batch_size // render_n_samples,
        **train_dataset_kwargs,
    )

    radiance_field = NGPradianceField(
        aabb=aabb,
        unbounded=unbounded,
        encoding=coordinate_encoding,
        mlp=mlp,
        activation=activation,
        n_hidden_layers=n_hidden_layers,
        n_neurons=n_neurons,
        encoding_size=encoding_size
    ).to(device)
    radiance_field.load_state_dict(torch.load(data_root / subject_id / "nerf_weights.pth"))
    radiance_field = radiance_field.eval()

    occupancy_grid = OccupancyGrid(
        roi_aabb=aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)
    occupancy_grid.load_state_dict(torch.load(data_root / subject_id / "grid.pth"))
    occupancy_grid = occupancy_grid.eval()

    with torch.no_grad():
        create_video(
            720, 
            480, 
            device, 
            train_dataset.focal, 
            radiance_field, 
            occupancy_grid, 
            scene_aabb,
            near_plane, 
            far_plane, 
            render_step_size,
            render_bkgd= torch.zeros(3, device=device),
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
            # test options
            test_chunk_size=test_chunk_size,
            path=video_path
        )
