## [modelnet40](modelnet40) 

Subset of the validation set of ModelNet40, a dataset of point clouds of 3D shapes. This subset contains one shape per class, i.e. 40 shapes.

Each .h5 file contains:
* `class_id`: integer class identifier of the shape.

* `params`: flattened weights and biases of the neural field (an MLP) trained on the point cloud, i.e. a 1D array of 790529 floats. This is because the MLP has dimensions `3→512→512→512→512→1` (with sine activation function in between), resulting in
    ```
    (3*512+512) + (512*512+512)*3 + (512*1+1) = 790529
    ```
    weights and biases.

* `pcd`: the point cloud, i.e. an array of of shape `(2048, 3)`(2034 points w/ 3 coordinates for each point).

To handle this dataset, refer to the [inr2vec offical repository](https://github.com/CVLAB-Unibo/inr2vec).

## [manifold40](manifold40)

Subset of the validation set of Manifold40, a dataset of meshes of 3D shapes. This subset contains one shape per class, i.e. 40 shapes.

The content of .h5 files is analogous to ModelNet40.

## [shapenet_render](shapenet_render)

Subset of the validation set of ShapenetRender, a dataset of NeRFs of 3D objects. This subset contains one object per class, i.e. 13 objects.

Each subdirectory has the following structure:
```
class_id
 └──nerf_id
     ├──train
     │   └──*.png
     ├──grid.pth
     └──nerf_weights.pth
```
where
* `class_id` is a numeric identifier of the object class.
* `nerf_id` is an alphanumeric identifier of the object itself.
* `train` contains the object views (i.e. images) the NeRF was trained on.
* `grid.pth` are the parameters of the occupancy grid, a data structure used to accelerate NeRF training (see Appendix E of the [Instant-NGP paper](https://arxiv.org/pdf/2201.05989) for details).
* `nerf_weights.pth` are the flattened weights of the NeRF, i.e. an MLP. This MLP has shape `3→144→64→64→64→16` (with ReLU activation function in between and no biases), where `3→144` is caused by a frequency encoding of the input and the `16` output dimension is the concatenation of 4 output values (RGBσ) and 12 values of padding (for memory efficiency reasons).

To handle this dataset, refer to the [nf2vec offical repository](https://github.com/CVLAB-Unibo/nf2vec).