import numpy as np
from segmentation.deploy_model import merge_point_cloud, parse_input
import sys
from ruamel.yaml import YAML

def voxel_downsample(point_cloud: np.ndarray, voxel_size: float):
    # extract the features
    xyz = point_cloud[:, :3]
    intensity = point_cloud[:, 3]

    # compute grid boundaries
    max_coords = xyz.max(axis=0)
    min_coords = xyz.min(axis=0)
    grid_size = np.ceil((max_coords - min_coords) / voxel_size).astype(np.int64)  # x_size / y_size / z_size

    if voxel_size > 0:
        # compute the point to voxel indices
        centered_xyz = xyz - min_coords
        point_indices = np.floor(centered_xyz / voxel_size).astype(np.int64)

        # prepare indices and conversion table
        multi_index = (point_indices[:, 0], point_indices[:, 1], point_indices[:, 2])
        lin_point_indices = np.ravel_multi_index(multi_index, grid_size)
        unique_lin_indices, inverse_indices = np.unique(lin_point_indices, return_inverse=True)
        temp_indices = np.arange(unique_lin_indices.shape[0])

        indices = temp_indices[inverse_indices]

        # compute the counts of each voxel idx in pointcloud
        index_counts = np.zeros_like(unique_lin_indices)
        np.add.at(index_counts, indices, 1)

        # compute the centroids
        buffer_x = np.zeros_like(unique_lin_indices).astype(np.float64)
        buffer_y = buffer_x.copy()
        buffer_z = buffer_x.copy()
        np.add.at(buffer_x, indices, xyz[:, 0])
        np.add.at(buffer_y, indices, xyz[:, 1])
        np.add.at(buffer_z, indices, xyz[:, 2])
        centroids = np.concatenate([
            buffer_x.reshape(-1, 1), buffer_y.reshape(-1, 1), buffer_z.reshape(-1, 1)
        ], axis=1)
        centroids /= index_counts.reshape(-1, 1)

        # compute intensity average
        intensity_buffer = np.zeros_like(unique_lin_indices).astype(np.float64)
        np.add.at(intensity_buffer, indices, intensity)
        intensity_average = intensity_buffer / index_counts

        # assemble the pointcloud
        downsampled_point_cloud = np.concatenate([
            centroids, intensity_average.reshape(-1, 1)
        ], axis=1)
    else:
        downsampled_point_cloud = point_cloud[:, :4]


    return downsampled_point_cloud.astype(np.float64)


if __name__ == '__main__':
    pcd_file = sys.argv[1]

    pipeline_config_path = './common/pipeline/config.yaml'
    yaml = YAML()
    yaml.default_flow_style = False
    with open(pipeline_config_path, "r") as f:
        config = yaml.load(f)
    print('config loaded')
    scan_list, odom_list = parse_input(pcd_file)
    merged_pc = merge_point_cloud(scan_list, odom_list, config)
    merged_down_pc = voxel_downsample(merged_pc, 0.15)
    print(f'downsampled {merged_down_pc.shape}')
    np.savez('./merge_test.npz',data=merged_pc, data_down=merged_down_pc)

