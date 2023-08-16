import torch
import numpy as np
# from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN


def create_line_3d(center: torch.Tensor, dim_ratio: list = None, n: int = 100) -> torch.Tensor:
    """
    Create a dashed line in 3D space. The line is composed of n points that are randomly generated around the center
    with noise that is defined in each dimension by the dim_ratio.
    :param center: center of the line
    :param dim_ratio: ratio of the noise in each dimension
    :param n: number of points in the line
    :return: point cloud data
    """
    if dim_ratio is None:
        dim_ratio = torch.tensor([1.2, 0.4, 0.05])
    else:
        dim_ratio = torch.tensor(dim_ratio)
    df = torch.rand(n, 3) * 2 - 1
    df = df * dim_ratio + center
    return df


def creat_continuous_line(length: float = 2.0, width: float = 0.2, height: float = 0.02, center: list = None,
                          min_points: int = 300, max_points: int = 400) -> torch.tensor:
    """
    Create a continuous line in 3D space. The line lies on the x-axis. The points are generated randomly along the
    specified dimensions.
    :param length: length of the line
    :param width: width of the line
    :param height: height of the line
    :param center: center of the line
    :param min_points: minimum number of points in the line
    :param max_points: maximum number of points in the line
    :return: point cloud data (format: [num_points, 3] = [x, y, z])
    """
    # Set default center if not provided (prevent from using mutable object as default argument)
    if center is None:
        center = torch.tensor([0, 0, 0])
    else:
        center = torch.tensor(center)

    # Generate points
    n = np.random.randint(min_points, max_points)

    # Create line
    line = create_line_3d(center, dim_ratio=[length, width, height], n=n)

    return line


def generate_dashed_lines(num_lines: int, length: float = 0.6, width: float = 0.2, height: float = 0.02,
                          center: list = None, min_points: int = 300, max_points: int = 400) -> torch.tensor:
    """
    Generate dashed lines in 3D space. The lines lay on the x-axis and are separated by the length of the line. The
    points are generated randomly in the along the specified dimensions.
    :param num_lines: number of lines
    :param length: length of the line
    :param width: width of the line
    :param height: height of the line
    :param center: center of all lines
    :param min_points: minimum number of points in each line
    :param max_points: maximum number of points in each line
    :return: point cloud data (format: [num_points, 4] = [x, y, z, label])
    """
    # Set default center if not provided (prevent from using mutable object as default argument)
    if center is None:
        center = [0, 0, 0]

    # Generate centers of the lines (the space between the centers should be twice the length of the line, therefore
    # the length between two lines is one length of the line)
    centers = torch.zeros((num_lines, 3))
    centers[:, 0] = torch.linspace(-center[0] - (2 * num_lines - 2) * length, center[0] + (2 * num_lines - 2) * length,
                                   num_lines)

    pc = []
    for i in range(num_lines):
        # Number of points in each line (random number <300, 400>)
        n = torch.randint(min_points, max_points, (1, 1)).item()

        # Get center of this specific line
        center = centers[i]

        # Generate points for this line
        line_points = create_line_3d(center, dim_ratio=[length, width, height], n=n)

        # Add label to each line and append to the point cloud
        line_points = torch.cat([line_points, torch.ones((n, 1)) * i], dim=1)
        pc.append(line_points)

    # Concatenate all lines and make them torch tensor
    pc = torch.cat(pc, dim=0)
    return pc


def generate_instance_label_mask(pc: torch.Tensor, eps: float = 0.5, min_samples: int = 10, min_cluster_size: int = 10,
                                 method: str = "DBSCAN", device: torch.device = torch.cpu) -> torch.Tensor:
    """
    Generate label mask for instance segmentation via DBSCAN or HDBSCAN.
    :param pc: point cloud
    :param eps: The maximum distance between two samples for one to be considered as in the
                neighborhood of the other. (DBSCAN parameter)
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be
                        considered as a core point. This includes the point itself. (DBSCAN/HDBSCAN parameter)
    :param min_cluster_size: The minimum number of samples in a cluster for the cluster to be considered
                                significant. (HDBSCAN parameter)
    :param method: method to use for clustering (DBSCAN or HDBSCAN)
    :param device: device to use
    :return: ground truth label mask (format: [batch_size, num_points, max_instances])
    """
    # Generate labels
    if method == "DBSCAN":
        res = DBSCAN(eps=eps, min_samples=min_samples).fit(pc.cpu())
    elif method == "HDBSCAN":
        # TODO: On server, HDBSCAN does not work. Check if it works on local machine.
        pass
        # res = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(pc)
        # res.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        # res.condensed_tree_.plot()
    else:
        raise ValueError(f"Unknown method: {method}. Only DBSCAN and HDBSCAN are supported.")

    # Extract labels
    labels_pred = torch.tensor(res.labels_, device=device)

    # When a label is -1, it means that the point is an outlier and does not belong to any cluster.
    # TODO: handle outliers in a better way
    num_outliers = torch.sum(labels_pred == -1)
    percentage_outliers = num_outliers / labels_pred.shape[0]
    print(f"Number of outliers: {num_outliers} ({percentage_outliers * 100}%)")
    # Spread outliers over all instances
    # labels_pred[labels_pred == -1] = torch.randint(0, torch.max(labels_pred) + 1, (num_outliers,))
    labels_pred[labels_pred == -1] = torch.max(labels_pred) + 1

    # Create a mask -> format: [batch_size, num_points, max_instances] (one-hot encoding)
    if len(pc.shape) == 2:
        # If we have a single point cloud
        batch_size = 1
        num_points = pc.shape[0]
    else:
        # If we have multiple point clouds
        batch_size = pc.shape[0]
        num_points = pc.shape[1]

    # Initialize mask
    num_instances = torch.max(labels_pred) + 1
    mask = torch.zeros(batch_size, num_points, num_instances, device=device)

    # Create indices for batch dimension indices [shape: (batch_size, 1)] and point cloud
    # dimension indices [shape: (num_points,)]
    indices = torch.arange(batch_size).unsqueeze(1)
    mask_indices = torch.arange(num_points)

    # Fill the mask
    mask[indices, mask_indices, labels_pred[mask_indices]] = 1
    return mask


def generate_instance_labels(pc: torch.Tensor, eps: float = 0.5, min_samples: int = 10, min_cluster_size: int = 10,
                             method: str = "DBSCAN", device: torch.device = torch.cpu) -> torch.Tensor:
    """
    Generate labels for instance segmentation via DBSCAN or HDBSCAN.
    :param pc: point cloud (format: [num_points, 3])
    :param eps: The maximum distance between two samples for one to be considered as in the
                neighborhood of the other. (DBSCAN parameter)
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be
                        considered as a core point. This includes the point itself. (DBSCAN/HDBSCAN parameter)
    :param min_cluster_size: The minimum number of samples in a cluster for the cluster to be considered
                                significant. (HDBSCAN parameter)
    :param method: method to use for clustering (DBSCAN or HDBSCAN)
    :param device: device to use
    :return: ground truth labels (format: [num_points])
    """
    # Generate labels
    if method == "DBSCAN":
        res = DBSCAN(eps=eps, min_samples=min_samples).fit(pc)
    elif method == "HDBSCAN":
        pass
        # res = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(pc)
    else:
        raise ValueError(f"Unknown method: {method}. Only DBSCAN and HDBSCAN are supported.")
    return torch.tensor(res.labels_, device=device)
