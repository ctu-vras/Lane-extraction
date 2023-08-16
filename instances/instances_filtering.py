import torch
import matplotlib.pyplot as plt
import heapq
from timeit import default_timer as timer


def filter_pc(pc: torch.Tensor, config: dict = None) -> torch.Tensor:
    """
    Filter the given point cloud.
    :param pc: point cloud to filter (format: [num_points, 4] = [x, y, z, label])
    :param config: configuration dictionary
    :return: mask (same size as the original point cloud) that contains True for points that are in the filtered
            point cloud and False for points that are not in the filtered point cloud
    """
    # Initialize mask
    mask = torch.ones(pc.shape[0], dtype=torch.bool)

    # Filter out points that belong to small clusters
    min_points = 50 if config is None else config["filtering"]["min_points"]
    mask[mask.clone()] = filter_pc_small_clusters(pc, min_points=min_points)

    # Filter out deviated points
    per = 0.5 if config is None else config["filtering"]["inliers_percentage_threshold"]
    eps = 0.1 if config is None else config["filtering"]["eps_curve_coarse"]
    mask[mask.clone()] = filter_pc_clusters_along_curve(pc[mask], eps=eps, keep_inliers_only=True,
                                                        inliers_percentage_threshold=per, config=config)

    # Filter out points that belong to small clusters
    mask[mask.clone()] = filter_pc_small_clusters(pc[mask], min_points=min_points)

    # Find lanes
    eps = 0.1 if config is None else config["filtering"]["eps_curve_fine"]
    mask[mask.clone()] = filter_pc_clusters_along_curve(pc[mask], eps=eps, keep_inliers_only=False,
                                                        inliers_percentage_threshold=per, config=config)

    # Filter out points that belong to small clusters
    mask[mask.clone()] = filter_pc_small_clusters(pc[mask], min_points=min_points)

    return mask


def filter_along_fitted_line(points: torch.Tensor, threshold: float = 0.5, iterations: int = 1000) -> dict:
    """
    Fit a line to a set of points using the RANSAC algorithm. Then, return the inliers and outliers indices and
    the end points of the line.
    :param points: points to fit a line to (format: [num_points, 2] = [x, y])
    :param threshold: threshold for the RANSAC algorithm
    :param iterations: number of iterations for the RANSAC algorithm
    :return: end points of the line, inliers and outliers indices (dict with keys 'p1', 'p2', 'inliers_idx',
            'outliers_idx')
    """
    # Randomly choose two points for all iterations
    idx = torch.randint(0, points.shape[0], (iterations, 2), device=points.device)
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]

    # Vector from p1 to p2 for each line
    v = p2 - p1

    # Vector from p1 to points for each line
    w = points - p1.unsqueeze(1)

    # Compute dot product between v and w for each line
    dot = torch.sum(v.unsqueeze(1) * w, dim=2)

    # Compute squared length of v for each line
    v_norm = torch.sum(v ** 2, dim=1)

    # Compute projection for each line
    proj = (dot / v_norm.unsqueeze(1)).unsqueeze(2) * v.unsqueeze(1)

    # Compute distance for each line
    dist = torch.sqrt(torch.sum((w - proj) ** 2, dim=2))

    # Count inliers for all iterations using vectorized comparison
    inliers_idx = dist < threshold
    outliers_idx = ~inliers_idx
    num_inliers = torch.sum(inliers_idx, dim=1)

    # Find the iteration with the maximum inliers
    best_iteration = torch.argmax(num_inliers)

    # Return only the best iteration
    inliers_idx = inliers_idx[best_iteration]
    outliers_idx = outliers_idx[best_iteration]

    return {'p1': p1[best_iteration], 'p2': p2[best_iteration], 'inliers_idx': inliers_idx,
            'outliers_idx': outliers_idx}


def filter_pc_clusters_along_fitted_lines(pc: torch.Tensor, eps: float = 0.5, keep_inliers_only: bool = True,
                                          inliers_percentage_threshold: float = 0.8,
                                          iterations: int = 1000) -> torch.Tensor:
    """
    First, fit a straight line to each cluster using the RANSAC algorithm. Keep only inliers of those clusters that have
    at least inliers_percentage_threshold of their points close enough to the fitted line.
    :param pc: point cloud to filter (format: [num_points, 4] = [x, y, z, label])
    :param eps: threshold for the distance to the line
    :param keep_inliers_only: whether to keep only inliers or also outliers of passed clusters
    :param inliers_percentage_threshold: threshold for the percentage of inliers in a cluster
    :param iterations: number of iterations for the RANSAC algorithm
    :return: mask of the filtered point cloud (format: [num_points])
    """
    # Extract clusters
    # - Get points and labels
    labels = pc[:, 3]

    # - Get unique labels
    unique_labels = torch.unique(labels)

    # Filter clusters
    mask = torch.zeros(pc.shape[0], dtype=torch.bool)
    for label in unique_labels:
        # Cluster mask
        cluster_mask = labels == label

        # Get cluster
        cluster = pc[cluster_mask]

        # Filter cluster
        res = filter_along_fitted_line(cluster[:, :2], eps, iterations)

        # If the cluster has less than inliers_percentage_threshold % of its points close enough to the fitted line,
        # throw the cluster away
        if torch.sum(res["inliers_idx"]) / cluster.shape[0] > inliers_percentage_threshold:
            if keep_inliers_only:
                mask[cluster_mask] = res["inliers_idx"]
            else:
                mask[cluster_mask] = True

    return mask


def filter_pc_small_clusters(pc: torch.Tensor, min_points: int = 100) -> torch.Tensor:
    """
    Filter small clusters from the given point cloud.
    :param pc: point cloud to filter (format: [num_points, 4] = [x, y, z, label])
    :param min_points: threshold for the number of points in a cluster
    :return: mask of the filtered point cloud (format: [num_points])
    """
    # Extract clusters
    # - Get labels
    labels = pc[:, 3]

    # - Get unique labels
    unique_labels, label_counts = torch.unique(labels, return_counts=True)

    # Get mask labels
    mask_labels = unique_labels[label_counts > min_points]

    # Create the mask
    mask = labels.unsqueeze(0) == mask_labels.unsqueeze(1)
    mask = mask.any(dim=0)

    return mask


def filter_pc_clusters_along_curve(pc: torch.Tensor, eps: float = 0.5, keep_inliers_only: bool = True,
                                   inliers_percentage_threshold: float = 0.8, config: dict = None) -> torch.Tensor:
    """
    First, find the two most distant points in each cluster. Then, find the shortest path between them through other
    points of the cluster. Finally, check how many points are close enough to the found path. Keep only those clusters
    that have more than inliers_percentage_threshold inliers.
    :param pc: point cloud to filter (format: [num_points, 4] = [x, y, z, label])
    :param eps: threshold for the distance to the line
    :param keep_inliers_only: whether to keep only inliers or also outliers of passed clusters
    :param inliers_percentage_threshold: threshold for the percentage of inliers in a cluster
    :param config: configuration dictionary
    :return: mask of the filtered point cloud (format: [num_points])
    """
    # Extract clusters
    # - Get labels
    labels = pc[:, 3]

    # - Get unique labels
    unique_labels = torch.unique(labels)

    # Define variables for the shortest path algorithm
    base_max_jump = 0.2 if config is None else config["filtering"]["base_max_jump"]
    max_jump_increase = 0.05 if config is None else config["filtering"]["max_jump_increase"]
    l = 0.01 if config is None else config["filtering"]["l"]    # interpolation step

    # Filter clusters
    mask = torch.zeros(pc.shape[0], dtype=torch.bool)
    for label in unique_labels:
        # Get cluster mask
        cluster_mask = labels == label

        # Get cluster
        cluster = pc[cluster_mask]

        # Create 2D cluster
        cluster_2d = cluster[:, :2]

        # Shift the points so the center of mass is in the origin (minimize the numerical errors)
        center_of_mass = torch.mean(cluster_2d, dim=0)
        cluster_2d = cluster_2d - center_of_mass

        # Get the two most distant points in the cluster
        dists = torch.cdist(cluster_2d, cluster_2d)
        max_dist_ind = torch.argmax(dists)

        # Get the indices of the two points
        start_point_idx = int(torch.div(max_dist_ind, dists.shape[0], rounding_mode='floor'))
        end_point_idx = max_dist_ind % dists.shape[0]

        # Find the shortest path between the two points
        path, cost = find_shortest_path(cluster, start_point_idx, end_point_idx, base_max_jump=base_max_jump,
                                        max_jump_increase=max_jump_increase)

        # Get inliers and outliers of the path
        inlier_ind = get_path_inlier_indices(cluster_2d, path, eps=eps, l=l)

        # # TODO: remove eventually (below)
        # # Visualize the result
        # plt.scatter(cluster[:, 0], cluster[:, 1], color="black")
        # plt.scatter(cluster[inlier_ind, 0], cluster[inlier_ind, 1], color="blue")
        # plt.scatter(cluster[start_point_idx, 0], cluster[start_point_idx, 1], color='green')
        # plt.scatter(cluster[end_point_idx, 0], cluster[end_point_idx, 1], color='green')
        # plt.scatter(cluster[path, 0], cluster[path, 1], color='red')
        # plt.plot(cluster[path, 0], cluster[path, 1], color='red')
        # plt.show()
        # # TODO: remove eventually (above)

        # If the cluster has less than inliers_percentage_threshold % of its points close enough to the fitted line,
        # throw the cluster away
        if torch.sum(inlier_ind) / cluster.shape[0] > inliers_percentage_threshold:
            if keep_inliers_only:
                mask[cluster_mask] = inlier_ind
            else:
                mask[cluster_mask] = True

    return mask


def find_shortest_path(points: torch.tensor, start_point_idx: int, end_point_idx: int, base_max_jump: float = 0.2,
                       max_jump_increase: float = 0.05) -> torch.tensor:
    """
    Find the shortest path between the two given points using Dijkstra's algorithm. The cost of the path is the sum of
    the distances between the points. The maximum jump cost (maximum distance between two points in the path) is
    iteratively increased until a path is found.
    :param points: point cloud (format: [num_points, 2] = [x, y])
    :param start_point_idx: index of the first point
    :param end_point_idx: index of the second point
    :param base_max_jump: base maximum jump cost
    :param max_jump_increase: maximum jump cost increase
    :return: indices of the points in the shortest path
    """
    # Initialize maximum jump cost and number of points
    num_points = points.shape[0]
    max_jump_cost = base_max_jump

    # Calculate distances between all points
    all_distances = torch.cdist(points, points)

    # Find the shortest path (increase the maximum jump cost until a path is found)
    while True:
        # Initialize Dijkstra's algorithm
        # - Initialize distances (make them infinite except for the start point)
        distances = torch.full((num_points,), float('inf'))
        distances[start_point_idx] = 0

        # - Initialize previous (make them -1 -> no previous point)
        previous = torch.full((num_points,), -1)

        # - Initialize heap
        heap = [(0, start_point_idx)]

        # Run Dijkstra's algorithm
        while heap:
            # Get the point with the smallest distance
            dist, current = heapq.heappop(heap)

            # Ignore paths that are already longer then the current shortest path to the current point
            if dist > distances[current]:
                continue

            # Calculate everything for all neighbors of the current point at once
            # - Get costs to all neighbors (i.e. distances to all neighbors from all_distances variable)
            costs = all_distances[current]

            # - Get indices of the neighbors that are closer than the maximum jump cost
            neighbors = costs <= max_jump_cost

            # - Calculate the total costs to all neighbors
            total_costs = dist + costs

            # - Create an update mask
            update = total_costs < distances

            # - Update the distances and previous points
            distances[neighbors & update] = total_costs[neighbors & update]
            previous[neighbors & update] = current

            # - Push the neighbors to the heap
            for neighbor in torch.nonzero(neighbors & update).flatten():
                heapq.heappush(heap, (total_costs[neighbor], neighbor))

        # Increase the maximum jump cost
        max_jump_cost += max_jump_increase

        # Stop if the end point is reached (i.e. it has a previous point)
        if previous[end_point_idx] != -1:
            break

    # Reconstruct the path
    path = []
    current = end_point_idx
    while current != -1:
        path.append(current)
        current = previous[current]
    path.reverse()

    return path, distances[end_point_idx]


def get_path_inlier_indices(points: torch.Tensor, path: list, eps: float = 1.0, l: float = 0.01) -> torch.Tensor:
    """
    Get the inlier indices of a path.
    :param points: points to get the inliers and outliers from (format: [num_points, 2] = [x, y])
    :param path: indices of path points (format: [num_path_points] = [point_idx])
    :param eps: epsilon for the distance
    :param l: interpolation step
    :return: inliers indices
    """
    # Convert path to tensor
    path = torch.tensor(path)

    # Get path points
    path_points = points[path]

    # Interpolate the path
    # print(f"\n\nInterpolating path with {path_points.shape[0]} points")
    start = timer()
    path_points = interpolate_path(path_points, l=l)
    end = timer()
    # print(f"Interpolation took {end - start} seconds")

    # Get the distance between each point and the interpolated path
    start = timer()
    dists = torch.cdist(points, path_points)
    min_dists, _ = torch.min(dists, dim=1)
    end = timer()
    # print(f"Distance calculation took {end - start} seconds")

    # Get the inliers and outliers
    inliers_idx = min_dists <= eps

    return inliers_idx


def interpolate_path(path_points: torch.Tensor, l: float = 0.01) -> torch.Tensor:
    """
    Interpolate a path with more points.
    :param path_points: path points (format: [num_path_points, 2] = [x, y])
    :param l: distance between interpolated points
    :return: interpolated path points
    """
    # Calculate the pairwise differences between consecutive points
    point_diffs = path_points[1:] - path_points[:-1]

    # Calculate distances between consecutive points
    distances = torch.norm(point_diffs, dim=1)

    # Calculate the number of points to interpolate between each pair of points
    n = (distances / l).ceil().int()  # Using ceil to ensure we cover the required distance

    # Create interpolation ratios for each segment
    t_values = [torch.linspace(0, 1, steps=n[i]+1, device=path_points.device) for i in range(len(n))]

    # Calculate interpolated points for each segment
    interpolated_segments = [path_points[i] + t[:, None] * point_diffs[i] for i, t in enumerate(t_values)]

    # Stack the interpolated segments
    interp_points = torch.cat(interpolated_segments, dim=0)

    return interp_points


if __name__ == "__main__":
    # Generate line data (noised)
    a = 1
    b = 0
    c = 0
    x = torch.linspace(-1, 1, 100)
    y = a * x + b + torch.rand(100) * 2 - 1
    z = torch.zeros(100)
    labels = torch.zeros(60)
    labels = torch.cat((labels, torch.ones(40)))
    points = torch.stack((x, y, z, labels), dim=1)

    # Add outliers
    filtered_points = filter_pc_small_clusters(points, min_points=50)

    # Plot
    plt.plot(points[:, 0], points[:, 1], 'o', c='b')
    plt.plot(filtered_points[:, 0], filtered_points[:, 1], 'o', c='r')
    plt.plot(x, a*x+b, 'r')
    plt.show()
