import torch
import subprocess


def my_gumbel_softmax(logits, hard=False, dim=-1):
    """
    Hot one encoding with gumbel softmax.
    Adjusted version of gumbel_softmax from pytorch.nn.functional - now it is deterministic. (no noise)
    (https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax)
    :param logits: input tensor
    :param hard: if True, the returned samples will be discretized as one-hot vectors,
                    but will be differentiated as if it is the soft sample in autograd
    :param dim: dimension along which softmax will be computed
    :return: hot one encoding of the input tensor
    """
    gumbels = logits
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret


def initialize_vectors_for_instances(pc: torch.Tensor) -> {torch.Tensor, torch.Tensor}:
    """
    Initialize vectors for instances via PCA and place them at the center of mass.
    :param pc: point cloud (format: [num_points, 4] = [num_points, x, y, z, i])
    :return: dictionary with keys "vectors" and "centers" and values being the corresponding tensors
    """
    # Check if point cloud is of correct format
    if len(pc.shape) != 2:
        raise ValueError("Point cloud must be of format [num_points, 4] = [num_points, x, y, z, i]")

    # Get instances per point
    instances = pc[:, 3]

    # Get unique instances
    unique_instances = torch.unique(instances)

    # Compute centers of mass and PCA
    centers = torch.zeros((len(unique_instances), 3), device=pc.device)
    vectors = torch.zeros((len(unique_instances), 3), device=pc.device)
    for i in range(len(unique_instances)):
        # Get points of this instance
        points = pc[instances == unique_instances[i], :3]

        # Compute center of mass
        centers[i] = torch.mean(points, dim=0)

        # Compute PCA
        # Find principal components
        _, _, v = torch.pca_lowrank(points - centers[i])

        try:
            vectors[i] = (v[:, 0])
        except IndexError:
            print("It is not possible to find principal components of this cluster.")
            vectors[i] = torch.tensor([0, 0, 0])

    # Return dictionary
    return {"vectors": vectors, "centers": centers}


def group_instances(pc: torch.Tensor) -> torch.Tensor:
    """
    Group instances extracted from multiple frames into one instance if they share at least one point.
    :param pc: a tensor of shape (N, 3) = [[x, y, instance_id], ...] where N is the number of points
    :return: a tensor of shape (N, 3) = [[x, y, instance_id], ...] where N is the number of points
    """
    # Get all instances
    instances = torch.unique(pc[:, 2])

    # Group instance
    replaced_instances = []
    for instance in instances:
        if instance in replaced_instances:
            continue
        # Get all points of the current instance
        mask = pc[:, 2] == instance
        instance_points = pc[mask]

        # Get all points of the other instances
        other_points = pc[~mask]

        # Get all points that are in instance_points as well as in other_points
        intersection_indices = find_matching_indices(instance_points[:, :2], other_points[:, :2])

        # Get all points that are in the intersection
        intersection_points = other_points[intersection_indices]

        # Get all instances that are in the intersection
        intersection_instances = torch.unique(intersection_points[:, 2])

        # Change all instances of the points in the intersection to the current instance
        for intersection_instance in intersection_instances:
            pc[pc[:, 2] == intersection_instance, 2] = instance
            replaced_instances.append(intersection_instance)

    return pc


def find_matching_indices(points1, points2):
    """
    Find indices of points in points2 that are also in points1.
    :param points1: a tensor of shape (N, 2) = [[x, y], ...] where N is the number of points
    :param points2: a tensor of shape (M, 2) = [[x, y], ...] where M is the number of points
    :return: a tensor of shape (K, 1) = [[index], ...] where K is the number of matching points
    """
    # Expand dimensions for broadcasting
    expanded_points1 = points1[:, None, :]
    expanded_points2 = points2[None, :, :]

    # Create a boolean mask for matching points
    matching_mask = torch.all(expanded_points1 == expanded_points2, dim=2)

    # Find indices of True values in the mask
    matching_indices = torch.nonzero(matching_mask, as_tuple=False)

    return matching_indices[:, 1]


def get_number_of_frames_in_group(number_of_frames: int, max_frames: int, min_frames: int) -> int:
    """
    Function for splitting the point cloud into groups of frames. The number of frames in each group should be
    between min_frames and max_frames and as balanced as possible.
    :param number_of_frames: number of frames in the point cloud
    :param max_frames: maximum number of frames in a group
    :param min_frames: minimum number of frames in a group
    :return: number of frames in a group
    """
    # Check if the number of frames is at least min_frames
    if number_of_frames < min_frames:
        print(f"! Not enough frames provided. The minimum number of frames is {min_frames} but only {number_of_frames} "
              f"were provided. !")
        return number_of_frames

    # Check if the number of frames is at most max_frames
    if number_of_frames < max_frames:
        return number_of_frames

    # Find the best number of frames in a group
    max_remainder = 0
    number_of_frames_in_group = number_of_frames
    zero_remainder_found = False
    for i in range(min_frames, max_frames + 1):
        remainder = number_of_frames % i
        if remainder > max_remainder and not zero_remainder_found:
            max_remainder = remainder
            number_of_frames_in_group = i

        # If the remainder is 0, then we will try to find a bigger number of frames in a group that also has a 0
        # remainder
        if remainder == 0:
            zero_remainder_found = True
            number_of_frames_in_group = i

    return number_of_frames_in_group
