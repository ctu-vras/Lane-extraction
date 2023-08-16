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


