import torch
import numpy as np
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
import matplotlib.pyplot as plt


def calculate_differentiable_smoothness(centers_array, vectors_array):
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]
    indices = torch.zeros(vectors_number, 3, dtype=torch.int64)
    for i in range(vectors_number):
        mask = np.ones(num_centers, dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 3)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 3)
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=2)
        idx = idx[0][0][0]
        if idx >= i:
            idx += 1
        indices[i, :] = idx
    vec = torch.zeros(num_centers, 3)
    vec = vec.scatter_reduce(dim=0, index=indices, src=vectors_array, reduce='mean', include_self=False)
    res = torch.nn.functional.mse_loss(vectors_array, vec[indices[:, 0]], reduction='sum')
    return res


def calculate_continuity(centers_array, vectors_array):
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]

    indices = torch.zeros(vectors_number, 3, dtype=torch.int64)
    for i in range(vectors_array.shape[0]):
        mask = np.ones(num_centers, dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 3)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 3)
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=2)
        idx = idx[0][0][0]
        if idx >= i:
            idx += 1
        indices[i, :] = idx
    vec = vectors_array.mean(dim=0).repeat(vectors_number, 1)
    res = torch.nn.functional.mse_loss(vectors_array, vec, reduction='sum')
    return res


def vector_exclusivity(centers_array, vectors_array):
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]

    indices = torch.zeros(vectors_number, dtype=torch.int64)
    for i in range(vectors_number):
        mask = np.ones(num_centers, dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 3)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 3)
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=1)
        idx = idx[0][0][0]
        if idx >= i:
            idx += 1
        indices[i] = idx
    sum_of_losses = torch.tensor(0.0)
    for j in range(num_centers):
        mask = torch.nonzero(indices == j).flatten()
        if len(mask) > 1:
            pairs = torch.combinations(mask, r=2)
            index1, index2 = pairs.unbind(1)
            res = torch.nn.functional.mse_loss(
                vectors_array[index1],
                vectors_array[index2],
                reduction='sum'
            )
            sum_of_losses += res
    return sum_of_losses


def matching_main(point_cloud):
    padded_centers_array = torch.tensor(point_cloud['data'].copy())
    vectors_directions = point_cloud['vectors']
    optimized_directions = torch.tensor(vectors_directions)
    optimized_directions = torch.nn.functional.normalize(optimized_directions)
    optimized_directions.requires_grad = True
    optimizer = torch.optim.SGD([optimized_directions], lr=0.02)
    iteration_number = 150
    chamfer_losses = torch.zeros(iteration_number).detach()
    vector_smoothness_losses = torch.zeros(iteration_number).detach()
    exclusivity_losses = torch.zeros(iteration_number).detach()

    for j in range(iteration_number):
        optimizer.zero_grad()
        sum_of_losses = torch.tensor(0.0)
        for i in range(optimized_directions.shape[0]):
            mask = np.ones(padded_centers_array.shape[0], dtype=bool)
            mask[i] = False
            new_centers_array = padded_centers_array[mask].clone().detach()
            new_centers_array = new_centers_array.reshape(1, -1, 3)  # Reshape to (1, num_points, 3)
            centers_array_i = (padded_centers_array[i] + optimized_directions[i]).reshape(1, -1, 3)
            chamfer_loss, _ = chamfer_distance(centers_array_i, new_centers_array, single_directional=True)
            sum_of_losses += chamfer_loss
        chamfer_losses[j] = sum_of_losses
        value = 4 * calculate_differentiable_smoothness(padded_centers_array, optimized_directions)
        exclusivity = 22 * vector_exclusivity(padded_centers_array, optimized_directions)

        vector_smoothness_losses[j] = value
        exclusivity_losses[j] = exclusivity
        sum_of_losses += value
        sum_of_losses.backward()
        optimizer.step()

    for j in range(iteration_number):
        optimizer.zero_grad()
        sum_of_losses = torch.tensor(0.0)
        for i in range(optimized_directions.shape[0]):
            mask = np.ones(padded_centers_array.shape[0], dtype=bool)
            mask[i] = False
            new_centers_array = padded_centers_array[mask].clone().detach()
            new_centers_array = new_centers_array.reshape(1, -1, 3)  # Reshape to (1, num_points, 3)
            centers_array_i = (padded_centers_array[i] + optimized_directions[i]).reshape(1, -1, 3)
            chamfer_loss, _ = chamfer_distance(centers_array_i, new_centers_array, single_directional=True)
            sum_of_losses += chamfer_loss
        sum_of_losses.backward()
        optimizer.step()

    #to have some return
    point_cloud['matching'] = np.ones(point_cloud['data'].shape[0])
