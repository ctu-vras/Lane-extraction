#!/usr/bin/env python
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points


def calculate_chamfer_loss(centers_array, vectors_array, outreach_mask,device):
    #print(outreach_mask)
    sum_of_losses = torch.tensor(0.0,device=device)
    for i in range(vectors_array.shape[0]):
        if outreach_mask[i].item() is False:
            continue
        mask = np.ones(centers_array.shape[0], dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)  # Reshape to (1, num_points, 3)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 2)
        #chamfer_loss, _ = chamfer_distance(centers_array_i, new_centers_array, single_directional=True)
        chamfer_loss,_,_ = knn_points(centers_array_i,new_centers_array,K=1)
        sum_of_losses += chamfer_loss[0][0][0]
    return sum_of_losses


def calculate_differentiable_smoothness(centers_array, vectors_array, outreach_mask,device):
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]
    indices = torch.zeros(vectors_number, 2, dtype=torch.int64, device=device)
    for i in range(vectors_number):
        if outreach_mask[i].item() is False:
            continue
        mask = np.ones(num_centers, dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 2)
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=2)
        idx = idx[0][0][0]
        if idx >= i:
            idx += 1
        indices[i, :] = idx
    vec = torch.zeros(num_centers, 2, dtype=torch.float32, device=device)
    vec = vec.scatter_reduce(dim=0, index=indices, src=vectors_array, reduce='mean', include_self=False)
    res = torch.nn.functional.mse_loss(vectors_array, vec[indices[:, 0]], reduction='sum')
    return res


def calculate_continuity(centers_array, vectors_array):
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]

    indices = torch.zeros(vectors_number, 2, dtype=torch.int64)
    for i in range(vectors_array.shape[0]):
        mask = np.ones(num_centers, dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 2)
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=2)
        idx = idx[0][0][0]
        if idx >= i:
            idx += 1
        indices[i, :] = idx
    vec = vectors_array.mean(dim=0).repeat(vectors_number, 1)
    res = torch.nn.functional.mse_loss(vectors_array, vec, reduction='sum')
    return res


def vector_exclusivity(centers_array, vectors_array, outreach_mask,device):
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]

    indices = torch.zeros(vectors_number, dtype=torch.int64, device=device)
    for i in range(vectors_number):
        mask = np.ones(num_centers, dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 2)
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=1)
        idx = idx[0][0][0]
        if idx >= i:
            idx += 1
        indices[i] = idx
    sum_of_losses = torch.tensor(0.0)
    for j in range(num_centers):
        if outreach_mask[j].item() is False:
            continue
        mask = torch.nonzero(indices == j).flatten()
        if len(mask) > 1:
            pairs = torch.combinations(mask, r=2)
            index1, index2 = pairs.unbind(1)
            if outreach_mask[index1].item() is False or outreach_mask[index2].item() is False:
                continue
            res = torch.nn.functional.mse_loss(
                vectors_array[index1],
                vectors_array[index2],
                reduction='sum'
            )
            sum_of_losses += res
    return sum_of_losses


def exclusivity_repulsion(centers_array, vectors_array, outreach_mask,device):
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]
    indices = torch.zeros(vectors_number, dtype=torch.int64,device=device)
    for i in range(vectors_number):
        mask = np.ones(num_centers, dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 2)
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=1)
        idx = idx[0][0][0]
        if idx >= i:
            idx += 1
        indices[i] = idx
    sum_of_losses = torch.tensor(0.0)
    for j in range(num_centers):
        if outreach_mask[j].item() is False:
            continue
        mask = torch.nonzero(indices == j).flatten()
        if len(mask) > 1:
            pairs = torch.combinations(mask, r=2)
            index1, index2 = pairs.unbind(1)
            if outreach_mask[index1].item() is False or outreach_mask[index2].item() is False:
                continue
            res = -torch.nn.functional.mse_loss(
                centers_array[j] - (centers_array[index1] + vectors_array[index1]),
                centers_array[j] - (centers_array[index2] + vectors_array[index2]),
                reduction='sum'
            )
            sum_of_losses += res
    return sum_of_losses


def pca_differ(vectors_array, initial_vectors):
    sum_of_losses = torch.tensor(0.0)
    vector_normal = torch.nn.functional.normalize(vectors_array)
    initial_vector_normal = torch.nn.functional.normalize(initial_vectors)
    for i in range(vectors_array.shape[0]):
        res = torch.nn.functional.mse_loss(vector_normal[i], initial_vector_normal[i], reduction='sum')
        sum_of_losses += res
    return sum_of_losses


def vector_differ(vectors_array, multiplier):
    sum_of_losses = torch.tensor(0.0)
    mean_vector = torch.mean(vectors_array, dim=0)
    normalized_mean = multiplier * torch.nn.functional.normalize(mean_vector, dim=0)
    for i in range(vectors_array.shape[0]):
        res = torch.nn.functional.mse_loss(vectors_array[i], normalized_mean, reduction='sum')
        sum_of_losses += res
    return sum_of_losses


def compute_multiplier(centers_array, vectors_array, knn_taken):
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]
    mean_dist = 0
    for i in range(vectors_number):
        mask = np.ones(num_centers, dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 2)
        dist, idx, _ = knn_points(centers_array_i, new_centers_array, K=knn_taken)
        mean_dist += dist[0][0][knn_taken - 1].item()
    mean_dist /= vectors_number
    return mean_dist ** (1 / 2)


def find_closest_direction(padded_centers_array, padded_vectors_directions):
    # with torch.no_grad():
    output_vector = torch.ones(padded_centers_array.shape[0], dtype=torch.bool)
    for i in range(padded_centers_array.shape[0]):
        mask = np.ones(padded_centers_array.shape[0], dtype=bool)
        mask[i] = False
        new_centers_array = padded_centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)
        centers_array_i = (padded_centers_array[i] + padded_vectors_directions[i]).reshape(1, -1, 2)
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=1)
        neighbours = idx

        # print(padded_vectors_directions[i])
        ex_idx = neighbours[0][0][0].item()
        if ex_idx >= i:
            ex_idx += 1
        point_in_half = (padded_centers_array[i] + ((1 / 2) * padded_vectors_directions[i]))
        new_direct = (1 / 2) * padded_vectors_directions[i]
        c = -torch.dot(new_direct, point_in_half)
        value_of_center = (new_direct[0] * padded_centers_array[i][0]) + (
                new_direct[1] * padded_centers_array[i][1]) + c
        value_of_togo = (new_direct[0] * padded_centers_array[ex_idx][0]) + (
                new_direct[1] * padded_centers_array[ex_idx][1]) + c
        if value_of_togo <= 0 and value_of_center <= 0 or value_of_togo >= 0 and value_of_center >= 0:
            output_vector[i] = False
    return output_vector

def compute_opposite_pca(vectors_array):
    correct_orient = torch.ones(vectors_array.shape[0], dtype=torch.float)
    mean_vector = torch.mean(vectors_array, dim=0)
    for i in range(vectors_array.shape[0]):
        value = torch.dot(mean_vector, vectors_array[i])
        if value < 0:
            correct_orient[i] = -1.0
    return correct_orient
