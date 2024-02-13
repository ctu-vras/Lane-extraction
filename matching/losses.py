#!/usr/bin/env python
import numpy as np
import torch
from pytorch3d.ops import knn_points


# functioan that calculates distance to nearest point in the pointcloud
def calculate_chamfer_loss(centers_array, vectors_array, outreach_mask, device):
    """
    :param centers_array: centers of clusters
    :param vectors_array: vectors from pca
    :param outreach_mask: mask that masks last points of polyline
    :param device: cuda device
    """
    # print(outreach_mask)
    sum_of_losses = torch.tensor(0.0, device=device)
    for i in range(vectors_array.shape[0]):
        # skip point if it is from end of polyline
        if outreach_mask[i].item() is False:
            continue
        # create mask to create a point cloud excluding the current point
        mask = np.ones(centers_array.shape[0], dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)  # Reshape to (1, num_points, 3)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 2)
        # calculate distance to nearest point from center+vector
        chamfer_loss, _, _ = knn_points(centers_array_i, new_centers_array, K=1)
        sum_of_losses += chamfer_loss[0][0][0]
    return sum_of_losses


# function that calculates how smooth is each connection of points
def calculate_differentiable_smoothness(centers_array, vectors_array, outreach_mask, device):
    """
    :param centers_array: centers of clusters
    :param vectors_array: vectors from pca
    :param outreach_mask: mask that masks last points of polyline
    :param device: cuda device

    """
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]
    indices = torch.zeros(vectors_number, 2, dtype=torch.int64, device=device)
    for i in range(vectors_number):
        # skip if is end of polyline
        if outreach_mask[i].item() is False:
            continue
        mask = np.ones(num_centers, dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)
        centers_array_i = (centers_array[i] + vectors_array[i]).reshape(1, -1, 2)
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=1)
        idx = idx[0][0][0]
        # shift index because of skipping current center
        # so if we have lower index than current point we dont need to shift
        # if it is equal or higher, we need to add+1 to get vector we want
        if idx >= i:
            idx += 1
        indices[i, :] = idx  # save index of nearest point
    vec = torch.zeros(num_centers, 2, dtype=torch.float32, device=device)
    # this was between me,god and computer
    # now only god and computer knows what is happening here
    # basically it calculates diff for each center so that the input vectors have same direction as output vector
    # here it creaters mean of vectors that are connected to the same center
    vec = vec.scatter_reduce(dim=0, index=indices, src=vectors_array, reduce='mean', include_self=False)
    res = torch.nn.functional.mse_loss(vectors_array, vec[indices[:, 0]], reduction='sum')
    return res


# not used anymore, but it took mean of all vectors and tried to make them point in that direction
# so all vectors would have same size,and orientation
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
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=1)
        idx = idx[0][0][0]
        if idx >= i:
            idx += 1
        indices[i, :] = idx

    vec = vectors_array.mean(dim=0).repeat(vectors_number, 1)
    res = torch.nn.functional.mse_loss(vectors_array, vec, reduction='sum')
    return res


# function that for each center that for each pair of vectors, heading to same center, it forces them to be same
# which in combination that both of them came from different centers, it makes them further from the center
# and hopefully one of therm find different center to go to
def vector_exclusivity(centers_array, vectors_array, outreach_mask, device):
    """
    :param centers_array: centers of clusters
    :param vectors_array: vectors from pca
    :param outreach_mask: mask that masks last points of polyline
    :param device: cuda device
    """
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]
    # discover what is the nearest neigbour for each point
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
    sum_of_losses = torch.tensor(0.0,device=device)
    for j in range(num_centers):
        # skip points that are from end of polyline
        if outreach_mask[j].item() is False:
            continue
        # for each center find all vectors that are heading to it
        mask = torch.nonzero(indices == j).flatten()
        # if more that one vector heading to city
        if len(mask) > 1:
            # creaters all pairs of vectors that are heading to same city
            pairs = torch.combinations(mask, r=2)
            for pair in pairs:
                index1,index2 = pair
                index1 = index1.item()
                index2 = index2.item()

            # skip if one of the vectors is from end of polyline
                if outreach_mask[index1].item() is False or outreach_mask[index2].item() is False:
                    continue
                # loss in form of diff between two vectors heading in
                res = torch.nn.functional.mse_loss(
                    vectors_array[index1],
                    vectors_array[index2],
                    reduction='sum'
                )
                sum_of_losses += res
    return sum_of_losses


# almost the same as vector_exclusivity, but it forces vectors to go far from center,
# so both of them are repulsed from center, until one of them finds different city.
def exclusivity_repulsion(centers_array, vectors_array, outreach_mask, device):
    """
    :param centers_array: centers of clusters
    :param vectors_array: vectors from pca
    :param outreach_mask: mask that masks last points of polyline
    :param device: cuda device
    """
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]
    # discover what is the nearest neighbour for each point
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
    sum_of_losses = torch.tensor(0.0,device=device)
    for j in range(num_centers):
        # skip points that are from end of polyline
        if outreach_mask[j].item() is False:
            continue
        # for each center find all vectors that are heading to it
        mask = torch.nonzero(indices == j).flatten()
        # if more that one vector heading to city
        if len(mask) > 1:
            # creaters all pairs of vectors that are heading to same city
            pairs = torch.combinations(mask, r=2)
            for pair in pairs:
                index1, index2 = pair
                index1 = index1.item()
                index2 = index2.item()

                # skip if one of the vectors is from end of polyline
                if outreach_mask[index1].item() is False or outreach_mask[index2].item() is False:
                    continue
            # loss is to minimize negative of distance to center
            # so if there would not be -, it would push them togther
            # now it pushes them away, until they connect to different center
                res = -torch.nn.functional.mse_loss(
                    centers_array[j] - (centers_array[index1] + vectors_array[index1]),
                    centers_array[j] - (centers_array[index2] + vectors_array[index2]),
                    reduction='sum'
                )
                sum_of_losses += res
    return sum_of_losses


# function that calculates how much vectors differ from what it originally was
def pca_differ(vectors_array, initial_vectors):
    sum_of_losses = torch.tensor(0.0)
    vector_normal = torch.nn.functional.normalize(vectors_array)
    initial_vector_normal = torch.nn.functional.normalize(initial_vectors)
    for i in range(vectors_array.shape[0]):
        res = torch.nn.functional.mse_loss(vector_normal[i], initial_vector_normal[i], reduction='sum')
        sum_of_losses += res
    return sum_of_losses


# function that calculates how much vectors differ from mean vector
# pottenially same as calculate_continuity but either of them is not used
def vector_differ(vectors_array, multiplier):
    sum_of_losses = torch.tensor(0.0)
    mean_vector = torch.mean(vectors_array, dim=0)
    normalized_mean = multiplier * torch.nn.functional.normalize(mean_vector, dim=0)
    for i in range(vectors_array.shape[0]):
        res = torch.nn.functional.mse_loss(vectors_array[i], normalized_mean, reduction='sum')
        sum_of_losses += res
    return sum_of_losses


# function that for each vector calculates how much bigger it should be to create a good initialization

# vectors need to be normalized
def compute_multiplier(centers_array, vectors_array, knn_taken=5):
    """
    :param centers_array: centers of clusters
    :param vectors_array: vectors from pca, normalized
    :param knn_taken: how many nearest neighbours to take into account
    """
    num_centers = centers_array.shape[0]
    vectors_number = vectors_array.shape[0]
    # mask to store how much will each vector be multiplied
    mult_mask = torch.zeros(num_centers)
    # check not to have more neighbours than there is centers
    if knn_taken > num_centers - 1:
        knn_taken = num_centers - 1
    for i in range(vectors_number):
        mask = np.ones(num_centers, dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)
        centers_array_i = (centers_array[i]).reshape(1, -1, 2)
        # find knn_taken nearest neighbours
        dist, idx, _ = knn_points(centers_array_i, new_centers_array, K=knn_taken)
        # set initial values
        min_loss = np.inf
        distance_to_set = 0
        # dist are squared distances, so we need to take sqrt
        dist = torch.sqrt(dist)
        # take point furthest away from center
        dist_max = torch.max(dist)
        for j in range(knn_taken):
            # get index of center to check in centers_array
            cur_idx = idx[0][0][j]
            if cur_idx >= i:
                cur_idx += 1
            # calculate angle between vector we have from pca and direction to center we want to check
            inner_prod = torch.dot(vectors_array[i],
                                   torch.nn.functional.normalize((centers_array[cur_idx] - centers_array[i]), dim=0))
            angle_rad = torch.arccos(inner_prod)
            # select center that has combination of smallest angle between vector and direction to center and is close
            cur_loss = dist[0][0][j] + angle_rad * dist_max
            # save current best candidate
            if min_loss >= cur_loss:
                min_loss = cur_loss
                distance_to_set = dist[0][0][j]
        # set multiplier to be distance to that candidate
        mult_mask[i] = 0.9 * distance_to_set
    return torch.tensor(mult_mask).clone().detach()


def find_closest_direction(centers_array, vectors_directions,device):
    # with torch.no_grad():
    output_vector = torch.ones(centers_array.shape[0], dtype=torch.bool)
    for i in range(centers_array.shape[0]):
        mask = np.ones(centers_array.shape[0], dtype=bool)
        mask[i] = False
        new_centers_array = centers_array[mask].clone().detach()
        new_centers_array = new_centers_array.reshape(1, -1, 2)
        centers_array_i = (centers_array[i] + vectors_directions[i]).reshape(1, -1, 2)
        _, idx, _ = knn_points(centers_array_i, new_centers_array, K=1)
        neighbours = idx

        # print(padded_vectors_directions[i])
        ex_idx = neighbours[0][0][0].item()
        if ex_idx >= i:
            ex_idx += 1
        point_in_half = (centers_array[i] + ((3 / 4) * vectors_directions[i]))
        new_direct = (3 / 4) * vectors_directions[i]
        c = -torch.dot(new_direct, point_in_half)
        value_of_center = (new_direct[0] * centers_array[i][0]) + (
                new_direct[1] * centers_array[i][1]) + c
        value_of_togo = (new_direct[0] * centers_array[ex_idx][0]) + (
                new_direct[1] * centers_array[ex_idx][1]) + c
        if value_of_togo <= 0 and value_of_center <= 0 or value_of_togo >= 0 and value_of_center >= 0:
            output_vector[i] = False
        output_vector = output_vector.to(device)
    return output_vector


def compute_opposite_pca(vectors_array):
    correct_orient = torch.ones(vectors_array.shape[0], dtype=torch.float)
    mean_vector = torch.mean(vectors_array, dim=0)
    for i in range(vectors_array.shape[0]):
        value = torch.dot(mean_vector, vectors_array[i])
        if value < 0:
            correct_orient[i] = -1.0
    return correct_orient