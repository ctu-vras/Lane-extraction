import torch

from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.points_normals import estimate_pointcloud_normals


def chamfer_distance_loss(x, y, x_lengths=None, y_lengths=None, both_ways=False, normals_K=0, loss_norm=1):
    '''
    Unique Nearest Neightboors
    :param x: first point cloud 1 x N x 3
    :param y: second point cloud 1 x M x 3
    :param x_lengths:
    :param y_lengths:
    :param reduction:
    :return:
    '''
    if normals_K >= 3:
        normals1 = estimate_pointcloud_normals(x, neighborhood_size=normals_K)
        normals2 = estimate_pointcloud_normals(y, neighborhood_size=normals_K)

        x = torch.cat([x, normals1], dim=-1)
        y = torch.cat([y, normals2], dim=-1)


    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1, norm=loss_norm)
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    x_nearest_to_y = x_nn[1]

    if both_ways:
        y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1, norm=loss_norm)
        cham_y = y_nn.dists[..., 0]  # (N, P2)
        y_nearest_to_x = y_nn[1]

        nn_loss = (cham_x.mean() + cham_y.mean() ) / 2 # different shapes

    else:
        nn_loss = cham_x.mean()

    return nn_loss, cham_x, x_nearest_to_y


def smoothness_loss(est_flow, NN_idx, loss_norm=1, mask=None):
    '''

    :param est_flow: vector field you want to push together
    :param NN_idx: Indices of KNN N x K, that push together points in the row by index
    :param loss_norm:
    :param mask:
    :return:
    '''
    bs, n, c = est_flow.shape

    if bs > 1:
        print("Smoothness Maybe not working, needs testing!")
    K = NN_idx.shape[2]

    est_flow_neigh = est_flow.view(bs * n, c)
    est_flow_neigh = est_flow_neigh[NN_idx.view(bs * n, K)]

    est_flow_neigh = est_flow_neigh[:, 1:K + 1, :]
    flow_diff = est_flow.view(bs * n, c) - est_flow_neigh.permute(1, 0, 2)

    flow_diff = (flow_diff).norm(p=loss_norm, dim=2)
    smooth_flow_loss = flow_diff.mean()
    smooth_flow_per_point = flow_diff.mean(dim=0).view(bs, n)

    return smooth_flow_loss, smooth_flow_per_point


def mask_NN_by_dist(dist, nn_ind, max_radius):
    '''
    :param dist:
    :param nn_ind:
    :param max_radius:
    :return:
    '''
    tmp_idx = nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, nn_ind.shape[-1]).to(nn_ind.device)
    nn_ind[dist > max_radius] = tmp_idx[dist > max_radius]

    return nn_ind


if __name__ == "__main__":
    pass
