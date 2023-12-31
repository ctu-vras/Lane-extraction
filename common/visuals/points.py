import os.path
import socket

import matplotlib.pyplot as plt
import numpy as np
import torch


class FileWatcher():

    def __init__(self, file_path):
        self.file_path = file_path
        self.last_modified = os.stat(file_path).st_mtime

    def check_modification(self):
        stamp = os.stat(self.file_path).st_mtime
        if stamp > self.last_modified:
            self.last_modified = stamp
            return True


def visualize_points3D(points, labels=None, point_size=0.02, **kwargs):
    if not socket.gethostname().startswith("Pat"):
        return

    if type(points) is not np.ndarray:
        points = points.detach().cpu().numpy()

    if type(labels) is not np.ndarray and labels is not None:
        labels = labels.detach().cpu().numpy()


    if labels is None:
        v = pptk.viewer(points[:,:3])
    else:
        v = pptk.viewer(points[:, :3], labels)

    v.set(point_size=point_size)
    v.set(**kwargs)

    return v


def visualize_voxel(voxel, cell_size=(0.2, 0.2, 0.2)):
    x,y,z = np.nonzero(voxel)
    label = voxel[x,y,z]
    pcl = np.stack([x / cell_size[0], y / cell_size[1], z / cell_size[2]]).T
    visualize_points3D(pcl, label)

def visualize_poses(poses):
    xyz = poses[:,:3,-1]
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(xyz[:,0], xyz[:,1])
    res = np.abs(poses[:-1, :3, -1] - poses[1:, :3, -1]).sum(1)
    axes[1].plot(res)
    plt.show()

def visualize_multiple_pcls(*args, **kwargs):
    p = []
    l = []

    for n, points in enumerate(args):
        if type(points) == torch.Tensor:
            p.append(points[:,:3].detach().cpu().numpy())
        else:
            p.append(points[:,:3])
        l.append(n * np.ones((points.shape[0])))

    p = np.concatenate(p)
    l = np.concatenate(l)
    v=visualize_points3D(p, l)
    v.set(**kwargs)

def visualize_plane_with_points(points, n_vector, d):

    xx, yy = np.meshgrid(np.linspace(points[:,0].min(), points[:,0].max(), 100),
                         np.linspace(points[:,1].min(), points[:,1].max(), 100))

    z = (- n_vector[0] * xx - n_vector[1] * yy - d) * 1. / n_vector[2]
    x = np.concatenate(xx)
    y = np.concatenate(yy)
    z = np.concatenate(z)

    plane_pts = np.stack((x, y, z, np.zeros(z.shape[0]))).T

    d_dash = - n_vector.T @ points[:,:3].T

    bin_points = np.concatenate((points, (d - d_dash)[:, None]), axis=1)

    vis_pts = np.concatenate((bin_points, plane_pts))

    visualize_points3D(vis_pts, vis_pts[:,3])


def visualize_flow3d(pts1, pts2, frame_flow):
    # flow from multiple pcl vis
    # valid_flow = frame_flow[:, 3] == 1
    # vis_flow = frame_flow[valid_flow]
    # threshold for dynamic is flow larger than 0.05 m

    if len(pts1.shape) == 3:
        pts1 = pts1[0]

    if len(pts2.shape) == 3:
        pts2 = pts2[0]

    if len(frame_flow.shape) == 3:
        frame_flow = frame_flow[0]

    if type(pts1) is not np.ndarray:
        pts1 = pts1.detach().cpu().numpy()

    if type(pts2) is not np.ndarray:
        pts2 = pts2.detach().cpu().numpy()

    if type(frame_flow) is not np.ndarray:
        frame_flow = frame_flow.detach().cpu().numpy()

    dist_mask = np.sqrt((frame_flow[:,:3] ** 2).sum(1)) > 0.05

    vis_pts = pts1[dist_mask,:3]
    vis_flow = frame_flow[dist_mask]

    # todo color for flow estimate
    # for raycast
    # vis_pts = pts1[valid_flow, :3]
    # vis_pts = pts1[dist_mask, :3]

    all_rays = []
    # breakpoint()
    for x in range(1, int(20)):
        ray_points = vis_pts + (vis_flow[:, :3]) * (x / int(20))
        all_rays.append(ray_points)

    all_rays = np.concatenate(all_rays)

    visualize_multiple_pcls(*[pts1, all_rays, pts2], point_size=0.02)

def visualizer_transform(p_i, p_j, trans_mat):
    '''

    :param p_i: source
    :param p_j: target
    :param trans_mat: p_i ---> p_j transform matrix
    :return:
    '''

    p_i = np.insert(p_i, obj=3, values=1, axis=1)
    vis_p_i = p_i @ trans_mat.T
    visualize_multiple_pcls(*[p_i, vis_p_i, p_j])


