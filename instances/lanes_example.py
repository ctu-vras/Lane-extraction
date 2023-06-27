import numpy as np
import torch
from pytorch3d.ops.knn import knn_points

# from vis import *

# synthetize 3d point based on 2D bounding box and number of points
def create_dash_line_3D(center, dim_ratio=np.array([1.2,0.4,0.05]), N=100):

    df = np.random.uniform(-1,1,(N,3)) * dim_ratio + center

    return df

center = np.array([10,5,0.1])
pcl = []

for i in range(10):
    # increase center-x for each line to create a dash line
    center[0] += 2

    line_points = create_dash_line_3D(center, dim_ratio=np.array([0.6,0.2,0.02]), N=100 - i * 4)
    line_points = np.insert(line_points, 3, i, axis=1)
    pcl.append(line_points)

# point cloud data
pcl = np.concatenate(pcl, axis=0)


x = torch.from_numpy(pcl).unsqueeze(0).float()  # points
y = torch.from_numpy(pcl[:,3]).unsqueeze(0) # labels
K = 12  # KNN
max_radius = 1 # max radius for KNN

# Model of predictions
class InstanceSegRefiner(torch.nn.Module):

    def __init__(self, x, max_instances=30):
        self.pred = torch.rand(x.shape[0], x.shape[1], max_instances, requires_grad=True)
        super().__init__()

    def forward(self):
        return torch.softmax(self.pred, dim=2)

# To split points based on IDs
def mask_split(tensor, indices):
    unique = torch.unique(indices)
    return [tensor[0][indices == i] for i in unique]

def svd_tensor_list(tensor_list):

    U_list = []
    S_list = []
    V_list = []
    for tensor in tensor_list:
        tensor_center = tensor - tensor.mean(dim=0)
        U, S, V = torch.svd(tensor_center)
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)
    return U_list, torch.stack(S_list), torch.stack(V_list)

model = InstanceSegRefiner(x)

# Calculate KNN indices
dist, nn_ind, _ = knn_points(x, x, K=K)
tmp_idx = nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, K).to(nn_ind.device)
nn_ind[dist > max_radius] = tmp_idx[dist > max_radius]  # rewrite the indices that are too far away

mask = model.pred
min_nbr_pts = 5
loss_norm = 1
# Build optimizer to gradiently update mask based on minimizing the loss function
optimizer = torch.optim.Adam([mask], lr=0.1)

# ~ diffentiable DBSCAN
# min distance from clusters

for e in range(1000):
    # still not batch-wise
    out = mask[0][nn_ind[0]]
    out = out.permute(0, 2, 1)
    out = out.unsqueeze(0)

    # norm for each of N separately
    smooth_loss = (mask.unsqueeze(3) - out).norm(p=loss_norm, dim=2).mean()

    u, c = torch.unique(mask.argmax(dim=2), return_counts=True)

    # If cluster is too small in terms of number of points, supress id there (minimize probability)
    small_clusters = u[c < min_nbr_pts]
    small_cluster_loss = (mask[0, :, small_clusters] ** 2).mean()

    # Calculate loss
    loss = smooth_loss + small_cluster_loss.mean() # + 0.1 * pseudo_xe_loss #

    # Backpropagate
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"{e} ---> {smooth_loss.item()}")

# final ids is maximum probability of mask per point
predicted_mask = mask.argmax(dim=2).detach().cpu().numpy()

# Visualize
from mayavi import mlab

figure = mlab.figure(1, bgcolor=(1, 1, 1), size=(640, 480))
# nodes = mlab.points3d(x[0, :, 0], x[0, :, 1], x[0, :, 2], y[0], scale_factor=0.5, scale_mode='none', colormap='jet')
nodes = mlab.points3d(x[0, :, 0], x[0, :, 1], x[0, :, 2], predicted_mask[0], scale_factor=0.1, scale_mode='none', colormap='jet')
mlab.show()

figure = mlab.figure(1, bgcolor=(1, 1, 1), size=(640, 480))

for idx in torch.unique(y):
    A = x[0][y[0] == idx]
    A_centered = A - A.mean(dim=0)
    U, S, V = torch.svd(A_centered)


    vis_pc = A.detach().cpu().numpy()
    vis_eigen_vectors = V.detach().cpu().numpy().copy()
    vis_eigen_vectors[0, :] = S[0] * vis_eigen_vectors[0, :]
    vis_eigen_vectors[1, :] = S[1] * vis_eigen_vectors[1, :]
    vis_eigen_vectors[2, :] = S[2] * vis_eigen_vectors[2, :]


    mlab.points3d(0, 0, 0, color=(0, 0, 1), scale_factor=0.3, mode='axes')
    mlab.points3d(A[:,0], A[:,1], A[:,2], color=(0,0,1), scale_factor=0.1)
    mlab.quiver3d(A[:,0].mean().repeat(3), A[:,1].mean().repeat(3), A[:,2].mean().repeat(3), vis_eigen_vectors[:,0], vis_eigen_vectors[:,1], vis_eigen_vectors[:,2], color=(0,1,0), scale_factor=1)

mlab.show()
