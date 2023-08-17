import torch

from pytorch3d.ops.knn import knn_points

from instances.instances_plot import *
from instances.instances_data_generating import *
from instances.instances_utils import *

from tqdm import tqdm


def get_clusters(pc: torch.Tensor, loss_names: [str] = None, save: bool = False,
                 visualize: bool = False, device: torch.device = torch.device("cpu"), config: dict = None) -> torch.Tensor:
    """
    Find relevant clusters in the point cloud - differentiable clusterization.
    :param pc: point cloud (format: [batch_size, num_points, 3])
    :param loss_names: list of loss function names
                        - options: ["smooth", "DBSCAN", "HDBSCAN", "small_clusters", "oscillation", "flow", "lock"]
                        - the final loss is weighted sum of all selected losses
    :param save: if True, an animation of the training process is saved
    :param visualize: if True, the training process is visualized
    :param device: device to use for the computation
    :param config: configuration dictionary
    :return: cluster label for each point (format: [batch_size, num_points])
    """
    # Initialize base parameters
    if loss_names is None:
        loss_names = ["smooth", "DBSCAN"]

    # Port pc to the device if necessary
    if pc.device != device:
        pc = pc.to(device)

    # INIT PARAMETERS
    if config is None:
        # Smooth loss parameters
        k = 25
        max_radius = 9

        # DBSCAN parameters
        eps = 1               # 0.15
        min_cluster_size = 2
        min_samples = 1

        # Small clusters loss parameters
        min_points = 5

        # Training parameters
        epochs = 400
        lr = 0.01
    else:
        # Smooth loss parameters
        k = config["clustering"]["k"]
        max_radius = config["clustering"]["max_radius"]

        # DBSCAN parameters
        eps = config["clustering"]["eps"]
        min_cluster_size = config["clustering"]["min_cluster_size"]
        min_samples = config["clustering"]["min_samples"]

        # Small clusters loss parameters
        min_points = config["clustering"]["min_points"]

        # Training parameters
        epochs = config["clustering"]["epochs"]
        lr = config["clustering"]["lr"]

    # Set number of instances to the maximum number of points in the point cloud
    num_instances = pc.shape[-2]

    # Get the ground truth label masks if DBSCAN or HDBSCAN is used
    # + We can much better approximate the number of instances
    if "DBSCAN" in loss_names:
        ground_truth_label_mask_DBSCAN = generate_instance_label_mask(pc.squeeze(), eps=eps/3, min_samples=min_samples,
                                                                      method="DBSCAN", device=device)
        num_instances = ground_truth_label_mask_DBSCAN.shape[-1]

    if "HDBSCAN" in loss_names:
        ground_truth_label_mask_HDBSCAN = generate_instance_label_mask(pc.squeeze(), min_cluster_size=min_cluster_size,
                                                                       min_samples=min_samples, method="HDBSCAN",
                                                                       device=device)
        if "DBSCAN" not in loss_names:
            num_instances = ground_truth_label_mask_HDBSCAN.shape[-1]

    # Initialize mask (probabilities of each point being in a cluster)
    # The mask contains probabilities of each point being in a cluster
    # -> format: [batch_size, num_points, max_instances]
    if len(pc.shape) == 2:
        # If we have a single point cloud
        mask = torch.rand(1, pc.shape[0], num_instances, requires_grad=True, device=device)
        pc = pc.unsqueeze(0)
    else:
        # If we have multiple point clouds
        mask = torch.rand(pc.shape[0], pc.shape[1], num_instances, requires_grad=True, device=device)

    # Setup mask and optimizer
    optimizer = torch.optim.Adam([mask], lr=lr)

    # Initialize parameters for the oscillation loss
    if "smooth" in loss_names:
        # Find the k nearest neighbors of each point in pc
        dist, nn_ind, _ = knn_points(pc, pc, K=k)

        # Remove from neighbours those points that are too far away
        tmp_idx = nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, k)
        nn_ind[dist > max_radius] = tmp_idx[dist > max_radius]

    # Initialize parameters for the oscillation loss
    if "oscillation" in loss_names:
        mask_history = torch.cat((mask.unsqueeze(1), mask.unsqueeze(1)), dim=1)

    # Initialize parameters for the flow and lock loss
    if "flow" in loss_names or "lock" in loss_names:
        stability_threshold = config["clustering"]["stability_threshold"]
        prev_mask = mask.clone()
        stable_iterations_mask = torch.zeros((mask.shape[0], mask.shape[2]), device=device)

    # Initialize visualization variables
    if visualize:
        losses = []
        heat_map = torch.zeros_like(mask[0].detach())
        sequence = None
        loss_labels = ["Loss", "Smooth", "DBSCAN", "HDBSCAN", "Small", "Oscillation", "Flow", "Lock"]

    # Initialize the losses
    smooth_loss = torch.tensor(0.0).to(device)
    DBSCAN_loss = torch.tensor(0.0).to(device)
    HDBSCAN_loss = torch.tensor(0.0).to(device)
    small_clusters_loss = torch.tensor(0.0).to(device)
    oscillation_loss = torch.tensor(0.0).to(device)
    flow_loss = torch.tensor(0.0).to(device)
    lock_loss = torch.tensor(0.0).to(device)

    # Iteratively update the mask
    for i in tqdm(range(epochs)):
        if "smooth" in loss_names:
            smooth_loss = calculate_smooth_loss(mask, nn_ind)

        if "DBSCAN" in loss_names:
            DBSCAN_loss = calculate_DBSCAN_loss(mask, ground_truth_label_mask_DBSCAN)

        if "HDBSCAN" in loss_names:
            HDBSCAN_loss = calculate_HDBSCAN_loss(mask, ground_truth_label_mask_HDBSCAN)

        if "small_clusters" in loss_names:
            small_clusters_loss = calculate_small_clusters_loss(mask, min_points)

        if "oscillation" in loss_names:
            # Get the masks [first -> second -> third - current]
            second_mask = mask_history[:, 1]
            third_mask = mask

            # Get the indices of the points that has changed their cluster number from the second to the third mask
            changed_indices = torch.argmax(second_mask, dim=2) != torch.argmax(third_mask, dim=2)

            # Calculate the oscillation loss
            oscillation_loss = calculate_oscillation_loss(mask, mask_history)

            # Update the mask history
            mask_history[changed_indices, 0] = second_mask[changed_indices]
            mask_history[changed_indices, 1] = third_mask[changed_indices]

        if "flow" in loss_names and i > epochs//2:
            flow_loss = calculate_flow_loss(mask, prev_mask)
            prev_mask = mask.clone()

        if "lock" in loss_names and i != 0:
            lock_loss = calculate_lock_loss(mask, prev_mask, stable_iterations_mask, stability_threshold)
            prev_mask = mask.clone()

        # ---------------- ADD ALL LOSSES TOGETHER ----------------
        # TODO: ADJUST HERE
        # Define scalar that linearly decreases from 1 to 0 over all epochs
        factor = 5
        b = i/epochs
        a = factor * b
        a = a if a < 1 else 1

        # Calculate the final loss as weighted sum of the smooth loss and the ground truth loss
        if "smooth" in loss_names and len(loss_names) == 1:
            loss = smooth_loss
        elif "DBSCAN" in loss_names and len(loss_names) == 1:
            loss = DBSCAN_loss
        elif "HDBSCAN" in loss_names and len(loss_names) == 1:
            loss = HDBSCAN_loss
        elif "small_clusters" in loss_names and len(loss_names) == 1:
            loss = small_clusters_loss
        elif "oscillation" in loss_names and len(loss_names) == 1:
            loss = oscillation_loss
        elif "flow" in loss_names and len(loss_names) == 1:
            loss = flow_loss
        elif "lock" in loss_names and len(loss_names) == 1:
            loss = lock_loss
        else:
            loss = a*(smooth_loss+small_clusters_loss+oscillation_loss) \
                   + (1-a)*(DBSCAN_loss + HDBSCAN_loss)+b*(flow_loss+lock_loss)

        # ---------------- SAVE ALL LOSSES ----------------
        if visualize:
            print("Loss: ", loss.item())
            loss_array = [loss.item(), smooth_loss.item(), DBSCAN_loss.item(), HDBSCAN_loss.item(),
                          small_clusters_loss.item(), oscillation_loss.item(), flow_loss.item(), lock_loss.item()]
            losses.append(loss_array)

        # ---------------- BACK-PROPAGATE GRADIENTS ----------------
        loss.backward()

        # ---------------- VISUALIZE ----------------
        if visualize:
            # TODO: ADJUST HERE
            # Show the heat/gradient/gumbel map of the mask
            empty_map = torch.zeros_like(mask[0])
            grad_map = mask.grad[0]
            mask_vis = mask[0].detach()
            gumbel = my_gumbel_softmax(mask[0].detach(), hard=True)

            # Plot each epoch
            tmp_instances = mask.argmax(dim=2).detach()
            pc_plot = torch.cat((pc[0], tmp_instances[0].unsqueeze(1)), dim=1)

            if i % (epochs // 10) == 0:
                heat_map = torch.cat((heat_map.detach(), empty_map, mask_vis), dim=1)
                # plot_point_cloud(pc_plot, num_clusters=num_instances, title="Differentiable Clusterization")

            if sequence is None:
                sequence = pc_plot.unsqueeze(0)
            else:
                sequence = torch.cat((sequence, pc_plot.unsqueeze(0)), dim=0)

        # ---------------- OPTIMIZE ----------------
        optimizer.step()
        optimizer.zero_grad()

    # Visualize
    if visualize:
        title = f"Differentiable Clusterization"
        # Add the used loss names to the title
        title += " -"
        for l in loss_names:
            print(l, end=" ")
            title += f" {l} |"
        title += "\n"

        animate(sequence, num_clusters=num_instances, title=title, save=save)
        plot_heat_map(heat_map)
        plot_loss(losses, loss_labels)

    # Make clusters start at 0 and at len(clusters)
    clusters = mask.argmax(dim=2)
    for i, cluster in enumerate(torch.unique(clusters)):
        clusters[clusters == cluster] = i

    return clusters


def calculate_smooth_loss(mask: torch.Tensor, nn_ind: torch.Tensor) -> torch.Tensor:
    """
    Calculate the smooth loss. (https://arxiv.org/pdf/2210.04458.pdf - page 4)
    :param mask: mask containing probabilities of each point being in a cluster
                    - format: [batch_size, num_points, max_clusters]
    :param nn_ind: indices of the k nearest neighbors of each point in pc
                    - format: [batch_size, num_points, k]
    :return: smooth loss
    """
    # Get the neighbor labels
    labels = mask[0][nn_ind[0]]

    # The labels contain the probabilities of the k nearest neighbors of each point in pc being in a cluster
    # -> format: [num_points, k, max_clusters]
    # We need to permute the dimensions to get the correct format and add a batch dimension
    # -> format: [batch_size, max_clusters, num_points, k]
    labels = labels.permute(0, 2, 1)
    labels = labels.unsqueeze(0)

    # According to the paper, we need to implement a smooth loss. The loss should be small if the probabilities
    # of the k nearest neighbors of each point in pc being in a cluster are similar and large otherwise.
    # The mathematical formula for the smooth loss is: l = 1/N * sum_i^N(1/H * sum_j^H(d(x_i, x_j^k))), where
    # N is the number of points in pc, H is the number of nearest neighbors, d is the euclidean distance and
    # x_j^k is the j-th nearest neighbor of the i-th point in pc.
    # The first problem with this loss is that it does not penalize the case when due to the initialization of the
    # mask two separate clusters are assigned the same label. The second problem is that some points in one cluster
    # tend to oscillate between two different clusters.
    smooth_loss = (mask.unsqueeze(3) - labels).norm(p=2, dim=2).mean()

    return smooth_loss


def calculate_DBSCAN_loss(mask: torch.Tensor, ground_truth_label_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate the DBSCAN loss.
    :param mask: mask containing probabilities of each point being in a cluster
                    - format: [batch_size, num_points, num_clusters]
    :param ground_truth_label_mask: ground truth label mask generated by DBSCAN
                    - format: [batch_size, num_points, num_clusters]
    :return: DBSCAN loss
    """
    # There is ofcourse a problem with the ground truth labeling, because it really does not matter
    # if we name a cluster 1 or 11. It turns out that it really is not a problem.
    ground_truth_loss = (mask[0] - ground_truth_label_mask[0]).norm(p=2, dim=1).mean()

    return ground_truth_loss


def calculate_HDBSCAN_loss(mask: torch.Tensor, ground_truth_label_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate the HDBSCAN loss. (Hierarchical DBSCAN)
    :param mask: mask containing probabilities of each point being in a cluster
                    - format: [batch_size, num_points, num_clusters]
    :param ground_truth_label_mask: ground truth label mask generated by HDBSCAN
                    - format: [batch_size, num_points, num_clusters]
    :return: HDBSCAN loss
    """
    ground_truth_loss = (mask[0] - ground_truth_label_mask[0]).norm(p=2, dim=1).mean()

    return ground_truth_loss


def calculate_small_clusters_loss(mask: torch.Tensor, min_points: int) -> torch.Tensor:
    """
    Calculate the small clusters loss. Penalize for having clusters with less than min_points points.
    :param mask: mask containing probabilities of each point being in a cluster
                    - format: [batch_size, num_points, num_clusters]
    :param min_points: minimum number of points in a cluster not to be considered small
    :return: small cluster loss
    """
    # Get number of points in each cluster
    unique, count = torch.unique(mask.argmax(dim=2), return_counts=True)

    # Calculate the small cluster loss
    small_clusters = unique[count < min_points]
    small_cluster_loss = (mask[0, :, small_clusters] ** 2).mean()

    # Make it zero if there are no small clusters
    if torch.isnan(small_cluster_loss):
        small_cluster_loss = torch.tensor(0.0).to(mask.device)

    return small_cluster_loss


def calculate_oscillation_loss(mask: torch.Tensor, mask_history: torch.Tensor) -> torch.Tensor:
    """
    Calculate the oscillation loss. Penalize for a points oscillating between two clusters. A point is considered
    to be oscillating if it changes its instance from one instance to another and back to the first instance (not ).
    In that case, we will try to push the point to the instance that contains more points.
    :param mask: mask containing probabilities of each point being in a cluster
                    - format: [batch_size, num_points, num_clusters]
    :param mask_history: mask history containing the masks from the two previous iterations
                    - format: [batch_size, 2, num_points, num_clusters]
    :return: oscillation loss
    """
    # Get the masks [first -> second -> third]
    first_mask = mask_history[:, 0]
    second_mask = mask_history[:, 1]
    third_mask = mask

    # Get the indices of the points that are oscillating
    oscillating_indices = (first_mask.argmax(dim=2) != second_mask.argmax(dim=2)) & \
                          (first_mask.argmax(dim=2) == third_mask.argmax(dim=2))

    non_oscillating_indices = ~oscillating_indices

    # Get the clusters of the points that are oscillating
    oscillating_clusters_first = first_mask.argmax(dim=2)[oscillating_indices]
    oscillating_clusters_second = second_mask.argmax(dim=2)[oscillating_indices]

    # Get the number of points in each instance in the current mask
    first_clusters_number = third_mask.argmax(dim=2)[:, oscillating_clusters_first]
    second_clusters_number = third_mask.argmax(dim=2)[:, oscillating_clusters_second]

    # From the two clusters for each point, choose the one that contains more points in the current mask
    idx = first_clusters_number < second_clusters_number
    idx = idx[0]

    oscillating_clusters_first[idx] = oscillating_clusters_second[idx]

    # Create desired mask - for non-oscillating points, the desired mask is the same as the current mask and for
    # oscillating points, the desired mask has ones for the bigger instance and zeros for all the other clusters
    # TODO - IDEA: maybe we should assign the instance that is smaller (from the two oscillating clusters) a negative
    #  number to accelerate the convergence?
    desired_mask = torch.zeros_like(mask, device=mask.device)
    desired_mask[oscillating_indices, oscillating_clusters_first] = 1
    desired_mask[non_oscillating_indices] = mask[non_oscillating_indices]

    # Calculate the oscillation loss (L2 norm between the current mask and the desired mask)
    oscillation_loss = (mask - desired_mask).norm(p=2, dim=2).mean()

    return oscillation_loss


def calculate_flow_loss(mask: torch.Tensor, previous_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate the flow loss. In the later iterations, we want to know which clusters are stable and which are still
    'fighting' over some points. Those that are stable, we want to fix and those that are still changing, we want to
    either boost (bigger ones) or destroy (smaller ones).
    :param mask: mask containing probabilities of each point being in a cluster
                    - format: [batch_size, num_points, num_clusters]
    :param previous_mask: mask from the previous iteration
                    - format: [batch_size, num_points, num_clusters]
    :return: flow loss
    """
    # Get indices of points that are in clusters that are not stable (they do not contain the same points as in the
    # previous iteration)
    # - Get the indices of the points that are not stable
    unstable_points_indices = (mask.argmax(dim=2) != previous_mask.argmax(dim=2))

    if torch.any(unstable_points_indices):
        # - Get clusters of the points that are not stable from both the current and the previous mask
        prev_mask_unstable_clusters = previous_mask.argmax(dim=2)[unstable_points_indices]
        curr_mask_unstable_clusters = mask.argmax(dim=2)[unstable_points_indices]

        # - For each point in the unstable clusters, get the number of points in the cluster in the current mask
        prev_mask_unstable_clusters_points_number = (mask.argmax(dim=2)[0].unsqueeze(0) == prev_mask_unstable_clusters
                                                     .unsqueeze(1)).sum(dim=1)
        curr_mask_unstable_clusters_points_number = (mask.argmax(dim=2)[0].unsqueeze(0) == curr_mask_unstable_clusters
                                                     .unsqueeze(1)).sum(dim=1)

        # Get the ones with bigger number
        idx = prev_mask_unstable_clusters_points_number < curr_mask_unstable_clusters_points_number

        # Get the clusters that are not stable
        unstable_clusters_bigger = torch.zeros_like(prev_mask_unstable_clusters, device=mask.device)
        unstable_clusters_bigger[idx] = curr_mask_unstable_clusters[idx]
        unstable_clusters_bigger[~idx] = prev_mask_unstable_clusters[~idx]

        unstable_clusters_smaller = torch.zeros_like(prev_mask_unstable_clusters, device=mask.device)
        unstable_clusters_smaller[~idx] = curr_mask_unstable_clusters[~idx]
        unstable_clusters_smaller[idx] = prev_mask_unstable_clusters[idx]

        # Get points that are in the unstable bigger/smaller clusters in the current mask
        tmp_mask_bigger = torch.eq(mask.argmax(dim=2).unsqueeze(-1), unstable_clusters_bigger)
        tmp_mask_smaller = torch.eq(mask.argmax(dim=2).unsqueeze(-1), unstable_clusters_smaller)
        unstable_points_indices_bigger = torch.any(tmp_mask_bigger, dim=-1).nonzero(as_tuple=False)[:, 1]
        unstable_points_indices_smaller = torch.any(tmp_mask_smaller, dim=-1).nonzero(as_tuple=False)[:, 1]

        desired_mask = mask.clone()

        for i in range(unstable_points_indices_bigger.shape[0]):
            desired_mask[:, unstable_points_indices_bigger[i], unstable_clusters_bigger] = 1
            desired_mask[:, unstable_points_indices_bigger[i], unstable_clusters_smaller] = 0

        for i in range(unstable_points_indices_smaller.shape[0]):
            desired_mask[:, unstable_points_indices_smaller[i], unstable_clusters_smaller] = 0
            desired_mask[:, unstable_points_indices_smaller[i], unstable_clusters_bigger] = 1

        # Calculate the flow loss (L2 norm between the current mask and the desired mask)
        flow_loss = (mask - desired_mask).norm(p=2, dim=2).mean()
    else:
        flow_loss = torch.tensor(0.0)

    return flow_loss


def calculate_lock_loss(mask: torch.Tensor, previous_mask: torch.Tensor, stable_iterations_mask: torch.Tensor,
                        stability_threshold: int = 20) -> torch.Tensor:
    """
    Calculate the lock loss. In the later iterations, we want to keep those clusters that are stable for some time.
    Stable clusters are those that contain the same points for some number of iterations.
    :param mask: mask containing probabilities of each point being in a cluster
                    - format: [batch_size, num_points, num_clusters]
    :param previous_mask: mask from the previous iteration
                    - format: [batch_size, num_points, num_clusters]
    :param stable_iterations_mask: mask containing the number of iterations that the clusters are stable
                    - format: [batch_size, num_clusters]
    :param stability_threshold: number of iterations that the clusters need to contain the same points to be considered
                                stable
    :return: lock loss
    """
    # For each point, get its cluster in the current and previous mask
    current_clusters_per_point = mask.argmax(dim=2)
    previous_clusters_per_point = previous_mask.argmax(dim=2)

    # For each point, find out if it changed its cluster from the previous mask to the current mask (unstable points)
    unstable_points_mask = (current_clusters_per_point != previous_clusters_per_point)

    # Get clusters that are not stable from the previous mask to the current mask
    unstable_clusters_per_point = current_clusters_per_point[unstable_points_mask]
    unstable_clusters = torch.unique(unstable_clusters_per_point)

    # Get dead clusters (clusters that are not present in the current mask)
    all_current_clusters = torch.arange(mask.shape[2], device=mask.device)
    all_present_clusters = torch.unique(current_clusters_per_point)
    dead_clusters = all_current_clusters[~torch.isin(all_current_clusters, all_present_clusters)]

    # Adjust the stable iterations mask (add one if the cluster is stable, otherwise [unstable or dead] set to zero)
    stable_iterations_mask += 1
    stable_iterations_mask[:, unstable_clusters] = 0
    stable_iterations_mask[:, dead_clusters] = 0

    # Get clusters that are stable long enough
    stable_clusters = all_current_clusters[(stable_iterations_mask >= stability_threshold)[0]]

    if stable_clusters.shape[0] > 0:
        # Get indices of the points that are in the stable clusters
        stable_points_indices = (current_clusters_per_point.unsqueeze(-1) == stable_clusters).nonzero()[:, 1]

        # Get indices of the clusters the points are in
        stable_clusters_indices_per_point = current_clusters_per_point[:, stable_points_indices]

        # Set the desired mask
        desired_mask = mask.clone()
        desired_mask[:, stable_points_indices, stable_clusters_indices_per_point] = 1

        # Calculate the lock loss (L2 norm between the current mask and the desired mask)
        lock_loss = (mask - desired_mask).norm(p=2, dim=2).mean()
    else:
        lock_loss = torch.tensor(0.0)

    return lock_loss


if __name__ == "__main__":
    prev_mask = torch.zeros((1, 4, 3))
    mask = torch.zeros((1, 4, 3))
    mask[0, 0, 0] = 1
    mask[0, 0, 1] = 0
    mask[0, 0, 2] = 0
    mask[0, 1, 0] = 0
    mask[0, 1, 1] = 1
    mask[0, 1, 2] = 0
    mask[0, 2, 0] = 0
    mask[0, 2, 1] = 0
    mask[0, 2, 2] = 1
    mask[0, 3, 0] = 1
    mask[0, 3, 1] = 0
    mask[0, 3, 2] = 0

    prev_mask[0, 0, 0] = 0
    prev_mask[0, 0, 1] = 0
    prev_mask[0, 0, 2] = 1
    prev_mask[0, 1, 0] = 0
    prev_mask[0, 1, 1] = 1
    prev_mask[0, 1, 2] = 0
    prev_mask[0, 2, 0] = 0
    prev_mask[0, 2, 1] = 0
    prev_mask[0, 2, 2] = 1
    prev_mask[0, 3, 0] = 0
    prev_mask[0, 3, 1] = 1
    prev_mask[0, 3, 2] = 0

    print("Mask")
    print(mask)
    print("Previous mask")
    print(prev_mask)
    calculate_flow_loss(mask, prev_mask)


