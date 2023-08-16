import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.animation import FuncAnimation


def plot_point_cloud(pc: torch.tensor, principal_components: torch.tensor = None, centers: torch.tensor = None,
                     point: torch.tensor = None, neighbours: torch.tensor = None, num_clusters: int = None,
                     num_points_cluster: int = 0, title: str = None, save: bool = False) -> None:
    """
    Plot point cloud data.
    :param pc: point cloud data
    :param principal_components: principal components of each cluster (instance)
    :param centers: centers of each cluster (instance)
    :param point: point to be highlighted
    :param neighbours: neighbours of the highlighted point
    :param num_clusters: number of clusters
    :param num_points_cluster: maximum number of points in one cluster
    :param title: title of the plot
    :param save: if True, the plot will be saved
    :return: None
    """
    # Plot point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Check if the point cloud has labels (if it has, plot it with color)
    if pc.shape[1] == 4:
        colors = pc[:, 3]
        num_clusters = colors.max().int().item() if num_clusters is None else num_clusters - 1
    else:
        colors = 'b'
        num_clusters = 1 if num_clusters is None else num_clusters - 1

    # Make alpha parameter dependent on the number of points linearly (from 1 to 0.05)
    alpha = 1 - num_points_cluster / 400 * 0.95 + 0.05
    alpha = 1 if alpha > 1 else alpha
    alpha = 0.05 if alpha < 0.05 else alpha

    # Fill the plot
    cb = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=colors, s=0.01, cmap='jet', alpha=alpha, vmin=0, vmax=num_clusters)

    # Find limits for the axes
    x_lower = torch.min(pc[:, 0])
    x_upper = torch.max(pc[:, 0])
    y_lower = torch.min(pc[:, 1])
    y_upper = torch.max(pc[:, 1])
    z_lower = torch.min(pc[:, 2])
    z_upper = torch.max(pc[:, 2])

    # # Create 10 percent margin
    x_margin = (x_upper - x_lower) * 0.1
    y_margin = (y_upper - y_lower) * 0.1
    z_margin = (z_upper - z_lower) * 0.1

    # Find maximum and set it as the limit for all axes
    max_range = np.array([x_upper - x_lower, y_upper - y_lower, z_upper - z_lower]).max() / 2.0

    # Calculate mid points for all axes
    mid_x = (x_upper + x_lower) * 0.5
    mid_y = (y_upper + y_lower) * 0.5
    mid_z = (z_upper + z_lower) * 0.5

    # Set limits for the axes
    ax.set_xlim(mid_x - max_range - x_margin, mid_x + max_range + x_margin)
    ax.set_ylim(mid_y - max_range - y_margin, mid_y + max_range + y_margin)
    ax.set_zlim(mid_z - max_range - z_margin, mid_z + max_range + z_margin)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if principal_components is not None:
        for i, princ_comp in enumerate(principal_components):
            # Select cluster
            cluster = pc[pc[:, 3] == i]

            # Get rid of the label
            cluster = cluster[:, :3]

            # Find the center of mass of the cluster
            center = centers[i]

            # Plot the principal components as an arrow starting from the center of mass
            ax.quiver(center[0], center[1], center[2], princ_comp[0], princ_comp[1], princ_comp[2], length=1, color='r')

    if point is not None and neighbours is not None:
        # Plot the point
        ax.scatter(point[0], point[1], point[2], c='r', s=1)

        # Plot the neighbours
        ax.scatter(neighbours[:, 0], neighbours[:, 1], neighbours[:, 2], c='g', s=1)

    # Set title
    if title is not None:
        plt.title(title)

    # Save plot
    if save:
        plt.savefig("img/" + title + '.png', dpi=300)

    # Set color bar
    plt.colorbar(cb)

    # Show plot
    plt.show()

    return


def plot_loss(losses: [list], loss_labels: [str] = None) -> None:
    """
    Plot losses.
    :param losses: list of losses
    :param loss_labels: list of labels for each loss
    :return: None
    """
    # If any loss is empty, remove it
    losses = torch.tensor(losses)
    relevant = torch.any(losses != 0, dim=0)
    losses = losses[:, relevant]

    # Also remove its label
    for i, label in enumerate(loss_labels):
        if not relevant[i]:
            loss_labels.remove(label)

    # Plot losses
    plt.plot(losses)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.legend(loss_labels)
    plt.show()


def plot_vectors(vectors: list) -> None:
    """
    Plot vectors as arrows.
    :param vectors: list of vectors
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']

    for i, vector in enumerate(vectors):
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], length=1, color=colors[i % len(colors)])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    plt.show()


def plot_vector_duos(first_vectors: np.ndarray, second_vectors: np.ndarray) -> None:
    """
    Plot duos of vectors as arrows.
    :param first_vectors: first list of vectors
    :param second_vectors: second list of vectors
    :return: None
    """

    if len(first_vectors) != len(second_vectors):
        raise ValueError('Length of first_vectors and second_vectors must be equal.')

    fig = plt.figure()

    # Get number of subplots
    num_subplots = first_vectors.shape[0]

    # Get number of rows and columns
    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    # Create subplots
    for i in range(num_subplots):
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection='3d')

        ax.quiver(0, 0, 0, first_vectors[i, 0], first_vectors[i, 1], first_vectors[i, 2], length=1, color='r')
        ax.quiver(0, 0, 0, second_vectors[i, 0], second_vectors[i, 1], second_vectors[i, 2], length=1, color='g')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

    plt.show()


def plot_heat_map(array: torch.Tensor) -> None:
    """
    Plot heat map of the array.
    :param array: array
    :return: None
    """
    plt.imshow(array, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


def animate(sequence: torch.tensor, num_clusters: int = None, num_points_cluster: int = 0, title: str = ".",
            save: bool = False) -> None:
    """
    Animate sequence of point clouds.
    :param sequence: sequence of point clouds
    :param num_clusters: number of clusters
    :param num_points_cluster: number of points in a cluster
    :param title: title of the animation
    :param save: save animation
    :return: None
    """
    # Convert sequence
    sequence = sequence.detach().cpu()

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Check if the point cloud has labels (if it has, plot it with color)
    if sequence.shape[2] == 4:
        colors = sequence[0, :, 3]
        num_clusters = int(torch.max(colors).item()) if num_clusters is None else num_clusters - 1
    else:
        colors = 'b'
        num_clusters = 1 if num_clusters is None else num_clusters - 1

    # Make alpha parameter dependent on the number of points linearly (from 1 to 0.05)
    alpha = 1 - num_points_cluster / 400 * 0.95 + 0.05
    alpha = 1 if alpha > 1 else alpha
    alpha = 0.05 if alpha < 0.05 else alpha

    # Get the reference to the color bar + write next to it the number of clusters
    cbar = ax.scatter(sequence[0, :, 0], sequence[0, :, 1], sequence[0, :, 2], c=colors, s=0.1, cmap="jet", alpha=alpha,
                      vmin=0, vmax=num_clusters)

    def update(i):
        # Clear the plot
        ax.clear()
        num_instances = 1

        # Check if the point cloud has labels (if it has, plot it with color)
        if sequence.shape[2] == 4:
            colors = sequence[i, :, 3]
            num_instances = int(torch.unique(colors).shape[0])
        else:
            colors = 'b'

        # Fill the plot
        ax.scatter(sequence[0, :, 0], sequence[0, :, 1], sequence[0, :, 2], c=colors, s=1, cmap="jet", alpha=alpha,
                   vmin=0, vmax=num_clusters)

        # Find limits for the axes
        x_lower = torch.min(sequence[i, :, 0])
        x_upper = torch.max(sequence[i, :, 0])
        y_lower = torch.min(sequence[i, :, 1])
        y_upper = torch.max(sequence[i, :, 1])
        z_lower = torch.min(sequence[i, :, 2])
        z_upper = torch.max(sequence[i, :, 2])

        # # Create 10 percent margin
        x_margin = (x_upper - x_lower) * 0.1
        y_margin = (y_upper - y_lower) * 0.1
        z_margin = (z_upper - z_lower) * 0.1

        # Find maximum and set it as the limit for all axes
        max_range = np.array([x_upper - x_lower, y_upper - y_lower, z_upper - z_lower]).max() / 2.0

        # Calculate mid points for all axes
        mid_x = (x_upper + x_lower) * 0.5
        mid_y = (y_upper + y_lower) * 0.5
        mid_z = (z_upper + z_lower) * 0.5

        # Set limits for the axes
        ax.set_xlim(mid_x - max_range - x_margin, mid_x + max_range + x_margin)
        ax.set_ylim(mid_y - max_range - y_margin, mid_y + max_range + y_margin)
        ax.set_zlim(mid_z - max_range - z_margin, mid_z + max_range + z_margin)

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title(f"{title} {str(i)} | instances: {num_instances}")

    # Set color bar
    fig.colorbar(cbar)

    # Create animation
    ani = FuncAnimation(fig, update, frames=sequence.shape[0], interval=100)

    # Show animation
    plt.show()

    if save:
        ani.save("img/" + title + ' k_12_mr_0.5 | eps_1.gif', writer='imagemagick', fps=10)
