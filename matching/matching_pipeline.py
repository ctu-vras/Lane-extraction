import matplotlib.pyplot as plt
import yaml
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA

from matching.dfs_components import Graph
from matching.export_lines import xml_result
from matching.losses import *
import torch

def matching_main(point_cloud, cuda_card, file_name, run_animation):
    # load config with constants
    with open('matching/config.yaml', "r") as f:
        config = yaml.safe_load(f)
    all_points = point_cloud['instances'].copy()
    # shift all instances to be from 0 to number of instances
    clusters_count = np.unique(all_points[:, 2]).shape[0]
    centers_array = np.zeros((clusters_count, 2))
    vectors_directions = np.zeros((clusters_count, 2))
    new_index = 0
    # for each cluster find center and direction using mean and PCA
    for i in np.unique(all_points[:, 2]):
        centers_array[new_index, :] = np.mean(all_points[all_points[:, 2] == i, :2], axis=0)
        pca = PCA(n_components=1)
        pca.fit(all_points[all_points[:, 2] == i, :2])
        vectors_directions[new_index] = pca.components_
        new_index += 1
    #print(centers_array)
    #print(vectors_directions)
    centers_array = torch.tensor(centers_array, dtype=torch.float32)
    optimized_directions = torch.tensor(vectors_directions, dtype=torch.float32)
    # move to cuda if available
    if torch.cuda.is_available():
        centers_array = centers_array.to(cuda_card)
        optimized_directions = optimized_directions.to(cuda_card)
    # normalize vectors
    optimized_directions = torch.nn.functional.normalize(optimized_directions)
    # shift vectors so that all head in the same direction
    opposite_pca = compute_opposite_pca(optimized_directions)
    opposite_pca = opposite_pca.to(cuda_card)
    optimized_directions = optimized_directions * opposite_pca[:, None]
    # for each vector find correct multiplier
    #print(optimized_directions)
    # multiply each vector to have good initial length
    multiply_mask = compute_multiplier(centers_array, optimized_directions, config['neighbours_num']).to(cuda_card)
    optimized_directions = optimized_directions * multiply_mask[:, None]
    #print(optimized_directions)
    # set up learning
    optimized_directions.requires_grad = True
    optimizer = torch.optim.SGD([optimized_directions], lr=0.02)
    # masks last points to not include them in optimization
    outreach_mask = find_closest_direction(centers_array, optimized_directions,device=cuda_card)
    #print(outreach_mask)
    # create image of before learing
    if run_animation:
        fig = plt.figure()
        ax = plt.axes()
        shorten_centers = centers_array.clone().detach().cpu()
        shorten_vectors = optimized_directions.clone().detach().cpu()
        x = shorten_centers[:, 0] + shorten_vectors[:, 0]
        y = shorten_centers[:, 1] + shorten_vectors[:, 1]
        Q = ax.quiver(shorten_centers[:, 0], shorten_centers[:, 1], shorten_vectors[:, 0], shorten_vectors[:, 1],
                      scale=1, color='g', angles='xy', scale_units='xy')
        ax.scatter(x, y, c='b', lw=2)
        x_from = shorten_centers[:, 0]
        y_from = shorten_centers[:, 1]
        ax.scatter(x_from, y_from, c='r', lw=2)
        time_text = ax.text(0.05, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        time_text.set_text('hello')
        # data = np.stack([x, y]).T
        plt.savefig('matching/lines_first.png')
    # create arrays to store losses
    chamfer_losses = torch.zeros(config['first_iteration_num']).detach()
    vector_smoothness_losses = torch.zeros(config['first_iteration_num']).detach()
    exclusivity_losses = torch.zeros(config['first_iteration_num']).detach()
    repulsion_losses = torch.zeros(config['first_iteration_num']).detach()
    vectors_directions = torch.zeros(config['first_iteration_num'], clusters_count, 2).detach()
    # run learning for first iteration
    for j in range(config['first_iteration_num']):
        optimizer.zero_grad()
        sum_of_losses = config['LOSSES']['chamfer'] * calculate_chamfer_loss(centers_array, optimized_directions,
                                                                             outreach_mask, device=cuda_card)
        smoothness = config['LOSSES']['smoothness'] * calculate_differentiable_smoothness(centers_array,
                                                                                          optimized_directions,
                                                                                          outreach_mask,
                                                                                          device=cuda_card)
        exclusivity = config['LOSSES']['exclusivity'] * vector_exclusivity(centers_array, optimized_directions,
                                                                           outreach_mask, device=cuda_card)
        repulsion = config['LOSSES']['repulsion'] * exclusivity_repulsion(centers_array,
                                                                          optimized_directions, outreach_mask,
                                                                          device=cuda_card)
        chamfer_losses[j] = sum_of_losses
        vector_smoothness_losses[j] = smoothness
        exclusivity_losses[j] = exclusivity
        repulsion_losses[j] = repulsion
        vectors_directions[j] = optimized_directions.clone().detach()
        sum_of_losses += smoothness
        sum_of_losses += exclusivity
        sum_of_losses += repulsion
        sum_of_losses.backward()
        optimizer.step()

    # animate learing process
    def animate(i):
        plt.cla()
        x = shorten_centers[:, 0] + vectors_directions[i, :, 0]
        y = shorten_centers[:, 1] + vectors_directions[i, :, 1]
        Q = ax.quiver(shorten_centers[:, 0], shorten_centers[:, 1], vectors_directions[i, :, 0],
                      vectors_directions[i, :, 1],
                      scale=1, color='g', angles='xy', scale_units='xy')
        ax.scatter(x, y, c='b', lw=2)
        x_from = shorten_centers[:, 0]
        y_from = shorten_centers[:, 1]
        ax.scatter(x_from, y_from, c='r', lw=2)
        time_text.set_text('time = %.1d' % i)
        # data = np.stack([x, y]).T
        return Q,

    # save animation and how it looks after first iteration
    if run_animation:
        print("before ANIMATION")
        anim = FuncAnimation(fig, animate, frames=config['first_iteration_num'], interval=200)
        anim.save('animation.gif', writer='pillow', fps=5)
        print("after ANIMATION")
        fig = plt.figure()
        ax = plt.axes()
        shorten_centers = centers_array.clone().detach().cpu()
        shorten_vectors = optimized_directions.clone().detach().cpu()
        x = shorten_centers[:, 0] + shorten_vectors[:, 0]
        y = shorten_centers[:, 1] + shorten_vectors[:, 1]
        Q = ax.quiver(shorten_centers[:, 0], shorten_centers[:, 1], shorten_vectors[:, 0], shorten_vectors[:, 1],
                      scale=1, color='g', angles='xy', scale_units='xy')
        ax.scatter(x, y, c='b', lw=2)
        x_from = shorten_centers[:, 0]
        y_from = shorten_centers[:, 1]
        ax.scatter(x_from, y_from, c='r', lw=2)
        # data = np.stack([x, y]).T
        plt.savefig('matching/lines_middle.png')
    # run second iteration to just get to nearest point
    for j in range(config['second_iteration_num']):
        optimizer.zero_grad()
        sum_of_losses = config['LOSSES']['chamfer'] * calculate_chamfer_loss(centers_array,
                                                                             optimized_directions, outreach_mask,
                                                                             device=cuda_card)
        sum_of_losses.backward()
        optimizer.step()
    # maybe remove doesnot do anything
    # to have some return
    # create graph from connected centers and vectors and run DFS
    g = Graph()
    g.create_graph(centers_array, optimized_directions, outreach_mask)
    # add visualisation of coloured lines
    lines = g.DFS()
    # most important create final xml
    matching_lines = []
    for line in lines:
        point_line = []
        for point in line:
            num_center = centers_array[point].cpu().numpy()
            point_line.append(num_center)
        matching_lines.append(point_line)
    point_cloud['matching'] = matching_lines

    xml_result(lines, centers_array, file_name)
    # save final image
    if run_animation:
        fig = plt.figure()
        ax = plt.axes()
        shorten_centers = centers_array.clone().detach().cpu()
        shorten_vectors = optimized_directions.clone().detach().cpu()
        x = shorten_centers[:, 0] + shorten_vectors[:, 0]
        y = shorten_centers[:, 1] + shorten_vectors[:, 1]
        Q = ax.quiver(shorten_centers[:, 0], shorten_centers[:, 1], shorten_vectors[:, 0], shorten_vectors[:, 1],
                      scale=1, color='g', angles='xy', scale_units='xy')
        ax.scatter(x, y, c='b', lw=2)
        x_from = shorten_centers[:, 0]
        y_from = shorten_centers[:, 1]
        ax.scatter(x_from, y_from, c='r', lw=2)
        # data = np.stack([x, y]).T
        plt.savefig('matching/lines_final.png')
    torch.cuda.empty_cache()
