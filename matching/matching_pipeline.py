import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from matching.export_lines import xml_result
from matching.losses import *
import yaml

from matching.dfs_components import Graph
from sklearn.decomposition import PCA

def matching_main(point_cloud, cuda_card):
    with open('matching/config.yaml', "r") as f:
        config = yaml.safe_load(f)
    all_points = point_cloud['instances'].copy()
    clusters_count  = np.unique(all_points[:,3]).shape[0]

    centers_array = np.zeros((clusters_count, 2))
    vectors_directions = np.zeros((clusters_count, 2))
    new_index = 0
    for i in np.unique(all_points[:, 3]):
        centers_array[new_index, :] = np.mean(all_points[all_points[:, 3] == i, :2],axis=0)
        pca = PCA(n_components=1)
        pca.fit(all_points[all_points[:, 3] == i, :2])
        vectors_directions[new_index] = pca.components_
        new_index +=1
    print(centers_array)
    print(vectors_directions)

    centers_array = torch.tensor(centers_array,dtype=torch.float32)
    optimized_directions = torch.tensor(vectors_directions,dtype=torch.float32)
    print(optimized_directions)
    if torch.cuda.is_available():
        centers_array = centers_array.to(cuda_card)
        optimized_directions = optimized_directions.to(cuda_card)
    optimized_directions = torch.nn.functional.normalize(optimized_directions)
    multiplier = compute_multiplier(centers_array, optimized_directions, config['neighbours_num'])
    #print(multiplier)
    #multiplier = 5
    opposite_pca = compute_opposite_pca(optimized_directions)
    opposite_pca = opposite_pca.to(cuda_card)
    optimized_directions = optimized_directions * opposite_pca[:, None] * multiplier
    optimized_directions.requires_grad = True
    optimizer = torch.optim.SGD([optimized_directions], lr=0.02)
    outreach_mask = find_closest_direction(centers_array, optimized_directions)

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

    chamfer_losses = torch.zeros(config['first_iteration_num']).detach()
    vector_smoothness_losses = torch.zeros(config['first_iteration_num']).detach()
    exclusivity_losses = torch.zeros(config['first_iteration_num']).detach()
    repulsion_losses = torch.zeros(config['first_iteration_num']).detach()
    vectors_directions = torch.zeros(config['first_iteration_num'],clusters_count,2).detach()

    for j in range(config['first_iteration_num']):
        optimizer.zero_grad()
        sum_of_losses = config['LOSSES']['chamfer'] * calculate_chamfer_loss(centers_array, optimized_directions,
                                                                             outreach_mask,device=cuda_card)
        smoothness = config['LOSSES']['smoothness'] * calculate_differentiable_smoothness(centers_array,
                                                                                          optimized_directions,
                                                                                          outreach_mask,device=cuda_card)
        exclusivity = config['LOSSES']['exclusivity'] * vector_exclusivity(centers_array, optimized_directions,
                                                                           outreach_mask,device=cuda_card)
        repulsion = config['LOSSES']['repulsion'] * exclusivity_repulsion(centers_array,
                                                                       optimized_directions, outreach_mask,device=cuda_card)
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

    def animate(i):
        plt.cla()
        x = shorten_centers[:, 0] + vectors_directions[i, :, 0]
        y = shorten_centers[:, 1] + vectors_directions[i, :, 1]
        Q = ax.quiver(shorten_centers[:, 0], shorten_centers[:, 1], vectors_directions[i, :, 0],vectors_directions[i, :, 1],
                      scale=1, color='g', angles='xy', scale_units='xy')
        ax.scatter(x, y, c='b', lw=2)
        x_from = shorten_centers[:, 0]
        y_from = shorten_centers[:, 1]
        ax.scatter(x_from, y_from, c='r', lw=2)
        time_text.set_text('time = %.1d' % i)
        # data = np.stack([x, y]).T
        return Q,
    print("before ANIMATION")
    anim = FuncAnimation(fig, animate, frames=config['first_iteration_num'], interval=200)
    anim.save('animation.gif', writer='ffmpeg', fps=5)
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

    for j in range(config['second_iteration_num']):
        optimizer.zero_grad()
        sum_of_losses = config['LOSSES']['chamfer'] * calculate_chamfer_loss(centers_array,
                                                                             optimized_directions, outreach_mask,device=cuda_card)
        sum_of_losses.backward()
        optimizer.step()

    # to have some return
    point_cloud['matching'] = np.ones(point_cloud['instances'].shape[0])

    g = Graph()
    g.create_graph(centers_array, optimized_directions, outreach_mask)
    # add visualisation of coloured lines
    lines = g.DFS()
    xml_result(lines,centers_array,g)

    fig = plt.figure()
    ax = plt.axes()
    shorten_centers = centers_array.clone().detach().cpu()
    shorten_vectors = optimized_directions.clone().detach().cpu()
    x = shorten_centers[:, 0] +  shorten_vectors[:, 0]
    y = shorten_centers[:, 1] +  shorten_vectors [:, 1]
    Q = ax.quiver(shorten_centers[:, 0], shorten_centers[:, 1], shorten_vectors[:, 0], shorten_vectors[:, 1],
                  scale=1, color='g', angles='xy', scale_units='xy')
    ax.scatter(x, y, c='b', lw=2)
    x_from = shorten_centers[:, 0]
    y_from = shorten_centers[:, 1]
    ax.scatter(x_from, y_from, c='r', lw=2)
    #data = np.stack([x, y]).T
    plt.savefig('matching/lines_final.png')
    torch.cuda.empty_cache()
