from matching.losses import *
import yaml

from matching.segmentation import Graph


def matching_main(point_cloud):
    with open('config.yaml', "r") as f:
        config = yaml.safe_load(f)
    centers_array = torch.tensor(point_cloud['instances'].copy())
    # take functions from Honza and create just centers and vectors
    vectors_directions = point_cloud['vectors']
    optimized_directions = torch.tensor(vectors_directions)
    if torch.cuda.is_available():
        optimized_directions = optimized_directions.to('cuda:2')
    optimized_directions = torch.nn.functional.normalize(optimized_directions)
    multiplier = compute_multiplier(centers_array, vectors_directions, config['neighbours_num'])
    opposite_pca = compute_opposite_pca(vectors_directions)
    optimized_directions = optimized_directions * opposite_pca[:, None] * multiplier
    optimized_directions.requires_grad = True
    optimizer = torch.optim.SGD([optimized_directions], lr=0.02)
    outreach_mask = find_closest_direction(centers_array, optimized_directions)

    for j in range(config['first_iteration_num']):
        optimizer.zero_grad()
        sum_of_losses = config['LOSSES']['chamfer'] * calculate_chamfer_loss(centers_array, optimized_directions,
                                                                             outreach_mask)
        smoothness = config['LOSSES']['smoothness'] * calculate_differentiable_smoothness(centers_array,
                                                                                          optimized_directions,
                                                                                          outreach_mask)
        exclusivity = config['LOSSES']['exclusivity'] * vector_exclusivity(centers_array, optimized_directions,
                                                                           outreach_mask)
        repulsion = config['LOSSES']['repulsion'] * exclusivity_repulsion(centers_array,
                                                                          optimized_directions, outreach_mask)

        sum_of_losses += smoothness
        sum_of_losses += exclusivity
        sum_of_losses += repulsion
        sum_of_losses.backward()
        optimizer.step()

    for j in range(config['second_iteration_num']):
        optimizer.zero_grad()
        sum_of_losses = config['LOSSES']['chamfer'] * calculate_chamfer_loss(centers_array,
                                                                             optimized_directions, outreach_mask)
        sum_of_losses.backward()
        optimizer.step()

    # to have some return
    point_cloud['matching'] = np.ones(point_cloud['data'].shape[0])

    g = Graph()
    g.create_graph(centers_array, optimized_directions, outreach_mask)
    # add visualisation of coloured lines
    lines = g.DFS()
    xml_result(lines,centers_array,g)
    torch.cuda.empty_cache()