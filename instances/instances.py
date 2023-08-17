import torch
import yaml

from instances.instances_clustering import *
from instances.instances_filtering import *


def instances_main(point_cloud_dictionary: dict) -> dict:
    """
    Main function for extracting information about instances from a point cloud.
    This function works on the data provided by the segmentation_main function - points of a point cloud that are
    considered to belong to lane markings. The function divides the points into clusters that are then filtered
    (e.g. by their shape) to remove those that are probably not lane markings. The remaining clusters are then labeled
    as instances. A continuous line should be represented by one instance whereas the dashed line should be represented
    by multiple instances; one for each part line.
    (Warning: The results are dependent on the number of frames in the point cloud)
    Lastly, the function fills the point_cloud_dictionary with the information about instances.
    :param point_cloud_dictionary: a dictionary with the following keys:
            - 'segmentation': a torch tensor of shape (M, 4) = [[x, y, z, frame_id], ...] where M is the number
                                of points in the point cloud.
            - 'instances': a torch tensor of shape (L, 3) = [[x, y, instance_id], ...] where L is the number of points
    :return: the same dictionary with filled 'instances' key
    """
    # Load input data
    segmented_pc = point_cloud_dictionary['segmentation']

    # Get only x, y, z coordinates
    segmented_pc = segmented_pc[:, :3]
    segmented_pc = torch.from_numpy(segmented_pc)

    # Load config
    config = yaml.load(open("instances/instances_config.yaml", "r"), Loader=yaml.FullLoader)

    # Port to GPU
    if torch.cuda.is_available():
        # Select GPU
        gpu = config["cuda"]["gpu"]
        torch.cuda.set_device(gpu)

        # Clear what's on GPU
        torch.cuda.empty_cache()

        # Port to GPU
        device = torch.device("cuda")
        segmented_pc = segmented_pc.to(device)
    else:
        device = torch.device("cpu")

    # Load methods and get clusters
    methods = config["clustering"]["methods"]
    clusters = get_clusters(segmented_pc, methods, device=device, config=config)

    # Make clusters start at 0 and at len(clusters)
    for i, cluster in enumerate(torch.unique(clusters)):
        clusters[clusters == cluster] = i

    # Add the found labels to point cloud
    pc_with_labels = torch.cat([segmented_pc, clusters.reshape(-1, 1)], dim=1).float()

    # Filter point cloud
    filter_mask = filter_pc(pc_with_labels, config=config)
    filtered_pc = pc_with_labels[filter_mask]

    # Save the results to the dictionary
    point_cloud_dictionary['instances'] = filtered_pc

    return point_cloud_dictionary

