import torch
import yaml

from instances.instances_clustering import *
from instances.instances_filtering import *
from instances.instances_utils import *


def instances_main(point_cloud_dictionary: dict, cuda_card: str) -> dict:
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
    :param cuda_card: a string with the name of the GPU card to be used
    :return: the same dictionary with filled 'instances' key
    """
    print("\n\n-------------------------------------------------")
    print("--------------- *INSTANCES MAIN* ----------------")
    print("-------------------------------------------------")

    # Load config
    config = yaml.load(open("instances/instances_config.yaml", "r"), Loader=yaml.FullLoader)
    #config = yaml.load(open("instances_config.yaml", "r"), Loader=yaml.FullLoader)

    # Load input data
    original_segmented_pc = point_cloud_dictionary['segmentation']
    original_segmented_pc = torch.tensor(original_segmented_pc, dtype=torch.float32)
    # Get all provided frames
    frames = torch.unique(original_segmented_pc[:, 3])

    # Get number of provided frames
    number_of_frames = len(frames)

    print("\n\n--------------- FRAME GROUPING ----------------")
    # Find a suitable number of frames in a group
    max_frames = config["data"]["max_frames"]
    min_frames = config["data"]["min_frames"]
    number_of_frames_in_group = get_number_of_frames_in_group(number_of_frames, max_frames=max_frames,
                                                              min_frames=min_frames)

    # Split the point cloud into groups of frames
    frame_groups = []
    for i in range(0, number_of_frames, number_of_frames_in_group):
        upper_bound = i + number_of_frames_in_group
        if upper_bound > number_of_frames:
            upper_bound = number_of_frames
        frame_groups.append(frames[i:upper_bound])

    print(f"- Number of groups: {len(frame_groups)}")
    print(f"- Number of frames in a group: {number_of_frames_in_group}")
    print(f"- Number of frames in the last group: {len(frame_groups[-1])}")

    print("\n\n--------------- INSTANCE CLUSTERING ----------------")
    # Initialize the intergroup variables
    current_cluster_number = 0
    complete_filtered_pc = []

    # Process each group of frames separately
    for i, frame_group in enumerate(frame_groups):
        torch.cuda.empty_cache()
        print(f"- Processing group {i + 1} out of {len(frame_groups)}")
        # Get only points from the current group of frames
        mask = torch.zeros(original_segmented_pc.shape[0], dtype=torch.bool)
        for frame in frame_group:
            mask = mask | (original_segmented_pc[:, 3] == frame)
        segmented_pc = original_segmented_pc[mask]
        print(f"- Number of points in the group: {segmented_pc.shape[0]}")

        # Get only x, y, z coordinates
        segmented_pc = segmented_pc[:, :3]

        # Port to GPU
        if torch.cuda.is_available():
            # Select GPU
            device = cuda_card
            torch.cuda.set_device(device)

            # Clear what's on GPU
            torch.cuda.empty_cache()

            # Port to GPU
            segmented_pc = segmented_pc.to(device)
        else:
            device = torch.device("cpu")

        print("******************* CLUSTERING *******************")
        # Load methods and get clusters
        methods = config["clustering"]["methods"]
        clusters = get_clusters(segmented_pc, methods, device=device, config=config)

        # Make clusters start at 0 and at len(clusters)
        for cluster in torch.unique(clusters):
            clusters[clusters == cluster] = current_cluster_number
            current_cluster_number += 1

        # Add the found labels to point cloud
        pc_with_labels = torch.cat([segmented_pc, clusters.reshape(-1, 1)], dim=1).float()

        print("...finished...")
        print("******************* FILTERING *******************")
        # Filter point cloud
        pc_with_labels.to(device)
        filter_mask = filter_pc(pc_with_labels, config=config)
        filtered_pc = pc_with_labels[filter_mask].cpu()
        complete_filtered_pc.append(filtered_pc)
        print("...finished...")
        print("-----------------------------------------------")
        torch.cuda.empty_cache()

    # Convert the list of tensors to one tensor
    complete_filtered_pc = torch.cat(complete_filtered_pc, dim=0)

    print("\n\n--------------- GROUPING INSTANCES ----------------")
    # Get rid of the z coordinate but keep the instance id
    complete_filtered_pc = complete_filtered_pc[:, [0, 1, 3]]

    # Group instances
    filtered_pc = group_instances(complete_filtered_pc)
    print("...finished...")
    # Save the results to the dictionary
    point_cloud_dictionary['instances'] = filtered_pc.numpy()

    # Release GPU memory
    if torch.cuda.is_available():
        del pc_with_labels
        del filtered_pc
        del complete_filtered_pc
        torch.cuda.empty_cache()

    return point_cloud_dictionary


if __name__ == "__main__":
    process = False

    if process:
        from instances_real_data_loading import *
        name = "test"
        # Load point cloud
        pc = load_real_data_pc(name, start_end=[11, 17], include_frame_id=True)

        # Create dictionary
        pc_dict = {"segmentation": pc}

        # Select GPU
        cuda_card = "cuda:7"

        # Run main function
        pc_dict = instances_main(pc_dict, cuda_card=cuda_card)

        # Save the results
        torch.save(pc_dict, "results/tmp_delete")
    else:
        # Load results from server
        subprocess.run(["python3", "load_results.py"])

        # Load the results
        pc_dict = torch.load("results/tmp_delete")

        # Visualize the results
        pc = pc_dict["instances"]

        import matplotlib.pyplot as plt
        for label in torch.unique(pc[:, 2]):
            mask = pc[:, 2] == label
            cluster = pc[mask]
            center = torch.mean(cluster[:, :2], dim=0)
            plt.scatter(center[0], center[1], c="red", s=10)
            plt.annotate(str(int(label)), (center[0], center[1]))
        plt.scatter(pc[:, 0], pc[:, 1], c=pc[:, 2], s=0.1)
        plt.show()
