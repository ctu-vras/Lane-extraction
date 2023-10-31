import numpy as np

from segmentation.plot_segmentation import visualize_points
from segmentation.deploy_model import deploy_model


def segmentation_main(data_dict, config):

    point_cloud = data_dict['data']
    print(f'Segmenting pointcloud with shape: {point_cloud.shape}')

    model_results = deploy_model(
        point_cloud=point_cloud,
        pipeline_config=config
    )



    print('Saving model results')
    if config['MODEL_RETURN_UPSAMPLED']:
        # using upsampled mask --> mask compatible with original pointcloud
        final_mask = model_results.astype(bool)
        data_dict['segmentation_mask'] = final_mask
        data_dict['segmentation'] = point_cloud[final_mask]
    else:
        # dorectly using downsampled output mask from model --> change the pointcloud to the used downsample
        final_mask = model_results[:, -2].astype(bool)
        point_cloud = model_results[:, :4]
        final_mask = np.logical_and(final_mask, point_cloud[:, 3] >= config['POSTPROCESSING_INTENSITY_TRESH'])
        data_dict['data'] = point_cloud
        data_dict['segmentation_mask'] = final_mask
        data_dict['segmentation'] = point_cloud[final_mask]

    if config['ANIMATION']:
        visualize_points(point_cloud, point_cloud[final_mask, :])

if __name__ == '__main__':
    import sys
    from pyntcloud import PyntCloud
    from ruamel.yaml import YAML

    pcd_path = sys.argv[1]
    pipeline_config_path = sys.argv[2]

    # load point cloudd
    #pcd_raw = PyntCloud.from_file(pcd_path)
    #pcd_numpy = pcd_raw.points.to_numpy()
    pcd_numpy = np.load('./../merge_test.npz')['data']
    data_dict = dict()
    data_dict['data'] = pcd_numpy[:, :5]
    print(f"pointcloud loaded {data_dict['data'].shape}")

    # load config
    yaml = YAML()
    yaml.default_flow_style = False
    with open(pipeline_config_path, "r") as f:
        config = yaml.load(f)
    print('config loaded')

    segmentation_main(data_dict, config)

    np.savez('./segmentation_test03.npz', data=data_dict['data'], labels=data_dict['segmentation_mask'])
    print('done')


