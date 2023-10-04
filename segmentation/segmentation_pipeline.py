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

    # TODO postprocessing

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


