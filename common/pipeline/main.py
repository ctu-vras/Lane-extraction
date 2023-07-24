import numpy as np
import pandas as pd
import pickle
import sys
import os
from pyntcloud import PyntCloud
#from segmentation import segmentation_main
#from instances import instances_main
from matching.instance_matching import matching_main


def instances_main(point_cloud):
    #dummy shoud be from import
    pass


def segmentation_main(point_cloud):
    #dummy shoud be from import

    pass


def main():
    #path = sys.argv[1]
    os.chdir("..")
    path = os.path.join(os.path.abspath(os.curdir), "yana's_approach/preprocessing_cpp/new_dataset/2023-01-17_12-15-18_1/projection.pcd")
    point_cloud = {}
    cloud = PyntCloud.from_file(path)
    data = cloud.points
    data_np = data.to_numpy()
    point_cloud['data'] = data_np
    point_cloud['segmentation'] = None
    point_cloud['instances'] = None
    point_cloud['matching'] = None
    point_cloud['labels'] = None
    point_cloud['vectors'] = data_np
    run_segmentation = True
    run_instances = True
    run_matching = True
    print("ok")
    """if run_segmentation:
        try:
            segmentation_main(point_cloud) #data are saved inside point_cloud
            if point_cloud['segmentation'].shape[0] == point_cloud['data'].shape[0]:
                np.save('segmentation.npy', point_cloud['segmentation'])
            else:
                print("Segmentation wrong shape or type")
                point_cloud['segmentation'] = np.load('segmentation.npy')
        except:
            print("Segmentation failed")
            point_cloud['segmentation'] = np.load('segmentation.npy')
    else:
        point_cloud['segmentation'] = np.load('segmentation.npy')
    if run_instances:
        try:
            instances_main(point_cloud)
            if point_cloud['instances'].shape[0] == point_cloud['data'].shape[0] and point_cloud['vectors'] is not None:
                np.save('instances.npy', point_cloud['instances'])
                np.save('vectors.npy', point_cloud['vectors'])
            else:
                print("Instances wrong shape")
                point_cloud['instances'] = np.load('instances.npy')
                point_cloud['vectors'] = np.load('vectors.npy')
        except:
            print("Instances failed")
            point_cloud['instances'] = np.load('instances.npy')
            point_cloud['vectors'] = np.load('vectors.npy')
    else:
        point_cloud['instances'] = np.load('instances.npy')
        point_cloud['vectors'] = np.load('vectors.npy')
    """
    if run_matching:
        try:
            matching_main(point_cloud)
            print(point_cloud['matching'].shape)
            if point_cloud['matching'].shape[0] == point_cloud['data'].shape[0]:
                np.save('matching.npy', point_cloud['matching'])
            else:
                print("Matching wrong shape")
                point_cloud['matching'] = np.load('matching.npy')
        except:
            print("Matching failed")
            point_cloud['matching'] = np.load('matching.npy')
    else:
        point_cloud['matching'] = np.load('matching.npy')
    #not sure how matching and xml should be measured
    return point_cloud

if __name__ == "__main__":
    main()