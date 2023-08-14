import numpy as np
import pandas as pd
import pickle
import sys
import os
from pyntcloud import PyntCloud
from segmentation.segmentation_pipeline import segmentation_main
# from instances import instances_main
from matching.instance_matching import matching_main


def instances_main(point_cloud):
    # dummy shoud be from import
    pass


def main():
    # path = sys.argv[1]
    # server and load data accordingly
    # config file to set what to load

    path = os.path.join(os.path.abspath(os.curdir),
                        #"yana's_approach/preprocessing_cpp/new_dataset/2023-01-17_12-15-18_1/projection.pcd")
                        "common/data/LR1_local-001.pcd")

    print(path)
    point_cloud = {}
    print(os.path.exists(path))
    cloud = PyntCloud.from_file(path)
    data = cloud.points
    data_np = data.to_numpy()
    point_cloud['data'] = data_np  # N*7
    point_cloud['segmentation_mask'] = None  # N*1 bool
    # point_cloud['instances_mask'] = None # M*2 (bool,instances_id)
    # point_cloud['matching_mask'] = None # M *1 int (instance id)
    point_cloud['segmentation'] = None  # M*4 (x,y,z,frame_id)
    point_cloud['instances'] = None  # L*3 (x,y,instances)
    point_cloud['matching'] = None  # L *1 int (instance id)
    run_segmentation = True
    run_instances = True
    run_matching = True
    print("ok")
    if run_segmentation:
        #try:
        segmentation_main(point_cloud)  # data are saved inside point_cloud
        if point_cloud['segmentation'] is not None:
            np.save('segmentation.npy', point_cloud['segmentation'])
        else:
            print("Segmentation wrong shape or type")
            if os.path.exists('segmentation.npy'):
                point_cloud['segmentation'] = np.load('segmentation.npy')
            else:
                return -1
        #except Exception as e:
         #   print("Segmentation failed")
          #  print(e)
        if os.path.exists('segmentation.npy'):
            point_cloud['segmentation'] = np.load('segmentation.npy')
        else:
           return -1
    else:
        if os.path.exists('segmentation.npy'):
            point_cloud['segmentation'] = np.load('segmentation.npy')
        else:
            return -1
    print("segmentation done")
    """if run_instances:
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
            if point_cloud['matching'] is not None:
                np.save('matching.npy', point_cloud['matching'])
            else:
                if os.path.exists('matching.npy'):
                    point_cloud['matching'] = np.load('matching.npy')
                else:
                    return -1
        except:
            print("Matching failed")
            if os.path.exists('matching.npy'):
                point_cloud['matching'] = np.load('matching.npy')
            else:
                return -1
    else:
        if os.path.exists('matching.npy'):
            point_cloud['matching'] = np.load('matching.npy')
        else:
            return -1
    # create final xml
    return point_cloud  # return filled dictionary


if __name__ == "__main__":
    main()
