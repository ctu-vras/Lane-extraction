import os

import numpy as np
import yaml
from pyntcloud import PyntCloud

from instances.instances import instances_main
from matching.instance_matching import matching_main
from segmentation.segmentation_pipeline import segmentation_main



def main():
    # path = sys.argv[1]
    # server and load data accordingly
    # config file to set what to load
    print("starting main")
    with open('common/pipeline/config.yaml', "r") as f:
        config = yaml.safe_load(f)
    path = os.path.join(os.path.abspath(os.curdir), config['DATA_PATH'])
    #print(path)
    point_cloud = {}
    #print("Loading point cloud")
    cloud = PyntCloud.from_file(path)
    data = cloud.points
    data_np = data.to_numpy()
    point_cloud['data'] = data_np  # N*7
    #print(point_cloud['data'].size*point_cloud['data'].itemsize/1000000000)
    #point_cloud['segmentation_mask'] = None  # N*1 bool
    # point_cloud['instances_mask'] = None # M*2 (bool,instances_id)
    # point_cloud['matching_mask'] = None # M *1 int (instance id)
    point_cloud['segmentation'] = None  # M*4 (x,y,z,frame_id)
    point_cloud['instances'] = None  # L*3 (x,y,instances)
    point_cloud['matching'] = None  # L*1 int (instance id)
    if config['RUN_PARTS']['SEGMENTATION']:
        #try:
        print("segmentation start")
        segmentation_main(point_cloud)  # data are saved inside point_cloud
        if point_cloud['segmentation'] is not None:
            np.save(config['SAVE_NAMES']['SEGMENTATION'], point_cloud['segmentation'])
        else:
            print("Segmentation wrong shape or type")
            if os.path.exists(config['LOAD_NAMES']['SEGMENTATION']):
                point_cloud['segmentation'] = np.load(config['LOAD_NAMES']['SEGMENTATION'])
            else:
                return -1
        #except Exception as e:
         #   print("Segmentation failed")
          #  print(e)
        if os.path.exists(config['LOAD_NAMES']['SEGMENTATION']):
            point_cloud['segmentation'] = np.load(config['LOAD_NAMES']['SEGMENTATION'])
        else:
           return -1
    else:
        if os.path.exists(config['LOAD_NAMES']['SEGMENTATION']):
            point_cloud['segmentation'] = np.load(config['LOAD_NAMES']['SEGMENTATION'])
        else:
            return -1

    print("segmentation done")
    print(point_cloud['segmentation'].shape)
    print("Memory size of numpy array in bytes:",
          point_cloud['segmentation'].size * point_cloud['segmentation'].itemsize/1000000000)
    if config['RUN_PARTS']['INSTANCES']:
        #try:
        print("Instances start")
        instances_main(point_cloud)
        print("Instances done")
        if point_cloud['instances'] is not None:
            np.save(config['SAVE_NAMES']['INSTANCES'], point_cloud['instances'])
        else:
            if os.path.exists(config['LOAD_NAMES']['INSTANCES']):
                point_cloud['instances'] = np.load(config['LOAD_NAMES']['INSTANCES'])
            else:
                return -1
        """except:
            print("Instances failed")
            if os.path.exists(config['LOAD_NAMES']['INSTANCES']):
                point_cloud['instances'] = np.load(config['LOAD_NAMES']['INSTANCES'])
            else:
                return -1"""
    else:
        if os.path.exists(config['LOAD_NAMES']['INSTANCES']):
            point_cloud['instances'] = np.load(config['LOAD_NAMES']['INSTANCES'])
        else:
            return -1

    if config['RUN_PARTS']['MATCHING']:
        print("Matching start")
        matching_main(point_cloud)
        if point_cloud['matching'] is not None:
            np.save(config['SAVE_NAMES']['MATCHING'], point_cloud['matching'])
        else:
            if os.path.exists(config['LOAD_NAMES']['MATCHING']):
                point_cloud['matching'] = np.load(config['LOAD_NAMES']['MATCHING'])
            else:
                return -1
        """except:
            print("Matching failed")
            if os.path.exists('matching.npy'):
                point_cloud['matching'] = np.load(config['LOAD_NAMES']['MATCHING'])
            else:
                return -1"""
    else:
        if os.path.exists(config['LOAD_NAMES']['MATCHING']):
            point_cloud['matching'] = np.load(config['LOAD_NAMES']['MATCHING'])
        else:
            return -1
    # create final xml
    return point_cloud  # return filled dictionary


if __name__ == "__main__":
    main()
