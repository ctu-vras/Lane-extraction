import os
import numpy as np
import torch
#import yaml
from pyntcloud import PyntCloud

from instances.instances_pipeline import instances_main
from matching.matching_pipeline import matching_main
from segmentation.segmentation_pipeline import segmentation_main
from segmentation.deploy_model import parse_input, merge_point_cloud
from ruamel.yaml import YAML

def main():
    # load config from yaml inside project
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    yaml = YAML()
    yaml.default_flow_style = False
    with open('./common/pipeline/config.yaml', "r") as f:
        config = yaml.load(f)
    # load config from yaml if there is one mounted from docker this config will
    #overwrite all properties that are same as the one inside the project
    if os.path.exists('source/in/config.yaml'):
        print("Loading config from outside")
        with open('source/in/config.yaml', "r") as f:
            outside_config = yaml.load(f)
        for key in outside_config:
            config[key] = outside_config[key]
    #save config to output to see what was used
    if os.path.exists('source/out'):
        with open('source/out/config_out.yaml', 'w') as yaml_output:
            yaml.dump(config, yaml_output)
    #create a file path to the data
    path = os.path.join(os.path.abspath(os.curdir), config['DATA_PATH'])
    print("starting main")
    #dictionary to save all the data
    point_cloud = {}
    #load point cloud from file
    if config['RUN_PARTS']['SEGMENTATION']:
        print("Loading point cloud")
        scan_list, pose_list = parse_input(path)
        point_cloud['data'] = merge_point_cloud(frames_list=scan_list, pose_list=pose_list, pipeline_config=config)
        point_cloud['poses'] = pose_list
    print("Point cloud loaded")
    #clear GPU before starting
    torch.cuda.set_device(config['CUDA_CARD'])
    torch.cuda.empty_cache()
    #prepare keys for the dictionary
    point_cloud['segmentation'] = None  # M*5 (x,y,z,intensity,frame_id)
    point_cloud['instances'] = None  # L*3 (x,y,instances)
    point_cloud['matching'] = None  # L*1 int (instance id)
    #either run segmentation else load it
    if config['RUN_PARTS']['SEGMENTATION']:
        #try:
        print("segmentation start")
        #start segmentation that will save result into dictionary
        segmentation_main(point_cloud, config)  # data are saved inside point_cloud
        print("segmentation done")
        #save segmentation if successful
        if point_cloud['segmentation'] is not None:
            np.save(config['SAVE_NAMES']['SEGMENTATION'], point_cloud['segmentation'])
        else:
            #segmentation failed try loading it
            print("Segmentation wrong shape or type")
            if os.path.exists(config['LOAD_NAMES']['SEGMENTATION']):
                point_cloud['segmentation'] = np.load(config['LOAD_NAMES']['SEGMENTATION'])
            else:
                return -1
    else:
        #load segmentation from file
        if os.path.exists(config['LOAD_NAMES']['SEGMENTATION']):
            point_cloud['segmentation'] = np.load(config['LOAD_NAMES']['SEGMENTATION'])
        else:
            return -1

    #clear GPU before starting instances
    torch.cuda.empty_cache()
    if config['RUN_PARTS']['INSTANCES']:
        #try:
        print("Instances start")
        instances_main(point_cloud,config['CUDA_CARD'])
        print("Instances done")
        #save if instances were successful
        if point_cloud['instances'] is not None:
            np.save(config['SAVE_NAMES']['INSTANCES'], point_cloud['instances'])
        else:
            #instances failed try loading them
            if os.path.exists(config['LOAD_NAMES']['INSTANCES']):
                point_cloud['instances'] = np.load(config['LOAD_NAMES']['INSTANCES'])
            else:
                return -1
    else:
        #load instances from file
        if os.path.exists(config['LOAD_NAMES']['INSTANCES']):
            point_cloud['instances'] = np.load(config['LOAD_NAMES']['INSTANCES'])
        else:
            return -1
    #clear before matching
    torch.cuda.empty_cache()
    if config['RUN_PARTS']['MATCHING']:
        #try:
        print("Matching start")
        matching_main(point_cloud,config['CUDA_CARD'],config['XML_FILE_NAME'],config['ANIMATION'])
        print("Matching done")
        #save if matching was successful
        if point_cloud['matching'] is not None:
            np.save(config['SAVE_NAMES']['MATCHING'], point_cloud['matching'])
        else:
            #matching failed try loading it
            if os.path.exists(config['LOAD_NAMES']['MATCHING']):
                point_cloud['matching'] = np.load(config['LOAD_NAMES']['MATCHING'])
            else:
                return -1

    else:
        #load matching from file
        if os.path.exists(config['LOAD_NAMES']['MATCHING']):
            point_cloud['matching'] = np.load(config['LOAD_NAMES']['MATCHING'])
        else:
            return -1
    torch.cuda.empty_cache()
    return point_cloud  # return filled dictionary


if __name__ == "__main__":
    main()
