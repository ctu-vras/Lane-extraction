# Lane Marking for Valeo

Python application for detecting lanes from Lidar. 

Takes data as a .pcd in format N*6(x,y,z,intensity,frame_id,laser_id) and outputs lines in xml format and in .npz.

## Installation
I strongly recommend to use docker image, because it is the easiest way to run the application and ensure that all libraries are installed correctly.

- First you need to get docker image from hub or file
  ```bash 
  docker load valeo.tar.gz
  ```
- Alternative is to copy project from github, which enables to use docker-compose file to run it.
  ```bash
  git clone https://github.com/ctu-vras/Lane-extraction.git
  docker compose up
  ```
Before running the application you need to prepare a folder that contains two files. 
  - filename.pcd with data you want to process
  - config.yaml - config file that will overwrite default parameters
    - lines that are recommended to overwrite
      - DATA_PATH: "source/in/filename.pcd" just change the filename to actual file. source/in is a directory in docker container that will be later mapped on your host directory
      - CUDA_CARD: 'cuda:0' default is cuda:0 but if you have more than one gpu, you can choose which one to use
      - XML_FILE_NAME: "source/out/result.xml" same as data path, just change the filename to what you want
      - OUTPUT_FILE_NAME: "source/out/output.npz" same as data path, just change the filename to what you want
      - ANIMATION: True If you want to see the animation of the process set it to True but it can take a lot of time and fail for big instances. If you don't want them set it to False
      - RUN_PARTS:
        - SEGMENTATION: True 
        - INSTANCES: True
        - MATCHING: True #set values to true or false if you want to run a part of program. If you have run a segmentation for example, it will store the data, so if you set it to False, it will load these data instead and do the next segment.
        
To run the application you need to run docker container with mapped volumes
```bash
  docker run -v path/to/your/folder:/Lane-extraction/source/in -v path/to/your/folder:/Lane-extraction/source/out kominma3/valeo_images:heuristicv2
```

## Description of the inside work:
There are three main parts of the application. Segmentation, Instances, Matching. To connect these parts, I created a folder in common/pipeline that has 2 files. main.py and config.yaml. If you want to run the application, you need to copy main.py into root folder, so into Lane-extraction.
This main.py reads config.yaml to set parameters as what data we want to read, naming of intermediate files. Then it calls segmentation, instances and matching modules that each has its own folder. Each of these modules has file module_pipeline.py and config.yaml. These files are used to run each module and return the results back.
To move the data between modules, we upload the results to dictionary. So main.py creates a dictionary, opens the pcd and then its send to segmentation. Segmentation returns the dictionary with segmented data, and then the main.py sends this dictionary to instances and so on.
Segmentation takes in N * 6 and outputs M * 6 of points that we identified as lane. It also outputs mask to apply to the original data.
Instances takes in M * 6 and outputs K * 3 of points that we identified as instances of lanes.(each dash). In format(x,y,instance_id)
Matching takes in K * 3 and outputs lines into dictionary and it creates a file with lines in xml format.
Than pipeline creates an .npz file with lines in valeo format. It has 'lanes' which is an np.array of (np.array(2,) of points,id,type_of_line,colour)
and odometry that doesnt change.
Example of a XML file:
![alt text](https://github.com/ctu-vras/Lane-extraction/blob/main/common/pipeline/img.png?raw=true)
Each lane has its id and then there are list of coordinates for each point that represents the line.

## Troubleshoting
If you have problems with running the application you can open an issue or email me at kominma3@fel.cvut.cz

