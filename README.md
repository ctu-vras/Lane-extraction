# Lane-extraction

[//]: # (## Ticket Structure)

[//]: # (    - General:)

[//]: # (    - Person: )

[//]: # (    - Tasks:)

[//]: # (    - References:)

[//]: # (    - Input:)

[//]: # (    - Output:)

## Links to presentations and codes from final meeting:
- Lane Matching presentation: https://docs.google.com/presentation/d/1_aKsyaCRaU2LQY1fK3FoJUxReD7GSdPmB6vxaPzUNW0/edit?usp=sharing
- Lane Matching ipynb notebook: https://github.com/KomajzCz/FEL-matching-notebook
  

## Current state of branches:
- main: branch has an old code. It should be similiar with heuristic but README for docker is in common/pipeline. Main purpose is overview of tasks given by Patrik.
- heuristic: actual branch with old segmentation, but it is in final state and can be packaged into Docker
- xyzi_nn_segmentation : actual branch with current version of cylydric segmentation that can be packaged into Docker
- cylinder_segmentation: old branch Ondra than switched to xyzi_nn_segmentation
- dev: very old branch nothing is there 


### Find Datasets for Lanes
- General: Potential datasets for using existing annotations,
images to LiDAR projection annotation, HD maps with lanes and LiDAR
- Person: Ondra
- Tasks: 
    1. List all potentially useful datasets with instance-level annotations
    2. Make a material (table, half page of text) we can decide on and send to Valeo 
- References:
    1. OpenDriveLab/OpenLane-V2: [NeurIPS 2023 Track Datasets] - No LiDAR
    2. Argoverse 2 HD maps
    3. [K-LANE](https://github.com/kaist-avelab/k-lane)
- Input: Search on internet and try to map existings datasets to our needs
- Output:
    1. List of datasets with parameters in text or table (has id annotations, scene diversity, LiDAR sensor, ...)
    2. Conclussion on what datasets we can use for learning the model to detect lanes id/polylines

    

### K-Lane Devkit + Metrics
  - General: Prepare annotation tool and understand the metrics Lane extraction scenario (K-Lane should use the most common evaluation protocols)
  - Person: Honza + Martin
  - Tasks:
    1. Download K-Lane devkit and run annotation GUI
    2. Annotate one point cloud from Valeo dataset to learn how to use it, annotate full line, not segments (dashed)
    3. Think about transfer of K-Lane annotations into Valeo polylines format
    4. Construct meaningful metrics which we can use for evaluation given what we have in Valeo and what we can annotate  
  - Reference:
    1. [Paper](https://arxiv.org/pdf/2110.11048.pdf)
    2. [K-LANE](https://github.com/kaist-avelab/k-lane)
  - Input: Codebase, existing data from Valeo
  - Output: System for annotating data, Exact and systematic protokol, how we should evaluate our results

### K-Lane <--> Valeo Data
  - General: Download and learn how to use the data and easily transfer with Valeo format
  - Person: Yana
  - Tasks:
      1. Download K-Lane dataset
      2. Learn how to use the dataset
      3. Convert to Valeo format
  - References: 
      1. [K-LANE](https://github.com/kaist-avelab/k-lane)
  - Input: K-Lane Dataset, Valeo Dataset
  - Output: Functions to allow for training on both datasets, merging the formats/annotations 

### K-Lane Detection Model
  - General: To get baseline for lane detection
  - Person: -
  - Tasks:
      1. Learn how to infer the K-Lane model used in [Paper](https://arxiv.org/pdf/2110.11048.pdf)
      3. Run The model on Valeo Dataset 
      4. Calculate metrics on Valeo Dataset 
      5. Retrain models on Valeo Dataset and show metrics 
  - References:
      1. [K-LANE](https://github.com/kaist-avelab/k-lane)
  - Input: Existing codebase, data samples
  - Output: Importable models, calculated metrics on both datasets using training on K-Lane and both.



### Future
- Camera propagated to LiDAR - To demonstrate additional data gathering
- Argoverse Dataset HD maps - to get additional data
- Pseudo-labelling - to easily boost performance
- Test-Time Augmentation - to easily boost performance
- Active learning framework - to save annotations

