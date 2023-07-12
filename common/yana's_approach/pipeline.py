import subprocess

input_file = "preprocessing_cpp/new_dataset/2023-01-17_12-15-18_1/projection.pcd"

subprocess.run(['python3', 'segmentation/main.py', input_file])

subprocess.run(['python3', 'line_fitting/main.py'])

subprocess.run(['python3', 'visualization/main.py', input_file])