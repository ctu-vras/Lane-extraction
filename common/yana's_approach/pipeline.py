import subprocess

input_directory = "PONE-LineDataset/dataset/2023-01-17_12-15-18_1"

subprocess.run(["mkdir", "-p", "preprocessing_cpp/build"], check=True)
subprocess.run(["cmake", ".."], cwd="preprocessing_cpp/build", check=True)
subprocess.run(["make"], cwd="preprocessing_cpp/build", check=True)
subprocess.run(["./preprocessing_cpp/build/preprocessing", f"{input_directory}/LR1_local.pcd"], check=True)

subprocess.run(["pip3", "install", "-r", "requirements.txt"], check=True)

subprocess.run(['python3', 'segmentation/main.py', "projection.pcd"])

subprocess.run(['python3', 'line_fitting/main.py'])

subprocess.run(['python3', 'visualization/main.py', "projection.pcd"])
