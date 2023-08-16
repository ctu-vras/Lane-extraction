import numpy as np
import torch
import xml.etree.ElementTree as ET


def load_real_data_pc(name: str, start_end: [int, int]) -> torch.Tensor:
    """
    Load the data extracted from the currently processed point cloud.
    :param name: name of the point cloud
    :param start_end: start and end of the sequence
    :return: the point cloud
    """
    print("\n\n--------------- LOADING REAL DATA ---------------")
    # Get start and end
    start = start_end[0]
    end = start_end[1]

    # Load from npz file
    if "test" in name and start == end:
        data_and_mask = np.load(f'real_data/{name}.npz')
        data = data_and_mask['data']
        mask = data_and_mask['mask']
        pc = data[mask]
        print("- Number of loaded frames: ", len(np.unique(pc[:, 4])))
        pc = pc[:, :3]
    elif "test" in name and start != end:
        for i in range(start, end+1):
            if len(str(i)) != 1:
                data_and_mask = np.load(f'real_data/test{i}.npz')
            else:
                data_and_mask = np.load(f'real_data/test0{i}.npz')
            data = data_and_mask['data']
            mask = data_and_mask['mask']
            if i == start:
                pc = data[mask]
            else:
                pc = np.concatenate([pc, data[mask]], axis=0)
        print("- Number of loaded frames: ", len(np.unique(pc[:, 4])))
        pc = pc[:, :3]
    else:
        pc = np.load(f'real_data/{name}.npz')
        pc = pc[pc.files[0]]
        print("- Number of loaded frames: ", len(np.unique(pc[:, 4])))
        pc = pc[:, :3]

    # Convert to torch tensor
    pc = torch.from_numpy(pc)

    # Convert it to float32
    pc = pc.float()
    print("- Number of loaded points: ", pc.shape[0])

    return pc


def read_xml_file(file_path):
    """
    Read the output xml file of the Yana's module and return a dictionary of all lines
    :param file_path: path to the xml file
    :return: dictionary of all lines - format: {line_id: [[xps, yps, zps, line_type], ...], ...}
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    all_lines = {}

    for line_group in root.findall(".//lines"):
        key = int(line_group.find("id").text)
        lines = []

        for line_elem in line_group.findall(".//line"):
            coordinates = line_elem.find(".//coordinates")
            points = []
            point = []
            for i, coordinate in enumerate(coordinates):
                point.append(float(coordinate.text))
                if i % 3 == 2:
                    points.append(point)
                    point = []
            line_type = line_group.find("Lane_type").text

            lines.append(points)
            lines.append(line_type)

        all_lines[key] = lines

    return all_lines


if __name__ == '__main__':
    path = '../common/yana\'s_approach/visualization/result.xml'
    all_lines = read_xml_file(path)

