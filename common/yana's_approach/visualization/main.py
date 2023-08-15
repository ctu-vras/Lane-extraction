import pickle
import sys
import xml.dom.minidom
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from pyntcloud import PyntCloud


class Line:
 def __init__(self, coef, intercept, points, type_of_line):
    self.coef = coef
    self.intercept = intercept
    self.points = points
    self.points_x = points[:, 0]
    self.points_y = self.coef * self.points_x + self.intercept
    self.type = type_of_line
    self.point1, self.point2 = self.end_points()

 def end_points(self):
    """Calculate the end points of a line based on the points on the line"""
    min_i = np.argmin(self.points_x)
    max_i = np.argmax(self.points_x)
    x_min, y_min = self.points_x[min_i], self.points_y[min_i]
    x_max, y_max = self.points_x[max_i], self.points_y[max_i]
    return [(x_min, y_min), (x_max, y_max)]

def plot_point_cloud_and_lines_2D(point_cloud, all_lines):
    all_points = []


    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # Plot the point cloud
    xs = point_cloud[:, 0]
    ys = point_cloud[:, 1]
    ax1.scatter(xs, ys, color='gray', s=1)  # change s for different size
    ax2.scatter(xs, ys, color='gray', s=1)  # change s for different size

    # Define colors for the lines
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_lines)))

    line_number = 1
    # Iterate over all lines
    for i, (color, (angle, lines)) in enumerate(zip(colors, all_lines.items())):
        for line in lines:

            all_points.append([(x, y) for x, y, _ in line.points])

            label = f"{line_number}"
            info = f"{line_number}: Coefficient = {round(line.coef)}, Intercept = {round(line.intercept)}, Type = {line.type}"

            # Extract points
            x = [line.point1[0], line.point2[0]]
            y = [line.coef * line.point1[0] + line.intercept, line.coef * line.point2[0] + line.intercept]
            
            # Plot the line
            ax1.plot(x, y, color=color, label=info)
            
            # Plot the points
            ax1.scatter(x, y, color=color, s=10)

            # Add text annotation for line
            mid_x = sum(x) / 2
            mid_y = sum(y) / 2
            ax1.text(mid_x, mid_y, label, fontsize=10, ha='center')

            all_x = line.points[:, 0]
            all_y = line.points[:, 1]
            ax2.scatter(all_x, all_y, color=color, s=2)

        line_number += 1

    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=4)
    fig1.tight_layout()
    plt.show()

def xml_result(all_lines):
    attributes = ET.Element('attributes', 
                {'datfile': '2023-01-17_12-15-18_1.dat', 
                    'majorversion': '1', 
                    'minorversion': '1', 
                    'structurefile': 'PONE_label_project_structure.xml', 
                    'structurefilechk': '262f5e2dc8dead7199d81a9168c7a3fd'})

    lanes = ET.SubElement(attributes, 'lanes')
    lines_group = ET.SubElement(lanes, 'Lines')

    for key, lines in all_lines.items():
        line_group = ET.SubElement(lines_group, 'lines')
        ET.SubElement(line_group, 'id').text = str(key)
        ET.SubElement(line_group, 'Lane_type').text = 'Single Dashed' if any(line.type == 'dashed' for line in lines) else 'Continuous'
        ET.SubElement(line_group, 'Line_color').text = 'WHITE'
        line_position = ET.SubElement(line_group, 'Line_position')
        line_timestamp = ET.SubElement(line_position, 'line_timestamp', 
                                        {'time': '1675850223743312', 
                                            'frame': '1', 
                                            'sampletime': '1675850223743312', 
                                            'interpolationState': 'start'})
        line_elem = ET.SubElement(line_timestamp, 'line')
        line_geom = ET.SubElement(line_elem, 'line_geom')
        ET.SubElement(line_geom, 'closed').text = 'false'
        coordinates = ET.SubElement(line_geom, 'coordinates')
        for i in range(len(lines)):
            ET.SubElement(coordinates, f'xp_{i}').text = str(lines[i].point1[0])
            ET.SubElement(coordinates, f'yp_{i}').text = str(lines[i].point1[1])
            ET.SubElement(coordinates, f'zp_{i}').text = '0'

        line_timestamp = ET.SubElement(line_position, 'line_timestamp', 
                        {'time': '1675850223743312', 
                            'frame': '1', 
                            'sampletime': '1675850223743312', 
                            'interpolationState': 'end'})
        line_elem = ET.SubElement(line_timestamp, 'line')
        line_geom = ET.SubElement(line_elem, 'line_geom')
        ET.SubElement(line_geom, 'closed').text = 'false'
        coordinates = ET.SubElement(line_geom, 'coordinates')
        for i in range(len(lines)):
            ET.SubElement(coordinates, f'xp_{i}').text = str(lines[i].point2[0])
            ET.SubElement(coordinates, f'yp_{i}').text = str(lines[i].point2[1])
            ET.SubElement(coordinates, f'zp_{i}').text = '0'

    # Generate the string representation of the XML
    xml_str = ET.tostring(attributes, encoding='iso-8859-1')

    # Parse the string with minidom, which can prettify the XML
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent="  ")

    # Save the XML in a file
    with open('visualization/result.xml', 'wb') as f:
        f.write(pretty_xml_str.encode())

def main():
    with open('line_fitting/data.pkl', 'rb') as f:
        all_lines = pickle.load(f)

    filepath = sys.argv[1]
    cloud = PyntCloud.from_file(filepath)
    data = cloud.points.to_numpy()
    
    plot_point_cloud_and_lines_2D(data, all_lines)

    xml_result(all_lines)

if __name__ == "__main__":
    main()
