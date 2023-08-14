import xml.etree.ElementTree as ET
import xml.dom.minidom

def xml_result(all_lines,centers_array,g):
    attributes = ET.Element('attributes',
                {'datfile': 'testing.dat',
                    'majorversion': '1',
                    'minorversion': '1',
                    'structurefile': 'test_label_project_structure.xml'
                })

    lanes = ET.SubElement(attributes, 'lanes')
    lines_group = ET.SubElement(lanes, 'Lines')
    id_line = 0
    for line in all_lines:
        for id_point in line:
            for other in line:
                if id_point != other and other in g.graph[id_point]:
                    line_group = ET.SubElement(lines_group, 'lines')
                    ET.SubElement(line_group, 'id').text = str(id_line)
                    ET.SubElement(line_group, 'Lane_type').text = 'Single Dashed'
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

                    ET.SubElement(coordinates, f'xp_{0}').text = str(centers_array[id_point][0].item())
                    ET.SubElement(coordinates, f'yp_{0}').text = str(centers_array[id_point][1].item())
                    ET.SubElement(coordinates, f'zp_{0}').text = '0'

                    line_timestamp = ET.SubElement(line_position, 'line_timestamp',
                                    {'time': '1675850223743312',
                                        'frame': '1',
                                        'sampletime': '1675850223743312',
                                        'interpolationState': 'end'})
                    line_elem = ET.SubElement(line_timestamp, 'line')
                    line_geom = ET.SubElement(line_elem, 'line_geom')
                    ET.SubElement(line_geom, 'closed').text = 'false'
                    coordinates = ET.SubElement(line_geom, 'coordinates')
                    ET.SubElement(coordinates, f'xp_{0}').text = str(centers_array[other][0].item())
                    ET.SubElement(coordinates, f'yp_{0}').text = str(centers_array[other][1].item())
                    ET.SubElement(coordinates, f'zp_{0}').text = '0'
        id_line += 1

    # Generate the string representation of the XML
    xml_str = ET.tostring(attributes, encoding='iso-8859-1')

    # Parse the string with minidom, which can prettify the XML
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml(indent="  ")

    # Save the XML in a file
    with open('case_real/result.xml', 'wb') as f:
        f.write(pretty_xml_str.encode())
