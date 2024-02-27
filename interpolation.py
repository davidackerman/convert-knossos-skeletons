# taken from https://github.com/nilsec/micron/blob/0f0807d1f45e398ea04e04e7f7cdeafd1a0dab8d/micron/post/analyse_graph.py#L288-L326
import operator


def interpolate_on_grid(z0, y0, x0, z1, y1, x1, voxel_size):
    """
    Interpolate a line on a 3D voxel grid.
    x0, y0, ... (int, physical space e.g. nm):
    returns: list of voxels forming a line from p0 to p1
    """

    def dda_round(x):
        return (x + 0.5).astype(int)

    start = np.array([z0, y0, x0], dtype=float)
    end = np.array([z1, y1, x1], dtype=float)

    voxel_size = np.array(voxel_size)
    if np.any(start % voxel_size) or np.any(end % voxel_size):
        print(start % voxel_size, end % voxel_size)
        raise ValueError("Start end end position must be multiples of voxel size")

    line = [dda_round(start / voxel_size)]

    if not np.any(start - end):
        return line

    max_direction, max_length = max(
        enumerate(abs(end - start)), key=operator.itemgetter(1)
    )

    dv = (end - start) / max_length

    # We interpolate in physical space to find the shortest distance
    # linear interpolation but the path is represented in voxels
    for step in range(int(max_length)):
        step_point_rescaled = np.array(
            dda_round(dda_round((step + 1) * dv + start) / voxel_size)
        )
        if not np.all(step_point_rescaled == line[-1]):
            line.append(step_point_rescaled)

    assert np.all(line[-1] == dda_round(end / voxel_size))
    return line


# from /nrs/funke/ecksteinn/nils_data/cosem_data/cosem_runs/full_cell/cosem_hela_2_full/00_data/exports/read_nml.py
from xml.dom import minidom
import numpy as np
import networkx as nx


def parse_nml(filename, edge_attribute=None):
    doc = minidom.parse(filename)
    annotations = doc.getElementsByTagName("thing")

    annotation_edge_dict = {}
    node_dic = {}
    edge_list = []
    for annotation in annotations:
        current_edge_list = []
        nodes = annotation.getElementsByTagName("node")
        for node in nodes:
            node_position, node_id = parse_node(node)
            node_dic[node_id] = node_position

        edges = annotation.getElementsByTagName("edge")

        for edge in edges:
            (source_id, target_id) = parse_attributes(
                edge, [["source", int], ["target", int]]
            )
            edge_list.append((source_id, target_id))
            current_edge_list.append((source_id, target_id))

        nodes, counts = np.unique(edge_list, return_counts=True)

        G = nx.Graph(current_edge_list)

        endnodes = [node for node, degree in G.degree() if degree == 1]
        sorted_nodes = nx.shortest_path(G, endnodes[0], endnodes[1])

        sorted_edge_list = []
        for i in range(len(sorted_nodes) - 1):
            sorted_edge_list.append((sorted_nodes[i], sorted_nodes[i + 1]))

        annotation_id = parse_attributes(annotation, [["id", int]])[0]
        annotation_edge_dict[annotation_id] = sorted_edge_list

    return node_dic, edge_list, annotation_edge_dict


def parse_node(node):
    [x, y, z, id_] = parse_attributes(
        node,
        [
            ["x", float],
            ["y", float],
            ["z", float],
            ["id", int],
        ],
    )

    point = np.array([z, y, x])

    return point, id_


def parse_edge(edge):
    [source, target] = parse_attributes(edge, [["source", int], ["target", int]])

    return source, target


def parse_attributes(xml_elem, parse_input):
    parse_output = []
    attributes = xml_elem.attributes
    for x in parse_input:
        try:
            parse_output.append(x[1](attributes[x[0]].value))
        except KeyError:
            parse_output.append(None)
    return parse_output
