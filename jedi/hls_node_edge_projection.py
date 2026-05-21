from itertools import permutations

import numpy as np
from hls4ml.converters.keras_v2_to_hls import layer_handlers


def make_proj_mat(_node_to_edge, _receiving, input_shape):
    if _node_to_edge:
        _n_nodes = input_shape[-2]
        _n_edges = _n_nodes * (_n_nodes - 1)
    else:
        _n_edges = input_shape[-2]
        _n_nodes = int((np.sqrt(4 * _n_edges + 1) + 1) / 2)
    receiver_sender_list = permutations(range(_n_nodes), r=2)

    if _node_to_edge:
        shape = (1, _n_edges, _n_nodes)
    else:
        shape = (1, _n_nodes, _n_edges)
    adjacency_matrix = np.zeros(shape, dtype=float)

    for i, (r, s) in enumerate(receiver_sender_list):
        if _node_to_edge:
            if _receiving:
                adjacency_matrix[0, i, r] = 1
            else:
                adjacency_matrix[0, i, s] = 1
        else:
            if _receiving:
                adjacency_matrix[0, r, i] = 1
            else:
                adjacency_matrix[0, s, i] = 1
    return adjacency_matrix


def parse_node_edge_projection_layer(keras_layer, input_names, input_shapes, data_reader):
    layer = {}
    layer['class_name'] = 'EinsumDense'
    layer['equation'] = '...ij,ki->...kj'
    layer['name'] = keras_layer['config']['name']
    layer['bias_data'] = None
    layer['weight_data'] = make_proj_mat(
        keras_layer['config']['node_to_edge'], keras_layer['config']['receiving'], input_shapes[0]
    )[0]

    layer['node_to_edge'] = keras_layer['config']['node_to_edge']
    if layer['node_to_edge']:
        layer['n_nodes'] = input_shapes[0][1]
        layer['n_edges'] = layer['n_nodes'] * (layer['n_nodes'] - 1)
    else:
        layer['n_edges'] = input_shapes[0][1]
        layer['n_nodes'] = int((np.sqrt(4 * layer['n_edges'] + 1) + 1) / 2)

    layer['in_width'] = input_shapes[0][1]
    layer['out_width'] = layer['n_edges'] if layer['node_to_edge'] else layer['n_nodes']

    output_shapes = [input_shapes[0][0], layer['out_width'], input_shapes[0][2]]
    layer['out_shape'] = tuple(output_shapes[1:])
    layer['inp_shape'] = tuple(input_shapes[0][1:])

    if input_names is not None:
        layer['inputs'] = input_names

    return layer, output_shapes


layer_handlers['NodeEdgeProjection'] = parse_node_edge_projection_layer
