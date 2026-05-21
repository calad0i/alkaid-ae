import hls4ml
import tensorflow as tf
from qkeras import QActivation, QConv1D, QDense, quantized_bits
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.sparsity.keras import strip_pruning

import hls_node_edge_projection  # Registers the NodeEdgeProjection hls4ml handler.
from node_edge_projection import NodeEdgeProjection


def synthesize(path, oname='tt', build=False, trace=False):
    model = tf.keras.models.load_model(
        path,
        custom_objects={
            'QDense': QDense,
            'QConv1D': QConv1D,
            'QActivation': QActivation,
            'quantized_bits': quantized_bits,
            'NodeEdgeProjection': NodeEdgeProjection,
            'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
        },
    )
    model = strip_pruning(model)

    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity='name',
        default_precision='ap_fixed<16,6>',
    )
    config['Model']['Strategy'] = 'Latency'

    input_precision = 'ap_fixed<12,4,AP_RND,AP_SAT>'
    for layer in model.layers:
        if layer.__class__.__name__ in ['InputLayer']:
            config['LayerName'][layer.name]['Precision'] = input_precision
            config['LayerName'][layer.name]['Trace'] = trace

        if layer.__class__.__name__ in [
            'Permute',
            'Concatenate',
            'Flatten',
            'Reshape',
        ]:
            print('Skipping trace for:', layer.name)
        else:
            config['LayerName'][layer.name]['Trace'] = trace

    reuse_factor_conv1d = 1
    for layer in model.layers:
        config['LayerName'][layer.name]['Strategy'] = 'latency'

        if 'NodeEdgeProjection' in layer.__class__.__name__ and layer.name == 'proj_3':
            print('Configure the precision of the NodeEdgeProjection proj_3')
            config['LayerName']['proj_3']['Precision'] = 'ap_fixed<16, 6>'

        if 'Concatenate' in layer.__class__.__name__ and layer.name == 'concatenate_1':
            config['LayerName']['concatenate_1']['Precision'] = 'ap_fixed<16,6,AP_RND,AP_SAT>'

        if 'Conv1D' in layer.__class__.__name__ and layer.name == 'conv1D_e1':
            config['LayerName']['conv1D_e1']['ReuseFactor'] = reuse_factor_conv1d
        if 'Conv1D' in layer.__class__.__name__ and layer.name == 'conv1D_e2':
            config['LayerName']['conv1D_e2']['ReuseFactor'] = reuse_factor_conv1d
        if 'Conv1D' in layer.__class__.__name__ and layer.name == 'conv1D_e3':
            config['LayerName']['conv1D_e3']['ReuseFactor'] = reuse_factor_conv1d

    config['LayerName']['conv1D_n1']['ReuseFactor'] = reuse_factor_conv1d
    config['LayerName']['conv1D_n2']['ReuseFactor'] = reuse_factor_conv1d
    if 'conv1D_n3' in config['LayerName']:
        config['LayerName']['conv1D_n3']['ReuseFactor'] = reuse_factor_conv1d

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=f'/tmp/{oname}',
        io_type='io_parallel',
        part='xcvu13p-flga2577-2-e',
        backend='vitis',
    )
    return model, hls_model
