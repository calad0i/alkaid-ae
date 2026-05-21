import os
import sys
from pathlib import Path
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '999'

import numpy as np
import tensorflow as tf
from alkaid.codegen import RTLModel
from alkaid.converter import trace_model
from alkaid.trace import trace
from qkeras import quantized_bits
from quantizers import FixedQ
from synthesize import synthesize

tf.get_logger().setLevel('CRITICAL')


@classmethod
def from_config(cls, config):
    assert config.pop('post_training_scale', None) is None  # Where is this post_training_scale coming from even?
    return cls(**config)


quantized_bits.from_config = from_config  # type: ignore


class ShutUp:
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def write(self, _):
        pass

    def flush(self):
        pass


if __name__ == '__main__':
    for N in (8, 16, 32):
        with ShutUp():
            model, model_hls = synthesize(
                Path(__file__).parent / f'models/model_QInteractionNetwork_NodeEdgeProj_Conv1D_nconst_{N}_nbits_8.h5', oname='tt'
            )
            assert 'fixed<12,4,RND,SAT,0>' == str(model_hls.get_input_variables()[0].type.precision)
            q = FixedQ(12, 4, 1, 'RND', 'SAT')
            data = q(np.random.rand(5000 * 32 // N, N, 3).astype(np.float32) * 16 - 8)

            t0 = time()
            inp, out = trace_model(model_hls, optimize=False)
            comb = trace(inp, out)
            t1 = time()

            model_hls.compile()
            r_hls = model_hls.predict(data)
            r_keras = model.predict(data)
            r_comb = comb.predict(data)
            err = np.mean(np.abs((r_comb - r_keras)))
            assert np.all(r_hls == r_comb)  # Behavior shall match exactly
        print('hls4ml behavior matches exactly')

        print(f'N={N}, mean_keras_err: {err:.4f}, alkaid conversion time: {t1 - t0:.2f}s')

        t0 = time()
        rtl = RTLModel(comb, latency_cutoff=5, clock_period=4, clock_uncertainty=0, path=f'rtl/{N}-3', prj_name='model')
        rtl.write(no_shreg=True)
        t1 = time()
        print(f'RTL generation time: {t1 - t0:.2f}s')
        rtl._compile(nproc=8)
        assert np.all(r_comb == rtl.predict(data))
        print('RTL behavior matches exactly')
