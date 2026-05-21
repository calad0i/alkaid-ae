import numpy as np
from alkaid.codegen import RTLModel
from alkaid.converter import trace_model
from alkaid.trace import FVArray, trace
from keras.datasets import mnist
from xgboost import XGBClassifier


def leaf_quantizer_1(x):
    v = np.round(x * 4)
    _min = np.sort(v.ravel())[2]
    return np.maximum(v - _min, 0), -_min


def leaf_quantizer_2(x):
    v = np.round(x * 2)
    _min = np.sort(v.ravel())[2]
    return np.maximum(v - _min, 0), -_min


def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train_quantized = np.floor(X_train / 2**7)
    X_test_quantized = np.floor(X_test / 2**7)
    return X_train_quantized, X_test_quantized, y_train, y_test


def convert_and_test(model: XGBClassifier, name: str, data, leaf_quantizer, n_stages, clock_period):

    X_train_quantized, X_test_quantized, y_train, y_test = data
    inp = FVArray.new(28**2).quantize(0, 1, 0).as_new()
    bst = model._Booster
    _, out = trace_model(bst, inputs=inp, leaf_quantizer=leaf_quantizer, mode='mux')
    comb = trace(inp, out)

    train_acc = np.mean(np.argmax(comb.predict(X_train_quantized), axis=1) == y_train)
    test_acc = np.mean(np.argmax(comb.predict(X_test_quantized), axis=1) == y_test)
    print(f'train_acc_hw: {train_acc:.3f}, test_acc_hw: {test_acc:.3f}')

    rtl = RTLModel(comb, f'{name}', 'model', n_stages=n_stages, clock_period=clock_period, clock_uncertainty=0)
    rtl.write(xls_opt=True, metadata={'comb_metric': float(test_acc)})

    rtl._compile(_env={'VERILATOR_FLAGS': ''}, nproc=4)  # verilating and bit-exact simulating
    assert np.all(rtl.predict(X_test_quantized, n_threads=4) == comb.predict(X_test_quantized, n_threads=4))
    print('verilator check passed')


if __name__ == '__main__':
    data = load_data()
    X_train_quantized, X_test_quantized, y_train, y_test = data

    model = XGBClassifier(num_class=10, n_estimators=30, max_depth=5, eta=0.8, worker_count=8)
    model.fit(X_train_quantized, y_train)
    y_pred_xgb = model.predict(X_test_quantized)
    print(f'sw acc: {np.mean(y_pred_xgb == y_test):.3f}')

    convert_and_test(model, 'rtl/mnist_large', data, leaf_quantizer_1, n_stages=2, clock_period=1.8)
    convert_and_test(model, 'rtl/mnist_small', data, leaf_quantizer_2, n_stages=2, clock_period=1.6)
