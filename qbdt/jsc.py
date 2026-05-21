import numpy as np
from alkaid.codegen import RTLModel
from alkaid.converter import trace_model
from alkaid.trace import FVArray, trace
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def get_data():
    data = fetch_openml('hls4ml_lhc_jets_hlf')
    X, y = np.array(data['data']), data['target']
    codecs = {'g': 0, 'q': 1, 'w': 2, 'z': 3, 't': 4}
    y = np.array([codecs[i] for i in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    _min, _max = X_train.min(axis=0), X_train.max(axis=0)
    X_train = np.floor((X_train - _min) / (_max - _min) * 255)
    X_test = np.floor((X_test - _min) / (_max - _min) * 255)
    return X_train, X_test, y_train, y_test


def leaf_quantizer_1(x):
    v = np.round(x * 7)
    _min = np.sort(v.ravel())[1] + 1
    return np.maximum(v - _min, 0), -_min


def leaf_quantizer_2(x):
    v = np.floor(x * 3)
    _min = np.sort(v.ravel())[1]
    return np.maximum(v - _min, 0), -_min


def convert_and_test(model: XGBClassifier, name: str, data, leaf_quantizer, n_stages, clock_period):

    X_train_quantized, X_test_quantized, y_train, y_test = data
    inp = FVArray.new(16).quantize(0, 8, 0).as_new()
    bst = model._Booster
    _, out = trace_model(bst, inputs=inp, leaf_quantizer=leaf_quantizer, mode='mux')
    comb = trace(inp, out)

    train_acc = np.mean(np.argmax(comb.predict(X_train_quantized), axis=1) == y_train)
    test_acc = np.mean(np.argmax(comb.predict(X_test_quantized), axis=1) == y_test)
    print(f'train_acc_hw: {train_acc:.3f}, test_acc_hw: {test_acc:.3f}')

    rtl = RTLModel(comb, f'{name}', 'model', n_stages=n_stages, clock_period=clock_period, clock_uncertainty=0)
    rtl.write(xls_opt=False, metadata={'comb_metric': float(test_acc)})

    rtl._compile(_env={'VERILATOR_FLAGS': ''}, nproc=4)  # verilating and bit-exact simulating
    assert np.all(rtl.predict(X_test_quantized, n_threads=4) == comb.predict(X_test_quantized, n_threads=4))
    print('verilator check passed')


if __name__ == '__main__':
    data = get_data()
    X_train, X_test, y_train, y_test = data

    model = XGBClassifier(num_class=5, n_estimators=13, max_depth=5, eta=0.8)
    model.fit(X_train, y_train)

    y_pred_xgb = model.predict(X_test)
    print(f'train_acc_sw: {np.mean(model.predict(X_train) == y_train):.3f}', end=', ')
    print(f' test_acc_sw: {np.mean(y_pred_xgb == y_test):.3f}')

    convert_and_test(model, 'rtl/jsc_large', data, leaf_quantizer_1, n_stages=2, clock_period=2)

    model = XGBClassifier(num_class=5, n_estimators=4, max_depth=5, eta=0.8)
    model.fit(X_train, y_train)
    y_pred_xgb = model.predict(X_test)
    print(f'train_acc_sw: {np.mean(model.predict(X_train) == y_train):.4f}', end=', ')
    print(f' test_acc_sw: {np.mean(y_pred_xgb == y_test):.4f}')

    convert_and_test(model, 'rtl/jsc_small', data, leaf_quantizer_2, n_stages=1, clock_period=2)
