import time

import joblib
import numpy as np


def predict_general(width, length, x_prev, y_prev, beta, theta, method, kind, version):
    var = -1
    pre_x = x_prev / width
    pre_y = y_prev / length
    data_point = [beta, theta, pre_x, pre_y]
    data_array = np.array(data_point, dtype=np.float32).reshape(1, -1)
    if method == 'DNN':
        # Load the scaler from the file
        loaded_scaler = joblib.load(f'MODELS/scaler_angle_nn_optuna_ver5.pkl')
        loaded_model = joblib.load(f'MODELS/model_{kind}_nn_optuna_ver{version}.pkl')
        var = loaded_model.predict(loaded_scaler.transform(data_array))[0][0]
    if method == 'XGBoost':
        loaded_model = joblib.load(f'MODELS/model_{kind}_XGBoost_ver{version}.pkl')
        var = loaded_model.predict(data_array)[0]
    if method == 'XGBoost_T':
        loaded_scaler = joblib.load(f'MODELS/scaler_xgb_t.pkl')
        loaded_model = joblib.load(f'MODELS/model_{kind}_XGBoost_ver{version}.pkl')
        var = loaded_model.predict(loaded_scaler.transform(data_array))[0]
    if method == 'TabNet':
        loaded_model = joblib.load(f'MODELS/model_{kind}_tabnet_ver{version}.pkl')
        var = loaded_model.predict(data_array)[0][0]
    return var


def predict_Y1(width, length, x_prev, y_prev, beta, theta, method, version):
    return predict_general(width, length, x_prev, y_prev, beta, theta, method, 'Y1', version)


def predict_Y2(width, length, x_prev, y_prev, beta, theta, method, version):
    return predict_general(width, length, x_prev, y_prev, beta, theta, method, 'Y2', version)


def predict_angle(width, length, x_prev, y_prev, beta, theta, method, version):
    return predict_general(width, length, x_prev, y_prev, beta, theta, method, 'angle', version)


def predict_T(width, length, x_prev, y_prev, beta, theta, method, version):
    return predict_general(width, length, x_prev, y_prev, beta, theta, method, 'T', version)


def predict_J(width, length, x_prev, y_prev, beta, theta, method, version):
    return predict_general(width, length, x_prev, y_prev, beta, theta, method, 'J', version)

