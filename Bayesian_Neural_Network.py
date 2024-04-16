from BNN_Forward_Propagation import bnn_forward_propagation
from BNN_Probabilistic_Back_Propagation import bnn_probabilistic_back_propagation

import numpy as np

input_layer = [1]
hidden_layers = [3, 3]
output_layer = [1]

feature_data = np.array([[1]])
target_data = np.array([[1]])

learning_rate = 0.0001

bnn_fp = bnn_forward_propagation(input_layer, hidden_layers, output_layer, feature_data, target_data, learning_rate)

model_structure = bnn_fp.generate_model_structure()
m = bnn_fp.generate_m(model_structure)
v = bnn_fp.generate_v(model_structure)

bnn_pbp = bnn_probabilistic_back_propagation(model_structure, target_data, learning_rate)

for i in range(5000):
    ma, va, cdf, minus_cdf, pdf, gamma, alpha, mz, vz = bnn_fp.forward_propagation(m, v, model_structure)

    if i % 50 == 0:
        print(mz[-1][0, 0], vz[-1][0, 0])

    d_logz_over_m, d_logz_over_v = bnn_pbp.calculate_derivatives(m, v, ma, va, cdf, minus_cdf, pdf, gamma, alpha, mz, vz)
    m = bnn_pbp.optimize_m(m, v, d_logz_over_m)
    v = bnn_pbp.optimize_v(m, v, d_logz_over_m, d_logz_over_v)