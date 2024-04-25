from BNN_Forward_Propagation import bnn_forward_propagation
from BNN_Probabilistic_Back_Propagation import bnn_probabilistic_back_propagation

from warnings import simplefilter
simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt

class bayesian_neural_network():
    def __init__(self, input_layer, hidden_layers, output_layer, feature_data, target_data, learning_rate):
        self.model_structure = np.concatenate((input_layer, hidden_layers, output_layer))
        self.feature_data = np.array([[[val]] for val in feature_data])
        self.target_data = np.array([[[val]] for val in feature_data])

        self.error = []
        self.variance = []

        self.bnn_fp = bnn_forward_propagation()
        self.bnn_pbp = bnn_probabilistic_back_propagation(learning_rate)

    def _generate_m(self, n_origin_neurons, n_destination_neurons):
        return np.random.random(size=(n_destination_neurons, n_origin_neurons))

    def _generate_v(self, n_origin_neurons, n_destination_neurons):
        return np.abs(np.random.random(size=(n_destination_neurons, n_origin_neurons)))
        
    def generate_m(self):
        self.m = [self._generate_m(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-1], self.model_structure[1:])]

        return

    def generate_v(self):
        self.v = [self._generate_v(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-1], self.model_structure[1:])]
    
        return
    
    def calculate_rmse_error(self, feature_data_i, target_data_i):
        return (((target_data_i[0, 0] / feature_data_i[0, 0]) - self.mz[-1][0, 0]) ** 2) ** 0.5

    def train(self, epochs):
        for epoch in range(epochs):
            for feature_data_i, target_data_i in zip(self.feature_data, self.target_data):
                forward_propagation_result = self.bnn_fp.forward_propagation(feature_data_i / feature_data_i, self.m, self.v, self.model_structure)
                d_logz_over_m, d_logz_over_v = self.bnn_pbp.calculate_derivatives(self.model_structure, target_data_i / feature_data_i, self.m, self.v, forward_propagation_result)
                
                self.m = self.bnn_pbp.optimize_m(self.m, self.v, d_logz_over_m)
                self.v = self.bnn_pbp.optimize_v(self.m, self.v, d_logz_over_m, d_logz_over_v)

                _, _, _, _, _, _, _, self.mz, self.vz = forward_propagation_result
                self.error.append(self.calculate_rmse_error(feature_data_i, target_data_i))
                self.variance.append(self.vz[-1][0, 0])
        
        return         
    
    def visualize_performance(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(20, 10)

        ax1.plot(self.error)
        ax2.plot(self.variance)

        fig.show()

        return 

    def _sampling_from_normal_dist(self):
        return [np.random.normal(self.mz[-1][0, 0], self.vz[-1][0, 0] ** 0.5) for _ in range(250)]
    
    def predict(self):
        self.prediction_mean = np.array([np.mean(self._sampling_from_normal_dist()) for feature_data_i in self.feature_data]).reshape(1, -1)[0] * self.target_data.reshape(1, -1)[0]
        self.prediction_std = np.array([np.std(self._sampling_from_normal_dist()) for feature_data_i in self.feature_data]).reshape(1, -1)[0] * self.target_data.reshape(1, -1)[0]

        return (self.prediction_mean, self.prediction_std)
    
    def visualize_predictions(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)

        upper = self.prediction_mean + self.prediction_std
        lower = self.prediction_mean - self.prediction_std

        ax.plot(self.feature_data.reshape(1, -1)[0], self.prediction_mean, color='green', label='Mean')
        ax.plot(self.feature_data.reshape(1, -1)[0], upper, color='red', label='Upper')
        ax.plot(self.feature_data.reshape(1, -1)[0], lower, color='red', label='Lower')
        ax.fill_between(self.feature_data.reshape(1, -1)[0], upper, lower, color="blue", alpha=0.15)

        fig.show()

        return