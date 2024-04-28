from BNN_Forward_Propagation import bnn_forward_propagation
from BNN_Probabilistic_Back_Propagation import bnn_probabilistic_back_propagation

from warnings import simplefilter
simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt

class bayesian_neural_network():
    def __init__(self, input_layer, hidden_layers, output_layer, feature_data, target_data, learning_rate):
        self.model_structure = np.concatenate((input_layer, hidden_layers, output_layer))
        self.feature_data = np.exp(feature_data)
        self.target_data = np.exp(target_data)

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

    def calculate_rmse_error(self, target_data_i, pred_mean_i):
        return (np.log(target_data_i) - pred_mean_i)[0, 0] ** 2
    
    def calculate_prediction_variance(self, target_data_i, pred_variance_i):
        return np.abs(np.log(target_data_i) - pred_variance_i)[0 ,0]

    def _standardize_feature_data(self, feature_data_i):
        return feature_data_i / np.max(feature_data_i)
    
    def _standardize_target_data(self, feature_data_i, target_data_i):
        return target_data_i / np.max(feature_data_i)

    def train(self, epochs):
        for epoch in range(epochs):
            for feature_data_i, target_data_i in zip(self.feature_data, self.target_data):
                forward_propagation_result = self.bnn_fp.forward_propagation(self._standardize_feature_data(feature_data_i), 
                                                                                self.m, 
                                                                                self.v, 
                                                                                self.model_structure)
                d_logz_over_m, d_logz_over_v = self.bnn_pbp.calculate_derivatives(self.model_structure, 
                                                                                    self._standardize_target_data(feature_data_i, target_data_i), 
                                                                                    self.m, 
                                                                                    self.v, 
                                                                                    forward_propagation_result)
                self.m = self.bnn_pbp.optimize_m(self.m, 
                                                    self.v, 
                                                    d_logz_over_m)
                                                    
                self.v = self.bnn_pbp.optimize_v(self.m, 
                                                    self.v, 
                                                    d_logz_over_m, 
                                                    d_logz_over_v)
                
                pred_mean_i = self.bnn_fp.feed_forward_neural_network(self.m, feature_data_i)
                pred_var_i = self.bnn_fp.feed_forward_neural_network(self.v, feature_data_i)

                self.error.append(self.calculate_rmse_error(target_data_i, 
                                                                pred_mean_i))
                self.variance.append(self.calculate_prediction_variance(target_data_i,
                                                                            pred_var_i))

        return         
    
    def visualize_performance(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(20, 10)

        ax1.plot(self.error)
        ax1.set_title('RMSE Throughout Training')
        ax2.plot(self.variance)
        ax2.set_title('Variance Throughout Training')

        fig.show()

        return 

    def _sampling_from_normal_dist(self, pred_mean_i, var_pred_i):
        return [np.random.normal(pred_mean_i, var_pred_i ** 0.5) for _ in range(250)]
    
    def predict_on_seen_data(self):
        pred_mean = [self.bnn_fp.feed_forward_neural_network(self.m, feature_data_i)[0, 0] for feature_data_i in self.feature_data]
        pred_var = [np.abs(self.bnn_fp.feed_forward_neural_network(self.v, feature_data_i) - np.log(self.target_data[i]))[0, 0] for i, feature_data_i in enumerate(self.feature_data)]

        normal_sample = [self._sampling_from_normal_dist(pred_mean_i, pred_var_i) for pred_mean_i, pred_var_i in zip(pred_mean, pred_var)]
        self.prediction_mean = np.mean(normal_sample, axis=1)
        self.prediction_std = np.std(normal_sample, axis=1)

        return
    
    def visualize_predictions(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)

        self.upper_bound = self.prediction_mean + self.prediction_std
        self.lower_bound = self.prediction_mean - self.prediction_std

        x_plot = np.arange(0, len(self.feature_data))
        ax.plot(x_plot, np.log(self.target_data.reshape(1, -1)[0]), color='black', label='Mean')
        ax.plot(x_plot, self.prediction_mean, color='green', label='Mean')
        ax.plot(x_plot, self.upper_bound, color='red', label='Upper')
        ax.plot(x_plot, self.lower_bound, color='red', label='Lower')
        ax.fill_between(x_plot, self.upper_bound, self.lower_bound, color="blue", alpha=0.15)
        ax.legend(['Data', 'Prediction Mean', 'Upper Bound x% Confidence Interval', 'Lower Bound x% Confidence Interval', 'Confidence Interval'])

        fig.show()

        return