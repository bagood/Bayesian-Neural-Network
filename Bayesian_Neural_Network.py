from BNN_Forward_Propagation import bnn_forward_propagation
from BNN_Probabilistic_Back_Propagation import bnn_probabilistic_back_propagation

from warnings import simplefilter
simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt

class bayesian_neural_network():
    def __init__(self, input_layer, hidden_layers, output_layer, feature_data, target_data, learning_rate, window_size):
        self.model_structure = np.concatenate((input_layer, hidden_layers, output_layer)) # a list containing number of neurons on each layers
        # create variables for both feature and target data
        self.feature_data = feature_data
        self.target_data = target_data
        # standardize both feature and target data by applying an exponential function, then divide it with the max value of the feature data
        self.feature_data_scaled = (np.exp(feature_data).reshape(-1, window_size) / np.max(np.exp(feature_data), axis=1)).reshape(-1, window_size, 1)
        self.target_data_scaled = (np.exp(target_data).reshape(-1, 1) / np.max(np.exp(feature_data), axis=1)).reshape(-1, 1, 1)
        # empty list to store all errors 
        self.mean_error = []
        self.std_error = []
        # create instances of class for the forward and backward propagation object
        self.bnn_fp = bnn_forward_propagation()
        self.bnn_pbp = bnn_probabilistic_back_propagation(learning_rate)

    def _generate_m(self, n_origin_neurons, n_destination_neurons):
        """
        create an initial weight's mean for all neurons in a layer

        args:
        n_origin_neurons (integer) - number of neurons in the previous layer
        n_destination_neurons (integer) - number of neurons in the current layer
        """
        return np.random.random(size=(n_destination_neurons, n_origin_neurons))

    def _generate_v(self, n_origin_neurons, n_destination_neurons):
        """
        create an initial weight's variance for all neurons in a layer, ensuring that the value is positive

        args:
        n_origin_neurons (integer) - number of neurons in the previous layer
        n_destination_neurons (integer) - number of neurons in the current layer
        """
        return np.abs(np.random.random(size=(n_destination_neurons, n_origin_neurons)))
        
    def generate_m(self):
        """
        automate the process of generating all initial weight's mean on all layers
        """
        self.m = [self._generate_m(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-1], self.model_structure[1:])]

        return

    def generate_v(self):
        """
        automate the process of generating all initial weight's variance on all layers
        """
        self.v = [self._generate_v(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-1], self.model_structure[1:])]
    
        return

    def _calculate_prediction_mean_error(self, target_data_i, pred_mean_i):
        """
        calculate the mean squared error for the mean of the prediction

        Args:
        target_data_i (float) - the current feature data used to determine the mean of prediction
        pred_mean_i (float) - the current mean of prediction
        """
        return (target_data_i - pred_mean_i)[0, 0] ** 2
    
    def train(self, epochs):
        """
        perform forward propagation to acquire all necessary variables, then perform backward propagation to optimize the model weight's mean and variance

        Args:
        epochs (integer) - the number of iteration of training using the whole feature dataset
        """
        for epoch in range(epochs):
            zipped_var = zip(self.feature_data, self.target_data, self.feature_data_scaled, self.target_data_scaled)
            for feature_data_i, target_data_i, feature_data_scaled_i, target_data_scaled_i in zipped_var:
                # perform forward propagation
                forward_propagation_result = self.bnn_fp.forward_propagation(feature_data_scaled_i, 
                                                                                self.m, 
                                                                                self.v, 
                                                                                self.model_structure)
                # perform backward propagation to acquire the derivatives for optimizing the model's weights
                d_logz_over_m, d_logz_over_v = self.bnn_pbp.calculate_derivatives(self.model_structure, 
                                                                                    target_data_scaled_i, 
                                                                                    self.m, 
                                                                                    self.v, 
                                                                                    forward_propagation_result)
                # optimize the model's weights
                self.m = self.bnn_pbp.optimize_m(self.m, 
                                                    self.v, 
                                                    d_logz_over_m)
                self.v = self.bnn_pbp.optimize_v(self.m, 
                                                    self.v, 
                                                    d_logz_over_m, 

                                                    d_logz_over_v)

            # calculate the mean and variance of the predicton
            pred_mean_i, pred_std_i = self.bnn_fp.feed_forward_neural_network(self.m, self.v, feature_data_i)

            # calculate the errors and store in the list
            self.mean_error.append(self._calculate_prediction_mean_error(target_data_i, 
                                                                        pred_mean_i))
            self.std_error.append(pred_std_i)

        return         
    
    def visualize_performance(self):
        """
        visualize the model performance throughout the training process
        """
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(20, 10)

        ax1.plot(self.mean_error)
        ax1.set_title('MSE Throughout Training')
        ax2.plot(self.std_error)
        ax2.set_title('Variance Throughout Training')

        fig.show()

        return 

    def _sampling_from_normal_dist(self, pred_mean_i, pred_var_i):
        """
        take sample from a normal distirbution with the mean and variance provided by the input of the function

        Args:
        pred_mean_i (float) - the mean of the prediction
        pred_var_i (float) - the variance of the prediction
        """
        return [np.random.normal(pred_mean_i, pred_var_i ** 0.5) for _ in range(250)]
    
    def predict_on_seen_data(self):
        """
        create predictions on the data used in the training process
        """
        # create a list containing mean and variance of the prediction for each feature data
        predictions = np.array([self.bnn_fp.feed_forward_neural_network(self.m, self.v, feature_data_i) for feature_data_i in self.feature_data])
        self.predictions_mean = predictions[:, 0]
        self.predictions_std = predictions[:, 1]
    
        return
    
    def visualize_predictions_on_seen_data(self):
        """
        visualize the preiditions on data used throughout training alongside its confidence interval
        """
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)

        # calculate the upper and lower bound of the prediction
        self.upper_bound = self.predictions_mean + self.predictions_std
        self.lower_bound = self.predictions_mean - self.predictions_std

        x_axis = np.arange(0, len(self.feature_data))
        ax.plot(x_axis, self.target_data.reshape(1, -1)[0], color='black', label='Mean')
        ax.plot(x_axis, self.predictions_mean, color='green', label='Mean')
        ax.plot(x_axis, self.upper_bound, color='red', label='Upper')
        ax.plot(x_axis, self.lower_bound, color='red', label='Lower')
        ax.fill_between(x_axis, self.upper_bound, self.lower_bound, color="blue", alpha=0.15)
        ax.legend(['Data', 'Prediction Mean', 'Upper Bound x% Confidence Interval', 'Lower Bound x% Confidence Interval', 'Confidence Interval'])

        fig.show()

        return