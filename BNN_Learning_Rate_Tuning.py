from Bayesian_Neural_Network import bayesian_neural_network

from warnings import simplefilter
simplefilter('ignore')

import numpy as np
from time import time
import matplotlib.pyplot as plt

class bnn_learning_rate_tuning():
    def __init__(self, input_layer, hidden_layers, output_layer, feature_data, target_data, error_type='mse', window_size=None, initial_lr_power=1, end_lr_power=10, total_iters=50, tuning_epochs=25):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.feature_data = feature_data
        self.target_data = target_data
        self.error_type = error_type
        self.window_size = window_size
        self.initial_lr_power = initial_lr_power
        self.end_lr_power = end_lr_power
        self.total_iters = total_iters
        self.tuning_epochs = tuning_epochs

        self.final_iter_mean_error = []
        self.final_iter_std_error = []        

    def generate_all_learning_rates(self):
        self.learning_rates = (10 ** (-1 * self.initial_lr_power)) * (10 ** (-1 * np.arange(self.total_iters) * (self.end_lr_power - self.initial_lr_power) / self.total_iters))

        return
    
    def _preparation_for_learning_rate_tuning(self, learning_rate):
        # create an instance of bayesian neural network for each new learning rate
        bnn_tuning = bayesian_neural_network(self.input_layer, 
                                                self.hidden_layers, 
                                                self.output_layer, 
                                                self.feature_data, 
                                                self.target_data, 
                                                error_type=self.error_type, 
                                                window_size=self.window_size, 
                                                learning_rate=learning_rate)

        # prepares the data before training
        if self.error_type == 'accuracy':
            bnn_tuning.standardize_dataset()
        else:  
            bnn_tuning.generate_windowed_dataset()
            bnn_tuning.standardize_windowed_dataset()

        # initialize the model weight's means and variances
        bnn_tuning.generate_m()
        bnn_tuning.generate_v()

        return bnn_tuning
     
    def perform_learning_rate_tuning(self):
        """
        measure the performance of the model using various learning rate
        """
        # generate all decayed learning rate
        self.generate_all_learning_rates()

        # itrerate over all learning rates
        for current_iter, lr in enumerate(self.learning_rates):
            start_time = time() # initialize the time where the training prcoess starts

            # initialize an instance of class from the main bayesian neural netowork class
            bnn_tuning = self._preparation_for_learning_rate_tuning(lr)

            # specify which type of error funtion to be used for measuring the model's perfomance
            if self.error_type == 'accuracy':
                error_func = bnn_tuning._calculate_prediction_accuracy
            else:
                error_func = bnn_tuning._calculate_prediction_mse

            # trains the model
            for epoch in range(self.tuning_epochs):
                bnn_tuning._training_sequences(lr, self.tuning_epochs, epoch + 1)
                
            # calculate and saves the model's performance based on the current learning rate
            final_mean_error, final_std_error = bnn_tuning._calculating_errors_sequences(error_func, bnn_tuning.feature_data, bnn_tuning.target_data)
            self.final_iter_mean_error.append(final_mean_error)
            self.final_iter_std_error.append(final_std_error)

            # since the print function utilize the global variables mean_error and std_error in bnn_tuning, we will append the calculated errors into it
            bnn_tuning.mean_error.append(final_mean_error)
            bnn_tuning.std_error.append(final_std_error)
            bnn_tuning._print_current_epoch_training_result(self.total_iters, current_iter + 1, lr, start_time, 'Iterations') # prints the training process result
            
            print(150 * '-') # just prints a line
        
        return

    def visualize_learning_rate_tuning(self):
        """
        visualize the model performance from using various learning rate
        """
        # set the figure for the visualization
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(20, 15)
        ax1.grid(True)
        ax2.grid(True)

        # create the visualization
        ax1.semilogx(self.learning_rates, self.final_iter_mean_error)
        ax2.semilogx(self.learning_rates, self.final_iter_std_error)

        # add labels to the visualization
        ax1.tick_params('both', length=10, width=1, which='both')
        ax2.tick_params('both', length=10, width=1, which='both')

        if self.error_type == 'accuracy':
            ax1.set_title('Prediction\'s Accuracy On Various Learning Rate')
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('Prediction\'s Accuracy')
        else:
            ax1.set_title('Prediction\'s MSE On Various Learning Rate')
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('Prediction\'s MSE')

        ax2.set_title('Prediction\'s Standard Deviation On Various Learning Rate')
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Prediction\'s Standard Deviation')

        fig.show()

        return