from Bayesian_Neural_Network import bayesian_neural_network

from warnings import simplefilter
simplefilter('ignore')

import numpy as np
from time import time
import matplotlib.pyplot as plt

class bnn_learning_rate_tuning():
    def __init__(self, input_layer, 
                        hidden_layers, 
                        output_layer, 
                        feature_data, 
                        target_data, 
                        model_purpose='regression', 
                        window_size=None, 
                        initial_lr_power=1, 
                        end_lr_power=10, 
                        total_iters=50, 
                        tuning_epochs=25):

        # create global variables every variabels
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.feature_data = feature_data
        self.target_data = target_data
        self.model_purpose = model_purpose
        self.window_size = window_size
        self.initial_lr_power = initial_lr_power
        self.end_lr_power = end_lr_power
        self.total_iters = total_iters
        self.tuning_epochs = tuning_epochs

        # create global variables that stores the model's performance on each epochs
        self.final_iter_mean_error = []
        self.final_iter_std_error = []        

    def generate_all_learning_rates(self):
        """
        generate all learning rates that will be used during the learning rate tuning process
        """
        self.learning_rates = (10 ** (-1 * self.initial_lr_power)) * (10 ** (-1 * np.arange(self.total_iters) * (self.end_lr_power - self.initial_lr_power) / self.total_iters))

        return

    def _print_current_iter_training_result(self, current_iter, learning_rate, start_time):
        """
        prints the current tuning iter result based on the training data

        Args:
        current_iter (integer) - the current tuning iterations
        learning_rate (float) - the current learning rate used to train the model
        start_time (integer) - the initial time where the current epoch starts
        """
        time_passed = np.round(time() - start_time, 2) # calculate the time passed from the initial start time
        
        # create texts that will be printed
        text_1 = f'Iter : {current_iter} / {self.total_iters} - Learning Rate : {learning_rate} - Time Passed : {time_passed} Second'
        
        if self.model_purpose == 'regression': # the specific text for a regression task
            text_2 = f'MSE : {self.final_iter_mean_error[-1]} - Standard Deviation : {self.final_iter_std_error[-1]}'
        else: # the specific text for a binary classification task
            text_2 = f'Accuracy : {self.final_iter_mean_error[-1]}% - Standard Deviation : {self.final_iter_std_error[-1]}'

        # print all texts
        print(150 * '-')
        print(text_1)
        print(text_2)
        
        return

    def perform_learning_rate_tuning(self):
        """
        train the model using various learning rates, then measure the model's performance for each learning rates
        """
        # generate all decayed learning rate
        self.generate_all_learning_rates()

        # itrerate over all learning rates
        for current_iter, learning_rate in enumerate(self.learning_rates):
            start_time = time() # initialize the time where the training prcoess starts

            # initialize an instance of class from the main bayesian neural netowork class
            bnn_tuning = bayesian_neural_network(self.input_layer, 
                                                    self.hidden_layers, 
                                                    self.output_layer, 
                                                    self.feature_data, 
                                                    self.target_data, 
                                                    validation_percentage=None, 
                                                    model_purpose=self.model_purpose, 
                                                    window_size=self.window_size, 
                                                    learning_rate=learning_rate,
                                                    learning_rate_decay_type=False,
                                                    total_epochs=self.tuning_epochs)  

            # specify which type of error funtion to be used for measuring the model's perfomance
            if self.model_purpose == 'regression':
                error_func = bnn_tuning._calculate_prediction_mse
            else:
                error_func = bnn_tuning._calculate_prediction_accuracy

            # trains the model
            for epoch in range(self.tuning_epochs):
                bnn_tuning._training_sequences(learning_rate, epoch + 1)
                
            # calculate and saves the model's performance based on the current learning rate
            final_mean_error, final_std_error = bnn_tuning._calculating_errors_sequences(bnn_tuning.feature_data, bnn_tuning.target_data)
            self.final_iter_mean_error.append(final_mean_error)
            self.final_iter_std_error.append(final_std_error)

            self._print_current_iter_training_result(current_iter + 1, learning_rate, start_time) # prints the tuning process result
            
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

        if self.model_purpose == 'regression':
            ax1.set_title('Prediction\'s MSE On Various Learning Rate')
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('Prediction\'s MSE')
        else:
            ax1.set_title('Prediction\'s Accuracy On Various Learning Rate')
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('Prediction\'s Accuracy')

        ax2.set_title('Prediction\'s Standard Deviation On Various Learning Rate')
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Prediction\'s Standard Deviation')

        fig.show()

        return