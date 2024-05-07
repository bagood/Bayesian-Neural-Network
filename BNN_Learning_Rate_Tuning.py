from Bayesian_Neural_Network import bayesian_neural_network

from warnings import simplefilter
simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt

class bnn_learning_rate_tuning():
    def __init__(self, input_layer, hidden_layers, output_layer, feature_data, target_data, window_size=1, initial_lr_power=10, end_lr_power=20, total_iters=10, tuning_epochs=25):            
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.feature_data = feature_data
        self.target_data = target_data
        self.window_size = window_size
        self.initial_lr_power = initial_lr_power
        self.end_lr_power = end_lr_power
        self.total_iters = total_iters
        self.tuning_epochs = tuning_epochs

        self.final_iter_mean_error = []
        self.final_iter_std_error = []

    def learning_rate_decay(self):
        self.learning_rates = (10 ** (-1 * self.initial_lr_power)) * (10 ** (-1 * np.arange(self.total_iters + 1) * (self.end_lr_power - self.initial_lr_power) / self.total_iters))

        return
     
    def learning_rate_tuning(self):
        """
        measure the performance of the model using various learning rate
        """
        self.learning_rate_decay()
        for lr in self.learning_rates:
            # create an instance of bayesian neural network for each new learning rate
            bnn_tuning = bayesian_neural_network(self.input_layer, 
                                                    self.hidden_layers, 
                                                    self.output_layer, 
                                                    self.feature_data, 
                                                    self.target_data, 
                                                    window_size=self.window_size,
                                                    learning_rate = lr
                                                    )
            # prepares the data and both the weight's mean and variance
            if self.window_size > 1:
                bnn_tuning.generate_windowed_dataset()                                                    
            bnn_tuning.standardize_dataset()
            bnn_tuning.generate_m()
            bnn_tuning.generate_v()
            # train the model
            for _ in range(self.tuning_epochs):
                bnn_tuning._training_sequences(lr)
                
            # saves the learning rate and final model performance
            bnn_tuning._calculating_errors_sequences()
            self.final_iter_mean_error.append(bnn_tuning.mean_error[-1])
            self.final_iter_std_error.append(bnn_tuning.std_error[-1])
        
        return
    
    def visualize_learning_rate_tuning(self):
        """
        visualize the model performance from using various learning rate
        """
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(20, 15)
        ax1.grid(True)
        ax2.grid(True)

        ax1.semilogx(self.learning_rates, np.exp(self.final_iter_mean_error))
        ax1.tick_params('both', length=10, width=1, which='both')
        ax1.axis([10 ** (-1 * self.end_lr_power), 10 ** (-1 * self.initial_lr_power), 0, self.total_iters])
        ax1.set_title('Prediction Mean MSE On Various Learning Rate')
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Prediction Mean MSE')

        ax2.semilogx(self.learning_rates, np.exp(self.final_iter_std_error))
        ax2.tick_params('both', length=10, width=1, which='both')
        ax2.axis([10 ** (-1 * self.end_lr_power), 10 ** (-1 * self.initial_lr_power), 0, self.total_iters])
        ax2.set_title('Prediciton Standard Devitaion On Various Learning Rate')
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Prediction Standard Deviation')

        fig.show()

        return