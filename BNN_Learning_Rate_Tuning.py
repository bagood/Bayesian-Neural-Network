from Bayesian_Neural_Network import bayesian_neural_network

from warnings import simplefilter
simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt

class bnn_learning_rate_tuning():
    def __init__(self, input_layer, hidden_layers, output_layer, feature_data, target_data, initial_lr, end_lr, total_iters, tuning_epochs, window_size=1):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.feature_data = feature_data
        self.target_data = target_data
        self.window_size = window_size
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_iters = total_iters
        self.tuning_epochs = tuning_epochs

        self.final_iter_mean_error = []
        self.final_iter_std_error = []
    
    def _linear_learningrate_decay(self, current_iter):
        """
        decay the learning rate in a linear fashion
        """
        return  self.initial_lr - ((self.initial_lr - self.end_lr) * (current_iter / self.total_iters))
    
    def generate_all_decayed_learning_rate(self):
        """
        generate all decayed learning rate
        """
        self.all_lr = np.array([self._linear_learningrate_decay(current_iter) for current_iter in range(1, self.total_iters+1)])
        return 

    def learning_rate_tuning(self):
        """
        measure the performance of the model using various learning rate
        """
        for lr in self.all_lr:
            # create an instance of bayesian neural network for each new learning rate
            bnn_tuning = bayesian_neural_network(self.input_layer, 
                                                    self.hidden_layers, 
                                                    self.output_layer, 
                                                    self.feature_data, 
                                                    self.target_data, 
                                                    lr)
            bnn_tuning.standardize_dataset()
            bnn_tuning.generate_m()
            bnn_tuning.generate_v()
            # train the model
            bnn_tuning.train_model(self.tuning_epochs)
            # saves the final model performance
            self.final_iter_mean_error.append(bnn_tuning.mean_error[-1])
            self.final_iter_std_error.append(bnn_tuning.std_error[-1])
        
        return
    
    def visualize_learning_rate_tuning(self):
        """
        visualize the model performance from using various learning rate
        """
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(20, 10)

        ax1.plot(-1 * self.all_lr, self.final_iter_mean_error)
        ax1.set_title('Prediction Mean MSE On Various Learning Rate')
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Prediction Mean MSE')
        ax2.plot(-1 * self.all_lr, self.final_iter_std_error)
        ax2.set_title('Prediciton Standard Devitaion On Various Learning Rate')
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Prediction Standard Deviation')

        fig.show()

        return