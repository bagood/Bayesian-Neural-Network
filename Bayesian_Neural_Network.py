from BNN_Forward_Propagation import bnn_forward_propagation
from BNN_Probabilistic_Back_Propagation import bnn_probabilistic_back_propagation

from warnings import simplefilter
simplefilter('ignore')

import numpy as np
from time import time
import matplotlib.pyplot as plt

class bayesian_neural_network():
    def __init__(self, input_layer, hidden_layers, output_layer, feature_data, target_data, error_type, window_size=None, learning_rate=None, initial_lr=None, end_lr=None):        
        # initilize all values required to build a bayesian neural network model
        self.model_structure = np.concatenate((input_layer, hidden_layers, output_layer) ) # create a list containing number of neurons on each layers
        self.error_type = error_type # create a global variable for determining the type of error that measures the model's perofrmance
        
        # create a global value for the window size value used to create the time-series feature data
        if window_size != None:
            self.window_size = window_size
        
        # the prediction activation function, feature data, and target data will be set accordingly based on the error type
        if error_type == 'mse':
            # the settings for time-series data
            self.transorm_pred_func = 'log'
            self.feature_data = np.exp(feature_data)
            self.target_data = np.exp(target_data)
        else:
            # the settings for non time-series data
            self.transorm_pred_func = 'sigmoid'
            self.feature_data = feature_data
            self.target_data = target_data

        # raises an error if learning rate, initial learning rate, and end learning rate present as inputs
        if (learning_rate != None) and ((initial_lr != None) and (end_lr != None)):
            raise ValueError('There should be only either learning rate or initial and end learning rate')
        
        if learning_rate != None:
            # the learning rate will be the same for every epochs
            self.learning_rate = learning_rate
        elif (initial_lr != None) and (end_lr != None):
            # the learning rate will be decayed over time based on decayed learning rate type
            self.initial_lr = initial_lr # the initial learning rate
            self.end_lr = end_lr # the final learning rate

        # create global variables that stores model performance on each epochs
        self.mean_error = []
        self.std_error = []

        # create instances of class for the forward and backward propagation object
        self.bnn_fp = bnn_forward_propagation()
        self.bnn_pbp = bnn_probabilistic_back_propagation(self.transorm_pred_func)
    
    def generate_windowed_dataset(self):
        """
        generate a windowed feature data and target data based on the number window size
        this function is made spcefically for time-series data
        """
        self.feature_data = np.concatenate([self.target_data[i:-self.window_size+i].reshape(-1, 1) for i in range(self.window_size)], axis=1).reshape(-1, self.window_size, 1)
        self.target_data = self.target_data[self.window_size:]
        
        return
    
    def standardize_windowed_dataset(self):
        """
        standardize both windowed feature data and target data by applying an exponential function, then divide it with the max value of the corresponding windowed feature data
        performing transformations as such ensures all values on the windowed feature data are non-positive valued and standardized to help improve the training process
        this function is made spcefically for time-series data
        """
        self.feature_data_scaled = (self.feature_data.reshape(-1, self.window_size) / np.max(self.feature_data, axis=1).reshape(-1, 1)).reshape(-1, self.window_size, 1)
        self.target_data_scaled = (self.target_data.reshape(-1, 1) / np.max(self.feature_data, axis=1).reshape(-1, 1)).reshape(-1, 1, 1)

        return
    
    def standardize_dataset(self):
        """
        standardize both feature data and target data by applying an exponential function, then divide it with the max value of the corresponding windowed feature data
        performing transformations as such helps improve the training process
        """
        self.feature_data_scaled = self.feature_data.reshape(-1, self.feature_data.shape[1], 1) / np.max(self.feature_data, axis=1).reshape(-1, 1, 1)
        self.target_data_scaled = self.target_data.reshape(-1, 1, 1)

        return

    def _generate_m(self, n_origin_neurons, n_destination_neurons):
        """
        create an initial weight's mean for all neurons in a layer

        Args:
        n_origin_neurons (integer) - number of neurons in the previous layer
        n_destination_neurons (integer) - number of neurons in the current layer
        """
        return np.random.normal(size=(n_destination_neurons, n_origin_neurons))

    def _generate_v(self, n_origin_neurons, n_destination_neurons):
        """
        create an initial weight's variance for all neurons in a layer
        since variance are always non-positive, it is necessary to apply an absolute function to ensure that
        
        Args:
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

        # divide the variance by 10 for all variances that are larger than the corresponding mean
        # for i in range(len(self.m)):
        #     while True:
        #         m_index_less_than_v = (np.abs(self.m[i]) < self.v[i]).reshape(1, -1)[0]
        #         self.v[i].reshape(1, -1)[0][m_index_less_than_v] /= 2
                
        #         if ~np.any((np.abs(self.m[i]) < self.v[i]).reshape(1, -1)[0]):
        #             break

        return

    def _calculate_prediction_mse(self, pred_mean):
        """
        calculate the prediction's mean squared error

        Args:
        pred_mean (1D array of floats) - the model's prediction on based on the feature data
        """
        return np.mean((np.log(self.target_data) - pred_mean) ** 2)
    
    def _calculate_prediction_accuracy(self, pred_mean):
        """
        calculate the prediction's accuracy

        Args:
        pred_mean (1D array of floats) - the model's prediction on based on the feature data
        """
        return 100 * np.sum(self.target_data == np.round(pred_mean)) / len(self.target_data)

    def _print_current_epoch_training_process(self, total_epochs, current_epoch, current_lr, total_trained, start_time):
        """
        prints the current condition of the training process

        Args:
        total_epochs (integer) - the total number of epochs for training the model
        current_epoch (integer) - the current epoch for training the model
        current_lr (float) - the current learning rate used to train the model
        total_trained (integer) - the total number of feature data used to train the model
        start_time (integer) - the initial time where the current epoch starts
        """
        time_passed = np.round(time() - start_time, 2) # calculate the time passed from the initial start time
        # create the text that will be printed
        text = f'Epoch: {current_epoch} / {total_epochs} - Learning Rate : {current_lr} - Succesfull Train Percentage : {self.succesfull_train / total_trained * 100}% - Time Passed : {time_passed} Second'
        print(text, end='\r')

        return

    def _training_sequences(self, learning_rate, total_epochs, current_epoch):
        """
        the ordered sequences to train the model

        Args:
        learning_rate (float) - the learning rate used to train the model
        total_epochs (integer) - the total number of epochs for training the model
        current_epoch (integer) - the current epoch for training the model
        """ 
        start_time = time() # initialize the time where the training prcoess starts
        self.succesfull_train = 0 # create a global value to count the number of succesful train
    
        # iterate over all feature data
        for total_trained, (feature_data_scaled_i, target_data_scaled_i) in enumerate(zip(self.feature_data_scaled, self.target_data_scaled)):
            # perform the forward propagation to acquire all of variables on each neuron
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

            # if there's a nan values in the gradient, the weight's mean and variance won't be updated
            if (~np.isnan(d_logz_over_m[-1]).any()) and (~np.isnan(d_logz_over_v[-1]).any()):
                # optimize the model's weights
                self.m = self.bnn_pbp.optimize_m(self.m, 
                                                    self.v, 
                                                    d_logz_over_m,
                                                    learning_rate)
                self.v = self.bnn_pbp.optimize_v(self.m, 
                                                    self.v, 
                                                    d_logz_over_m, 
                                                    d_logz_over_v,
                                                    learning_rate)    
                self.v = [np.abs(v) for v in self.v] # making sure that the variances are always non-negative
                
                self.succesfull_train += 1 # add 1 into the variable because the training is succesful
            
            # prints the current training process
            self._print_current_epoch_training_process(total_epochs, current_epoch, learning_rate, total_trained+1, start_time)
        
        return 
    
    def _calculating_errors_sequences(self, error_func):
        """
        the ordered sequences to calculate the model's performance

        Args:
        error_func (function) - the function used to train the model
        """
        # make prediction based on the model and calculate the prediction's mean and standard deviation
        pred_mean, pred_std = self.bnn_fp.feed_forward_neural_network(self.m, self.v, self.feature_data, self.model_structure, transorm_pred_func=self.transorm_pred_func)

        # calculate the error and add it into their respective list
        self.mean_error.append(error_func(pred_mean))
        self.std_error.append(np.mean(pred_std))

        return

    def _exponential_learning_rate_decay(self, total_epochs):
        """
        create an array of learning rates used to train the model
        the learning rates are acquired from decaying the initial learning rate in an exponential fashion
        
        Args:
        total_epochs (integer) - the total number of epochs for training the model
        """
        self.learning_rates = self.initial_lr / ((self.initial_lr / self.end_lr) * np.arange(1, total_epochs + 1) / total_epochs)
        return  

    def _print_current_epoch_training_result(self, total_epochs, current_epoch, current_lr, start_time):
        """
        prints the current training result

        Args:
        total_epochs (integer) - the total number of epochs for training the model
        current_epoch (integer) - the current epoch for training the model
        current_lr (float) - the current learning rate used to train the model
        start_time (integer) - the initial time where the current epoch starts
        """
        time_passed = np.round(time() - start_time, 2) # calculate the time passed from the initial start time
        
        # create texts that will be printed
        text_1 = f'Epoch : {current_epoch} / {total_epochs} - Learning Rate : {current_lr} - Succesfull Train Percentage : {self.succesfull_train / len(self.feature_data) * 100}% - Time Passed : {time_passed} Second'
        if self.transorm_pred_func == 'log':
            text_2 = f'MSE : {self.mean_error[-1]} - Standard Deviation : {self.std_error[-1]}'
        else:
            text_2 = f'Accuracy : {self.mean_error[-1]}% - Standard Deviation : {self.std_error[-1]}'

        # prints all texts
        print(150 * '-')
        print(text_1)
        print(text_2)
        
        return

    def train_model(self, total_epochs, learning_rate_decay_type=False):
        """
        perform forward propagation to acquire all necessary variables, then perform backward propagation to optimize the model weight's mean and variance

        Args:
        total_epochs (integer) - the total number of epochs for training the model
        learning_rate_decay_type (string) - the learning rate decay type
        """
        # generate all learning rates based on the learning rate decay type
        if learning_rate_decay_type == False:
            self.learning_rates = np.ones(total_epochs) * self.learning_rate
        elif learning_rate_decay_type == 'exponential':
            self._exponential_learning_rate_decay(total_epochs)
        
        # specify which type of error funtion to be used for measuring the model's perfomance
        if self.error_type == 'accuracy':
            error_func = self._calculate_prediction_accuracy
        else:
            error_func = self._calculate_prediction_mse
        
        # start the training process
        for epoch, lr in enumerate(self.learning_rates):
            start_time = time() # initialize the time where the training prcoess starts
            self._training_sequences(lr, total_epochs, epoch + 1) # trains the model
            self._calculating_errors_sequences(error_func) # calculate the model's performance
            self._print_current_epoch_training_result(total_epochs, epoch + 1, lr, start_time) # prints the training process result
        
        return         
    
    def visualize_performance(self):
        """
        visualize the model's performance throughout the training process
        """
        # set the figure for the visualization
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(20, 10)

        # visualize the model's performance based on the calculated prediction's error
        ax1.plot(self.mean_error)         
        if self.error_type == 'mse':
            ax1.set_title('MSE Throughout Training')
        else:
            ax1.set_title('Accuracy Throughout Training')
        
        # visualize the model's performance based on the prediction's standard deviation
        ax2.plot(self.std_error)
        ax2.set_title('Variance Throughout Training')

        # shows the visualizations
        fig.show()

        return 
    
    def visualize_predictions_on_seen_data(self):
        """
        visualize both the prediction's and actual target values based on the corresponding feature data as well as visualizing the prediction's confidence interval
        this function is made spcefically for time-series data 
        """
        # make predictions based on feature data used during training process
        self.predictions_mean, self.predictions_std = self.bnn_fp.feed_forward_neural_network(self.m, self.v, self.feature_data, self.model_structure, transorm_pred_func=self.transorm_pred_func)

        # calculate the upper and lower bound of the prediction
        self.upper_bound = self.predictions_mean + self.predictions_std
        self.lower_bound = self.predictions_mean - self.predictions_std

        # set the figure for the visualization
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)

        x_axis = np.arange(0, len(self.feature_data)) # sets the x-axis for the visualization
        ax.plot(x_axis, np.log(self.target_data.reshape(1, -1)[0]), color='black', label='Mean') # visualize the actual target data
        ax.plot(x_axis, self.predictions_mean, color='green', label='Mean') # visualize the predictions
        ax.plot(x_axis, self.upper_bound, color='red', label='Upper') # visualize the prediction's upper bound
        ax.plot(x_axis, self.lower_bound, color='red', label='Lower') # visualize the prediction's lower bound
        ax.fill_between(x_axis, self.upper_bound, self.lower_bound, color="blue", alpha=0.15) # fills a color between the upper and lower bound visualizations
        ax.legend(['Data', 'Prediction Mean', 'Upper Bound x% Confidence Interval', 'Lower Bound x% Confidence Interval', 'Confidence Interval']) # add legends to the visualizations

        # shows the visualizations
        fig.show()

        return