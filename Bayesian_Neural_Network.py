from BNN_Forward_Propagation import bnn_forward_propagation
from BNN_Probabilistic_Back_Propagation import bnn_probabilistic_back_propagation

from warnings import simplefilter
simplefilter('ignore')

import numpy as np
from time import time
import matplotlib.pyplot as plt

class bayesian_neural_network():
    def __init__(self, input_layer, hidden_layers, output_layer, feature_data, target_data, validation_percentage=None, model_purpose='regression', window_size=None, learning_rate=None, initial_lr=None, end_lr=None):        
        # initilize all values required to build a bayesian neural network model
        self.model_structure = np.concatenate((input_layer, hidden_layers, output_layer) ) # create a list containing number of neurons on each layers
        self.validation_percentage = validation_percentage
        self.model_purpose = model_purpose
        
        # create a global value for the window size value used to create the time-series feature data
        if window_size != None:
            self.window_size = window_size
        
        # the prediction activation function, feature data, and target data will be set accordingly based on the error type
        if self.model_purpose == 'regression':
            # the settings for time-series data
            self.feature_data = np.exp(feature_data)
            self.target_data = np.exp(target_data)
            self.initialization_type = 'random'
            self.error_func = self._calculate_prediction_mse
        else:
            # the settings for non time-series data
            self.feature_data = feature_data.reshape(-1, feature_data.shape[-1], 1)
            self.target_data = target_data.reshape(1, -1)[0]
            self.initialization_type = 'xavier' # specify the weight's initialization method
            self.error_func = self._calculate_prediction_accuracy # specify the type of error function for measuring the model's perfomance

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
        self.val_mean_error = []
        self.val_std_error = []

        # create instances of class for the forward and backward propagation object
        self.bnn_fp = bnn_forward_propagation()
        self.bnn_pbp = bnn_probabilistic_back_propagation(self.model_purpose)
    
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
        self.feature_data_scaled = self.feature_data
        self.target_data_scaled = self.target_data.reshape(-1, 1, 1)

        return

    def generate_validation_training_dataset(self):
        val_index = np.floor(len(self.feature_data) * (1 - self.validation_percentage)).astype(int)
        self.validation_feature_data = self.feature_data[val_index:]
        self.feature_data = self.feature_data[:val_index]

        self.validation_feature_data_scaled = self.feature_data_scaled[val_index:]
        self.feature_data_scaled = self.feature_data_scaled[:val_index]

        self.validation_target_data = self.target_data[val_index:]
        self.target_data = self.target_data[:val_index]

        self.validation_target_data_scaled = self.target_data_scaled[val_index:]
        self.target_data_scaled = self.target_data_scaled[:val_index]

        return

    def _generate_m_random_initialization(self, n_origin_neurons, n_destination_neurons):
        """
        create an initial weight's mean for all neurons in a layer

        Args:
        n_origin_neurons (integer) - number of neurons in the previous layer
        n_destination_neurons (integer) - number of neurons in the current layer
        """
        return np.random.uniform(-1, 1, size=(n_destination_neurons, n_origin_neurons))

    def _generate_v_random_initialization(self, n_origin_neurons, n_destination_neurons):
        """
        create an initial weight's variance for all neurons in a layer
        since variance are always non-positive, it is necessary to apply an absolute function to ensure that
        
        Args:
        n_origin_neurons (integer) - number of neurons in the previous layer
        n_destination_neurons (integer) - number of neurons in the current layer
        """
        return np.abs(np.random.random(size=(n_destination_neurons, n_origin_neurons)))

    def _generate_m_normal_xavier_initialization(self, n_origin_neurons, n_destination_neurons):
        """
        create an initial weight's mean for all neurons in a layer

        Args:
        n_origin_neurons (integer) - number of neurons in the previous layer
        n_destination_neurons (integer) - number of neurons in the current layer
        """

        return np.zeros((n_destination_neurons, n_origin_neurons))

    def _generate_v_normal_xavier_initialization(self, n_origin_neurons, n_destination_neurons):
        """
        create an initial weight's variance for all neurons in a layer
        since variance are always non-positive, it is necessary to apply an absolute function to ensure that
        
        Args:
        n_origin_neurons (integer) - number of neurons in the previous layer
        n_destination_neurons (integer) - number of neurons in the current layer
        """
        x = (1 / n_origin_neurons) ** 0.5
        
        return np.ones((n_destination_neurons, n_origin_neurons)) * x
        
    def generate_m(self):
        """
        automate the process of generating all initial weight's mean on all layers
        """
        if self.initialization_type == 'random':
            self.m = [self._generate_m_random_initialization(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-1], self.model_structure[1:])]
        else:
            self.m = [self._generate_m_normal_xavier_initialization(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-1], self.model_structure[1:])]

        return

    def generate_v(self):
        """
        automate the process of generating all initial weight's variance on all layers
        """
        if self.initialization_type == 'random':
            self.v = [self._generate_v_random_initialization(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-1], self.model_structure[1:])]
        else:
            self.v = [self._generate_v_normal_xavier_initialization(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-1], self.model_structure[1:])]

        return

    def _calculate_prediction_mse(self, target_data, pred_mean):
        """
        calculate the prediction's mean squared error

        Args:
        terget_data (array of floats) - the actual target data
        pred_mean (array of floats) - the model's prediction on based on the feature data
        """
        return np.mean((np.log(target_data) - pred_mean) ** 2)
    
    def _calculate_prediction_accuracy(self, target_data, pred_mean):
        """
        calculate the prediction's accuracy

        Args:
        terget_data (array of floats) - the actual target data
        pred_mean (array of floats) - the model's prediction on based on the feature data
        """

        return 100 * np.sum(np.abs(target_data - pred_mean) < 1) / len(target_data)

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
        text = f'Epoch: {current_epoch} / {total_epochs} - Learning Rate : {current_lr} - Succesfull Train Percentage : {np.round(self.succesfull_train / total_trained * 100, 2)}% - Time Passed : {time_passed} Second'
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
    
    def _calculating_errors_sequences(self, error_func, feature_data, target_data):
        """
        the ordered sequences to calculate the model's performance

        Args:
        error_func (function) - the function used to train the model
        """
        # make prediction based on the model and calculate the prediction's mean and standard deviation
        pred_mean, pred_std = self.bnn_fp.feed_forward_neural_network(self.m, self.v, feature_data, self.model_structure, model_purpose=self.model_purpose)

        # calculate the error and add it into their respective list 
        mean_error = error_func(target_data, pred_mean)
        std_error = np.mean(pred_std)
        
        return (mean_error, std_error)

    def _exponential_learning_rate_decay(self, total_epochs):
        """
        create an array of learning rates used to train the model
        the learning rates are acquired from decaying the initial learning rate in an exponential fashion
        
        Args:
        total_epochs (integer) - the total number of epochs for training the model
        """
        self.learning_rates = self.initial_lr / ((self.initial_lr / self.end_lr) * np.arange(1, total_epochs + 1) / total_epochs)
        return  

    def _print_current_epoch_training_result(self, total_epochs, current_epoch, current_lr, start_time, text):
        """
        prints the current training result

        Args:
        total_epochs (integer) - the total number of epochs for training the model
        current_epoch (integer) - the current epoch for training the model
        current_lr (float) - the current learning rate used to train the model
        start_time (integer) - the initial time where the current epoch starts
        text (string) - text that will be printed
        """
        time_passed = np.round(time() - start_time, 2) # calculate the time passed from the initial start time
        
        # create texts that will be printed
        text_1 = f'{text} : {current_epoch} / {total_epochs} - Learning Rate : {current_lr} - Succesfull Train Percentage : {self.succesfull_train / len(self.feature_data) * 100}% - Time Passed : {time_passed} Second'
        if self.model_purpose == 'regression':
            text_2 = f'MSE : {self.mean_error[-1]} - Standard Deviation : {self.std_error[-1]}'
        else:
            text_2 = f'Accuracy : {self.mean_error[-1]}% - Standard Deviation : {self.std_error[-1]}'

        # prints all texts
        print(150 * '-')
        print(text_1)
        print(text_2)
        
        return

    def _print_current_epoch_validation_result(self):
        """
        prints the current validation result

        Args:
        total_epochs (integer) - the total number of epochs for training the model
        current_epoch (integer) - the current epoch for training the model
        current_lr (float) - the current learning rate used to train the model
        start_time (integer) - the initial time where the current epoch starts
        text (string) - text that will be printed
        """
        # create texts that will be printed
        if self.model_purpose == 'regression':
            text = f'Validation MSE : {self.val_mean_error[-1]} - Validation Standard Deviation : {self.val_std_error[-1]}'
        else:
            text = f'Validation Accuracy : {self.val_mean_error[-1]}% - Validation Standard Deviation : {self.val_std_error[-1]}'

        # print all texts
        print(text)
        
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
        
        # start the training process
        for epoch, lr in enumerate(self.learning_rates):
            start_time = time() # initialize the time where the training prcoess starts

            self._training_sequences(lr, total_epochs, epoch + 1) # trains the model
            
            # calculate and save the current model's performance based on the training data
            mean_error, std_error = self._calculating_errors_sequences(self.error_func, self.feature_data, self.target_data)
            self.mean_error.append(mean_error)
            self.std_error.append(std_error)

            self._print_current_epoch_training_result(total_epochs, epoch + 1, lr, start_time, 'Epoch') # prints the training process result

            if self.validation_percentage != None:
                # calculate and save the current model's performance based on the validation data
                val_mean_error, val_std_error = self._calculating_errors_sequences(self.error_func, self.validation_feature_data, self.validation_target_data)
                self.val_mean_error.append(val_mean_error)
                self.val_std_error.append(val_std_error)
                
                self._print_current_epoch_validation_result() # prints the training process result
            
            print(150 * '-') # just prints a line
        
        return         
    
    def visualize_model_performance(self):
        """
        visualize the model's performance throughout the training process
        """
        # set the figure for the visualization
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(20, 10)

        # visualize the model's performance based on the calculated prediction's error and standard deviation
        ax1.plot(self.mean_error, color='blue')
        ax2.plot(self.std_error, color='blue')
        
        # sets the figure's title
        if self.model_purpose == 'regression':
            ax1.set_title('MSE Throughout Training')
        else:
            ax1.set_title('Accuracy Throughout Training')
        ax2.set_title('Variance Throughout Training')

        list_of_legends = ['Training']

        # visualize the model's performance based on the calculated prediction's error and standard deviation on validation data
        if self.validation_percentage != None:
            ax1.plot(self.val_mean_error, color='red')
            ax2.plot(self.val_std_error, color='red')
            list_of_legends.append('Validation')

        # add legends to the visualizations
        ax1.legend(list_of_legends)
        ax2.legend(list_of_legends)

        # shows the visualizations
        fig.show()

        return

    def _generate_predictions(self, feature_data):
        """
        create predictions based on the feature data

        Args:
        feature_data (array of floats) - the feature data sed to create predictions
        """
        # make predictions based on feature data used during training process
        predictions_mean, predictions_std = self.bnn_fp.feed_forward_neural_network(self.m, self.v, feature_data, self.model_structure, model_purpose=self.model_purpose)

        # calculate the upper and lower bound of the prediction
        upper_bound = predictions_mean + predictions_std
        lower_bound = predictions_mean - predictions_std

        return (predictions_mean, predictions_std, upper_bound, lower_bound)

    def _visualize_time_series_predictions(self, ax, x_axis, target_data, predictions_mean, upper_bound, lower_bound, colors):
        """
        create line plot of the prediction's mean and confidence interval

        Args:
        ax (figure) - the figure for the visualization
        x_axis (array of ints) - the visualization's x-axis
        target_data (array of floats) - the actual target data
        predictions_mean (array of floats) - the prediction's mean
        upper_bound (array of floats) - the upper bound for the confidence interval
        lower_bound (array of floats) - the upper bound for the confidence interval
        colors (list of strings) - the list of colors for the visualization
        """
        ax.plot(x_axis, np.log(target_data), color=colors[0]) # visualize the actual target data
        ax.plot(x_axis, predictions_mean, color=colors[1]) # visualize the predictions
        ax.plot(x_axis, upper_bound, color=colors[2]) # visualize the prediction's upper bound
        ax.plot(x_axis, lower_bound, color=colors[2]) # visualize the prediction's lower bound
        ax.fill_between(x_axis, upper_bound, lower_bound, color=colors[3], alpha=0.15) # fills a color between the upper and lower bound visualizations
        
        return
    
    def visualize_time_series_predictions(self):
        """
        visualize the actual target data used for training and valiladtion as well as their respective prediction's mean and confidence interval
        """
        # set the figure for the visualization
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 10)

        # generate prediction's mean and confidence interval based in the training data
        predictions_mean, predictions_std, upper_bound, lower_bound = self._generate_predictions(self.feature_data)

        x_axis = np.arange(0, len(self.feature_data)) # sets the x-axis for the visualization
        # visualize the training data alongside the prediction's mean and confidence interval
        self._visualize_time_series_predictions(ax, x_axis, self.target_data, predictions_mean, upper_bound, lower_bound, ['black', 'red', 'blue', 'green'])

        # lists of legends for the visualization
        list_of_legends = ['Data', 'Prediction Mean', 'Upper Bound x% Confidence Interval', 'Lower Bound x% Confidence Interval', 'Confidence Interval']

        if self.validation_percentage != None:
            # generate prediction's mean and confidence interval based in the validation data
            val_predictions_mean, val_predictions_std, val_upper_bound, val_lower_bound = self._generate_predictions(self.validation_feature_data)

            val_x_axis = np.arange(len(self.feature_data), len(self.feature_data) + len(self.validation_feature_data)) # sets the x-axis for the visualization
            # visualize the training data alongside the prediction's mean and confidence interval
            self._visualize_time_series_predictions(ax, val_x_axis, self.validation_target_data, val_predictions_mean, val_upper_bound, val_lower_bound, ['dimgray', 'lightcoral', 'lightskyblue', 'lightgreen'])
            
            # lists of legends for the visualization
            temp_list_of_legends = ['Validation\'s Data', 'Validation\'s Prediction Mean', 'Validation\'s Upper Bound x% Confidence Interval', 'Validation\'s Lower Bound x% Confidence Interval', 'Validation\'s Confidence Interval']
            list_of_legends = np.concatenate([list_of_legends, temp_list_of_legends])
        
        # add legends to the visualizations
        ax.legend(list_of_legends)

        # shows the visualizations
        fig.show()

        return