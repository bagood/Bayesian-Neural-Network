from BNN_Forward_Propagation import bnn_forward_propagation
from BNN_Probabilistic_Back_Propagation import bnn_probabilistic_back_propagation

from warnings import simplefilter
simplefilter('ignore')

import math
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class bayesian_neural_network():
    def __init__(self, input_layer, 
                        hidden_layers, 
                        output_layer, 
                        feature_data, 
                        target_data, 
                        batch_size=10,
                        learning_rate=None, 
                        initial_lr=None, 
                        lr_decay_rate=None,
                        total_epochs=100):       

        # initilize all variables required to build a bayesian neural network model
        self.model_structure = np.concatenate((input_layer, hidden_layers, output_layer) ) # create a list containing number of neurons on each layers
        self.model_structure[:-1] += 1 

        self.total_epochs = total_epochs # the number of epochs for training the data
        self.batch_size = batch_size # sets the batch size used to update the weights
    
        self.feature_data = feature_data.reshape(-1, feature_data.shape[-1], 1)
        self.target_data = target_data.reshape(-1, 1, 1)

        # raises an error if learning rate, initial learning rate, and end learning rate present as inputs
        if (learning_rate != None) and ((initial_lr != None) and (lr_decay_rate != None)):
            raise ValueError('There should be only either learning rate or both initial learning rate and learning rate decay')
        
        # determine the type of learning rate used duuring model training
        if learning_rate != None:
            self.learning_rate = learning_rate # the learning rate will be the same for every epochs
            self.learning_rates = np.ones(self.total_epochs) * self.learning_rate
        elif (initial_lr != None) and (lr_decay_rate != None):
            # the learning rate will be decayed over time based on decayed learning rate type
            self.initial_lr = initial_lr # the initial learning rate
            self.lr_decay_rate = lr_decay_rate # the final learning rate
            self._exponential_learning_rate_decay()

        # create global variables that stores the model's performance on each epochs
        self.pred_mean_errors = []
        self.pred_std_error = []

        # initialize model weight's mean and variance
        self._initialize_weight()

        # create instances of class for the forward and backward propagation object
        self.bnn_fp = bnn_forward_propagation()
        self.bnn_pbp = bnn_probabilistic_back_propagation()
                
    def _generate_m_normal_kaiming_initialization(self, n_origin_neurons, n_destination_neurons):
        """
        create an initial weight's mean for all neurons in the layer
        the initialization follows the normal kaiming scheme

        Args:
        n_origin_neurons (integer) - number of neurons in the previous layer
        n_destination_neurons (integer) - number of neurons in the current layer
        """
        x = (1 / n_origin_neurons) ** 0.5

        return np.random.normal(0, x, size=(n_destination_neurons, n_origin_neurons))

    def _generate_v_normal_kaiming_initialization(self, n_origin_neurons, n_destination_neurons):
        """
        create an initial weight's variance for all neurons in the layer
        the initialization follows the normal kaiming scheme
        
        Args:
        n_origin_neurons (integer) - number of neurons in the previous layer
        n_destination_neurons (integer) - number of neurons in the current layer
        """
        x = (1 / n_origin_neurons) ** 0.5
          
        return np.ones((n_destination_neurons, n_origin_neurons))

    def _initialize_weight(self):
        """
        initialize the model weight's mean and variance for all neuron on all layers
        """
        self.m = [self._generate_m_normal_kaiming_initialization(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-2], self.model_structure[1:-1])]
        self.v = [self._generate_v_normal_kaiming_initialization(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-2], self.model_structure[1:-1])]

        self.m.append(np.abs(self._generate_m_normal_kaiming_initialization(self.model_structure[-2], self.model_structure[-1])))
        self.v.append(self._generate_v_normal_kaiming_initialization(self.model_structure[-2], self.model_structure[-1]))
        
        return 
    
    def _shuffles_feature_target_data(self):
        """
        shuffles the feature and target data
        shuffling is done to ensure that each training batches are different from each iterations
        """
        indexes = np.random.permutation(len(self.feature_data)) # create the index for randomizations
        
        # shuffles the data based on the index od randomizations
        self.feature_data = self.feature_data[indexes]
        self.target_data = self.target_data[indexes]

        return

    def _create_empty_batch_optimizer(self):
        """
        create lists of zeros with the shape of the paramaters of the network
        these lists will hold the optimizing gradients for each paramaters
        """

        return [np.zeros(n_origin_neurons, n_destination_neurons) for n_origin_neurons, n_destination_neurons in zip(self.model_structure[:-1], self.model_structure[1:])]

    def _update_batch_optimizer(self, batch_optimizer, optimizer):
        """
        updates each weight's optimizer based on the new optimizing value
        """

        return [batch_opt + opt for batch_opt, opt in zip(batch_optimizer, optimizer)]

    def _generate_batch_weight_optimizer(self, feature_data_i, target_data_i, batch_d_logz_over_m, batch_d_logz_over_v):
        """
        calculate the weight's optimizing values based on the feature data and updates the current batch optimizer based on the acquired weight's optimizer

        Args:
        feature_data_i (array of floats) - the feature data used to calculate the weight optimizing values
        target_data_i (array of floats) - the target data for the current feature data
        batch_d_logz_over_m (array of floats) - the weight's mean optimizing values for the entire batch of data
        batch_d_logz_over_v (array of floats) - the weight's variance optimizing values for the entire batch of data
        """

        # perform the forward propagation to acquire all of variables on all neurons
        forward_propagation_result = self.bnn_fp.forward_propagation(feature_data_i,
                                                                        self.m, 
                                                                        self.v, 
                                                                        self.model_structure)

        # append the feed-forward mean and variance results to be used for calculating errors
        # this is done to help imptove the computational efficiency
        self.pred_mean.append(forward_propagation_result[-2][-1])
        self.pred_std.append(forward_propagation_result[-1][-1])

        # perform backward propagation to acquire the derivatives for optimizing the model's weights
        d_logz_over_m, d_logz_over_v = self.bnn_pbp.calculate_derivatives(self.model_structure, 
                                                                            target_data_i, 
                                                                            self.m, 
                                                                            self.v, 
                                                                            forward_propagation_result)

        # if there's a nan values in the optimizer, the batch optimizer won't be updated
        if (~np.isnan(d_logz_over_m[-1]).any()) and (~np.isnan(d_logz_over_v[-1]).any()):

            # updates the batch optimizer with the newly acquired optimizer 
            batch_d_logz_over_m = self._update_batch_optimizer(batch_d_logz_over_m, d_logz_over_m)
            batch_d_logz_over_v = self._update_batch_optimizer(batch_d_logz_over_v, d_logz_over_v)

            self.succesfull_train += 1 # updates if the gradients are usable
        
        self.total_trained += 1 # updates the number of data trained
        
        return (batch_d_logz_over_m, batch_d_logz_over_v)

    def _optimize_weights(self, batch_d_logz_over_m, batch_d_logz_over_v, learning_rate):
        """
        optimize the network's weights based on the weight's batch optimizer

        Args:
        batch_d_logz_over_m (array of floats) - the weight's mean optimizing values for the entire batch of data
        batch_d_logz_over_v (array of floats) - the weight's variance optimizing values for the entire batch of data
        learning_rate (float) - the size of steps for optimizing the weight's paramaters
        """
        # optimize the model's weights
        self.m = self.bnn_pbp.optimize_m(self.m, 
                                            self.v, 
                                            batch_d_logz_over_m,
                                            learning_rate)
        self.v = self.bnn_pbp.optimize_v(self.m, 
                                            self.v, 
                                            batch_d_logz_over_m, 
                                            batch_d_logz_over_v,
                                            learning_rate)    

        self.v = [np.abs(v) for v in self.v] # making sure that the variances are always non-negative

        return

    def _print_current_epoch_training_process(self, current_epoch, learning_rate, start_time):
        """
        prints the current condition of the training process

        Args:
        current_epoch (integer) - the current epoch for training the model
        learning_rate (float) - the current learning rate used to train the model
        start_time (integer) - the initial time where the current epoch starts
        """
        time_passed = np.round(time() - start_time, 2) # calculate the time passed from the initial start time

        # create the text that will be printed
        text = f'Epoch: {current_epoch} / {self.total_epochs} - Learning Rate : {learning_rate} - Succesfull Train Percentage : {np.round(self.succesfull_train / self.total_trained * 100, 2)}% - Time Passed : {time_passed} Second'
        
        print(text, end='\r')

        return

    def _training_sequences(self, learning_rate, current_epoch):
        """
        the ordered sequences for training the model

        Args:
        learning_rate (float) - the current learning rate used to train the model
        current_epoch (integer) - the current training epochs
        """ 
        start_time = time() # initialize the time where the training prcoess starts
        self.succesfull_train = 0 # create a global value to count the number of succesful train
        self.total_trained = 0 # create a global value to count the number of trained data
        
        # creates a list to store the predictions made from the data
        self.pred_mean = []
        self.pred_std = []

        self._shuffles_feature_target_data() # shuffles the feature and target data
        
        steps = np.ceil(len(self.feature_data) / self.batch_size).astype(int) # based on the number data in the batch, calucaltes the number of batch needed to create

        # iterate over the number of batch needed to create
        for s in range(steps):

            # create empty batch optimizers for the weight's mean and variance
            batch_d_logz_over_m = self._create_empty_batch_optimizer()
            batch_d_logz_over_v = self._create_empty_batch_optimizer()

            # iterate over the feature and target data on the current batch
            for feature_data_i, target_data_i in zip(self.feature_data[self.batch_size * s: self.batch_size * (s + 1)], self.target_data[self.batch_size * s: self.batch_size * (s + 1)]):
                
                # calculate the weight optimizing values based on the current data and update its value into the current batch weight's optimizing values
                batch_d_logz_over_m, batch_d_logz_over_v = self._generate_batch_weight_optimizer(feature_data_i, target_data_i, batch_d_logz_over_m, batch_d_logz_over_v)

            # optimize the network weight's mean and variance based on the current batch weight optimizing values
            self._optimize_weights(batch_d_logz_over_m, batch_d_logz_over_v, learning_rate)
                
            # prints the current training process
            self._print_current_epoch_training_process(current_epoch, learning_rate, start_time)
        
        # determine the mean and variance of the prediction for each feature data
        self.pred_mean = self.bnn_fp._binary_classification_output_activation_function(np.array(self.pred_mean))
        self.pred_std = np.array(self.pred_std)
        
        return 

    def _calculate_prediction_confusion_matrix(self):
        """
        calculate the model's evaluation metrics
        """ 
        # calculate the value of a 2 dimensional confusion matrix
        tp, fp, fn, tn  = confusion_matrix(self.target_data.T[0, 0], self.pred_mean.T[0, 0], labels=[1, -1]).ravel()

        # calculate the evaluation metrics for the model
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        precision = tp / (tp + fp) * 100
        recall = tp / (tp + fn) * 100
        precision = 0 if math.isnan(precision) else precision
        recall = 0 if math.isnan(recall) else recall

        return (accuracy, precision, recall)   

    def _measuring_model_performances_sequences(self):
        """
        the ordered sequences for measuring the model's performance
        """
        # calculate the model performance based on the model's evaluation metrics
        self.pred_mean_errors.append(self._calculate_prediction_confusion_matrix())
        self.pred_std_error.append(np.mean(self.pred_std))
        
        return 

    def _exponential_learning_rate_decay(self):
        """
        create an array of learning rates used to train the model
        the learning rates are acquired from decaying the initial learning rate in an exponential fashion        
        """
        self.learning_rates = [(self.lr_decay_rate ** i) * self.initial_lr for i in range(1, self.total_epochs+1)]
            
        return  

    def _print_current_epoch_training_result(self, current_epoch, learning_rate, start_time):
        """
        prints the current training epoch result based on the training data

        Args:
        current_epoch (integer) - the current training epochs
        learning_rate (float) - the current learning rate used to train the model
        start_time (integer) - the initial time where the current epoch starts
        """
        time_passed = np.round(time() - start_time, 2) # calculate the time passed from the initial start time
        
        # create texts that will be printed
        text_1 = f'Epoch : {current_epoch} / {self.total_epochs} - Learning Rate : {learning_rate} - Succesfull Train Percentage : {(self.succesfull_train / self.total_trained) * 100}% - Time Passed : {time_passed} Second'
        
        accuracy = np.array(self.pred_mean_errors)[:, 0][-1]
        precision = np.array(self.pred_mean_errors)[:, 1][-1]
        recall = np.array(self.pred_mean_errors)[:, 2][-1]
        text_2 = f'Accuracy : {accuracy}% - Precision : {precision}% - Sensitivity : {recall}% - Standard Deviation : {self.pred_std_error[-1]}'

        # print all texts
        print(150 * '-')
        print(text_1)
        print(text_2)
        
        return

    def train_model(self):
        """
        train the model by performing forward propagation to acquire all necessary variables, then performing backward propagation to optimize the model weight's mean and variance
        """        
        # start the training process
        for current_epoch, learning_rate in enumerate(self.learning_rates):
            start_time = time() # initialize the time where the training prcoess starts

            self._training_sequences(learning_rate, current_epoch + 1) # trains the model

            if self.succesfull_train == 0:
                raise ValueError('It appears that there is an exploding gradient during training')
        
            self._measuring_model_performances_sequences() # calculate the model's performance based on the training data

            self._print_current_epoch_training_result(current_epoch + 1, learning_rate, start_time) # prints the training process result
            
            print(150 * '-') # just prints a line
        
        return         
    
    def visualize_model_performance(self):
        """
        visualize the model's performance throughout the training process
        """
        # set the figure for the visualization
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(20, 10)

        # select the model's evaluation metrics and store it in a list
        accuracy = np.array(self.pred_mean_errors)[:, 0]
        precision = np.array(self.pred_mean_errors)[:, 1]
        recall = np.array(self.pred_mean_errors)[:, 2]

        # visualize the model's performance throuughout training
        ax1.plot(accuracy, color='blue')
        ax1.plot(precision, color='green')
        ax1.plot(recall, color='red')
        ax2.plot(self.pred_std_error, color='black')
        
        # set the legend for the visualizations
        ax1.legend(['Accuracy', 'Precision', 'Sensitivity'])
        ax2.legend(['Standard Deviation'])

        # sets the figure's title based on the model's task
        ax1.set_title('Accuracy, Precision, and Sensitivity Throughout Training')
        ax2.set_title('Standard Deviation Throughout Training')

        return