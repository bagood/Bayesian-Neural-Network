from warnings import simplefilter
simplefilter('ignore')

import numpy as np
from scipy.stats import norm

class bnn_forward_propagation():    
    def __init__(self):
        return

    def _calculate_ma_i(self, mz_i_1, m_i):
        """
        calculate the ith layer's input marginal means

        Args:
        mz_i_1 (matrices of floats) - the (i-1)th layer output marginal mean
        m_i (matrices of floats) - the ith layer weight's mean
        """
        return (m_i @ mz_i_1) / (len(m_i) ** 0.5)

    def _calculate_va_i(self, mz_i_1, vz_i_1, m_i, v_i):
        """
        calculate the ith layer's input marginal variances

        Args:
        mz_i_1 (matrices of floats) - the (i-1)th layer's output marginal mean
        vz_i_1 (matrices of floats) - the (i-1)th layer's output marginal variance
        m_i (matrices of floats) - the ith layer weight's mean
        v_i (matrices of floats) - the ith layer weight's variance
        """
        return  (((m_i * m_i) @ vz_i_1) \
                    + (v_i @ (mz_i_1 * mz_i_1)) ) \
                        / (len(mz_i_1) ** 0.5)

    def _calculate_alpha(self, ma_i, va_i):
        """
        calculate the ith layer's alpha values

        Args:
        ma_i (matrices of floats) - the ith layer's input marginal means
        va_i (matrices of floats) - the ith layer's input marginal variances
        """
        return ma_i / (va_i ** 0.5)

    def _calculate_gaussian_cdf(self, ma_i, va_i, minus_bool):
        """
        calculate the ith layer's gaussian cumulative distributive function values

        Args:
        ma_i (matrices of floats) - the ith layer's input marginal mean
        va_i (matrices of floats) - the ith layer's input marginal variance
        minus_bool (bool) - sets the alpha value to negative if the value is True, and the other way around
        """
        if minus_bool:
            alpha = -1 * self._calculate_alpha(ma_i, va_i)
        else:
            alpha = self._calculate_alpha(ma_i, va_i)
    
        return norm.cdf(alpha)

    def _calculate_gaussian_pdf(self, ma_i, va_i):
        """
        calculate the ith layer's gaussian probability density function values

        Args:
        ma_i (matrices of floats) - the ith layer's input marginal means
        va_i (matrices of floats) - the ith layer's input marginal variances
        """
        alpha = self._calculate_alpha(ma_i, va_i)
        
        return norm.pdf(alpha)

    def _calculate_gamma(self, cdf, pdf):
        """
        calculate the ith layer's gamma values

        Args:
        cdf (matrices of floats) - the ith layer's gaussian cumulative distributive function values
        pdf (matrices of floats) - the ith layer's gaussian probability density function values
        """
        return pdf / cdf
        
    def _calculate_mz_i(self, ma_i, va_i, cdf, gamma):
        """
        calculate the ith layer's output mariginal means

        Args:
        ma_i (matrices of floats) - the ith layer's input marginal means
        va_i (matrices of floats) - the ith layer's input marginal variances
        cdf (matrices of floats) - the ith layer's gaussian cumulative distributive function values
        gamma (matrices of floats) - the ith layer's gamma values
        """
        return cdf * (ma_i + ((va_i ** 0.5) * gamma))

    def _calculate_vz_i(self, ma_i, va_i, mz_i, cdf, minus_cdf, gamma, alpha):
        """
        calculate the ith layer's output mariginal means

        Args:
        ma_i (matrices of floats) - the ith layer's input marginal means
        va_i (matrices of floats) - the ith layer's input marginal variances
        mz_i (matrices of floats) - the ith layer's output marginal means
        cdf (matrices of floats) - the ith layer's gaussian cumulative distributive function values
        minus_cdf (matrices of floats) - the ith layer's gaussian cumulative distributive function values
        gamma (matrices of floats) - the ith layer's gamma values
        alpha (matrices of floats) - the ith layer's alpha values
        """
        return (mz_i * (ma_i + ((va_i ** 0.5) * gamma)) * minus_cdf) \
                    + (cdf * va_i * (np.ones(len(ma_i)).reshape(-1, 1) - (gamma ** 0.5) - (gamma * alpha)))

    def forward_propagation(self, feature_data_i, m, v, model_structure):
        """
        perform forward propagation to acquire all variables

        Args:
        feature_data_i (matrices of floats) - the current feature data
        m (matrices of floats) - the model weight's means
        v (matrices of floats) - the model weight's variances
        model_structure (matrices of floats) - list of the number of neurons on each layers
        """
        # create empty lists to store all variables
        ma, va, cdf, minus_cdf, pdf, gamma, alpha = [], [], [], [], [], [], []
        # the output marginal mean for the 0th layer is the feature data
        mz = [feature_data_i]
        # the output marginal variance for the 0th layer is a zero matrix
        vz = [np.zeros((model_structure[0], 1))]
        
        # iterate over each layers in the model
        for i in range(len(model_structure)-1):
            ma.append(self._calculate_ma_i(mz[i], m[i]))
            va.append(self._calculate_va_i(mz[i], vz[i], m[i], v[i]))
            cdf.append(self._calculate_gaussian_cdf(ma[i], va[i], False))
            minus_cdf.append(self._calculate_gaussian_cdf(ma[i], va[i], True))
            pdf.append(self._calculate_gaussian_pdf(ma[i], va[i]))
            gamma.append(self._calculate_gamma(cdf[i], pdf[i]))
            alpha.append(self._calculate_alpha(ma[i], va[i]))
            mz.append(self._calculate_mz_i(ma[i], va[i], cdf[i], gamma[i]))
            vz.append(self._calculate_vz_i(ma[i], va[i], mz[i+1], cdf[i], minus_cdf[i], gamma[i], alpha[i]))
                
        return (ma, va, cdf, minus_cdf, pdf, gamma, alpha, mz, vz)

    def _relu_activation_function(self, layers):
        """
        transform all neuron's values on a layer using the ReLu activation function

        Args:
        layers (vector of floats) - values on each neuron in a layer
        """
        original_shape = layers.shape

        return np.max(np.concatenate((layers.reshape(-1, 1), np.zeros(layers.reshape(-1, 1).shape)), axis=1), axis=1).reshape(original_shape)
    
    def _sigmoid_activation_function(self, predictions):
        """
        transform the neuron's values on a layer using the sigmoid function
        
        Args:
        predicitons (matrices of floats) - the predictions resulted from the model
        """

        return 1 / (1 + np.exp(-1 * predictions))
    
    def _feed_forward_neural_network(self, mean, variance, feature_data, model_structure):
        """
        the feed forward process to acquire prediction based on the feature data

        Args:
        mean (array of matrices of floats) - the weight's mean on all layers in the model
        variance (array of matrices of floats) - the weight's variance on all layers in the model
        """
        latest_neuron_values = feature_data # sets the input layer as the feature data
        
        # perform standard feed forward in the neural network
        for i, (mean_i, var_i) in enumerate(zip(mean, variance)):
            weight = np.random.normal(mean_i, var_i) # take sample from a normal distribution with the mean and variance are the corresponding weight's mean and variance
            layers = (weight @ latest_neuron_values) / (model_structure[i] ** 0.5)
            activated_layers = self._relu_activation_function(layers) # activate the neuron values in the current layer
            latest_neuron_values = activated_layers
        
        return latest_neuron_values.reshape(1, -1)[0]

    def feed_forward_neural_network(self, mean, variance, feature_data, model_structure, transorm_pred_func='log'):
        """
        perform feed forward to acquire the predictions
        
        Args:
        mean (array of matrices of floats) - the weight's mean on all layers in the model
        variance (array of matrices of floats) - the weight's variance on all layers in the model
        feature_data (matrices of floats) - the feature datas used to create the predictions
        """
        # create predictions based on the feature data for 100 times
        predictions = np.array([self._feed_forward_neural_network(mean, variance, feature_data, model_structure) for _ in range(100)])

        # for a time-series data, the predictions must be applied with a loogarithmic funtion because the target data used to train the model are applied with an exponential function
        # for classification purposes, the predictions must be applied with a sigmoid functiom
        if transorm_pred_func == 'log':
            # there are chances for the prediction's value to be zero, we will remove all prediction's with the value of zero since it will resulted in -inf if applied with a logarithmic function
            index_equal_to_zero = (predictions == 0).reshape(1, -1)[0]
            predictions.reshape(1, -1)[0][index_equal_to_zero] = np.nan
            predictions = np.log(predictions)
        else:
            predictions = np.array([self._sigmoid_activation_function(pred) for pred in predictions])

        # calculate the prediction's mean and standard deviation while ignoring any missing values
        predictions_mean = np.nanmean(predictions, axis=0)
        predictions_std = np.nanstd(predictions, axis=0)

        return (predictions_mean, predictions_std)