from warnings import simplefilter
simplefilter('ignore')

import numpy as np
from scipy.stats import norm

class bnn_forward_propagation():    
    def __init__(self):
        return

    def _calculate_ma_i(self, mz_i_1, m_i):
        """
        calculate the ith layer input marginal mean
        Args:
        mz_i_1 (matrices of floats) - the (i-1)th layer output marginal mean
        m_i (matrices of floats) - the ith layer weight's mean
        """
        return (m_i @ mz_i_1) / (len(m_i) ** 0.5)

    def _calculate_va_i(self, mz_i_1, vz_i_1, m_i, v_i):
        """
        calculate the ith layer input marginal variance
        Args:
        mz_i_1 (matrices of floats) - the (i-1)th layer output marginal mean
        vz_i_1 (matrices of floats) - the (i-1)th layer output marginal variance
        m_i (matrices of floats) - the ith layer weight's mean
        v_i (matrices of floats) - the ith layer weight's variance
        """
        return  (((m_i * m_i) @ vz_i_1) \
                    + (v_i @ (mz_i_1 * mz_i_1)) ) \
                        / (len(mz_i_1) ** 0.5)

    def _calculate_alpha(self, ma_i, va_i):
        """
        calculate the ith layer alpha value
        Args:
        ma_i (matrices of floats) - the ith layer input marginal mean
        va_i (matrices of floats) - the ith layer input marginal variance
        """
        return ma_i / (va_i ** 0.5)

    def _calculate_gaussian_cdf(self, ma_i, va_i, minus):
        """
        calculate the ith layer gaussian cumulative distributive function
        Args:
        ma_i (matrices of floats) - the ith layer input marginal mean
        va_i (matrices of floats) - the ith layer input marginal variance
        minus (bool) - ture if the alpha is negative
        """
        if minus:
            alpha = -1 * self._calculate_alpha(ma_i, va_i)
        else:
            alpha = self._calculate_alpha(ma_i, va_i)
    
        return norm.cdf(alpha)

    def _calculate_gaussian_pdf(self, ma_i, va_i):
        """
        calculate the ith layer gaussian probability density function
        Args:
        ma_i (matrices of floats) - the ith layer input marginal mean
        va_i (matrices of floats) - the ith layer input marginal variance
        """
        alpha = self._calculate_alpha(ma_i, va_i)
        
        return norm.pdf(alpha)

    def _calculate_gamma(self, cdf, pdf):
        """
        calculate the ith layer gamma value
        Args:
        cdf (matrices of floats) - the ith layer gaussian cumulative distributive function
        pdf (matrices of floats) - the ith layer gaussian probability density function
        """
        return pdf / cdf
        
    def _calculate_mz_i(self, ma_i, va_i, cdf, gamma):
        """
        calculate the ith layer output mariginal mean
        Args:
        ma_i (matrices of floats) - the ith layer input marginal mean
        va_i (matrices of floats) - the ith layer input marginal variance
        cdf (matrices of floats) - the ith layer gaussian cumulative distributive function
        gamma (matrices of floats) - the ith layer gamma value
        """
        return cdf * (ma_i + ((va_i ** 0.5) * gamma))

    def _calculate_vz_i(self, ma_i, va_i, mz_i, cdf, minus_cdf, gamma, alpha):
        """
        calculate the ith layer output mariginal mean
        Args:
        ma_i (matrices of floats) - the ith layer input marginal mean
        va_i (matrices of floats) - the ith layer input marginal variance
        mz_i (matrices of floats) - the ith layer output marginal mean
        cdf (matrices of floats) - the ith layer gaussian cumulative distributive function
        minus_cdf (matrices of floats) - the ith layer gaussian cumulative distributive function
        gamma (matrices of floats) - the ith layer gamma value
        alpha (matrices of floats) - the ith layer alpha value
        """
        return (mz_i * (ma_i + ((va_i ** 0.5) * gamma)) * minus_cdf) \
                    + (cdf * va_i * (np.ones(len(ma_i)).reshape(-1, 1) - (gamma ** 0.5) - (gamma * alpha)))

    def forward_propagation(self, feature_data_i, m, v, model_structure):
        """
        perform forward propagation to acquire all the necessary variables

        Args:
        feature_data_i (matrices of floats) - the current feature_data
        m (matrices of floats) - the model weight's mean
        v (matrices of floats) - the model weight's variance
        model_structure (matrices of floats) - list containing number of neurons on each layers
        """
        # empty list to store all variables
        ma, va, cdf, minus_cdf, pdf, gamma, alpha = [], [], [], [], [], [], []
        # the output marginal mean for the 0th layer is feature_data
        mz = [feature_data_i]
        # the output marginal variance for the 0th layer is zero matrix
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

    def _activation_function(self, neuron_value):
        return [np.max([[0], neuron_value])]

    def feed_forward_neural_network(self, mean, variance, feature_data_i, model_structure):
        """
        perform feed forward to acquire prediction using the model
        the feature data is transformed using an exponential function and the model output is coverted back using a natural logarithmic function

        Args:
        mean (matrices of floats) - the corresponding mean of the weight
        variance (matrices of floats) - the corresponding variance of the weight
        feature_data_i (matrices of floats) - the current feature data used to make prediction
        """
        predictions_i = [] # create an empty list to contain the prediction made by the model on each simulations
        for _ in range(50): # perform simulations for 100 times
            neuron_values = [feature_data_i] # since the untreated feature data is used, we would have to apply exponential function into the data
            # perform standard feed forward in the neural network
            for i, (mean_i, var_i) in enumerate(zip(mean, variance)):
                weight = np.random.normal(mean_i, var_i) # take sample for the weight from a normal sample using the mean and variance correspond to the weights
                layers = (weight @ neuron_values[-1]) / (model_structure[i] ** 0.5)
                activated_layers = np.array([self._activation_function(l) for l in layers])
                neuron_values.append(activated_layers)
            # the value of the value of the prediction should be positively valued so that the logarithm of it is defined
            if neuron_values[-1][0, 0] > 0:
                predictions_i.append(neuron_values[-1][0, 0])
        
        predictions_i = np.log(predictions_i)
        # calculate the mean and standard deviation of the prediction
        predictions_mean = np.mean(predictions_i)
        predictions_std = np.std(predictions_i)

        return (predictions_mean, predictions_std)