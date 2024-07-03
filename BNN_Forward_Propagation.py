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
        """
        
        return (m_i @ mz_i_1) / (len(m_i) ** 0.5)

    def _calculate_va_i(self, mz_i_1, vz_i_1, m_i, v_i):
        """
        calculate the ith layer's input marginal variances
        """

        return ((v_i @ (mz_i_1 ** 2)) \
                    + ((m_i ** 2) @ vz_i_1) \
                        + (v_i @ vz_i_1)) \
                            / len(mz_i_1)

    def _calculate_alpha(self, ma_i, va_i):
        """
        calculate the ith layer's alpha values
        """        
        return ma_i / (va_i ** 0.5) 

    def _calculate_gaussian_cdf(self, ma_i, va_i, minus_bool):
        """
        calculate the ith layer's gaussian cumulative distributive function values
        """
        if minus_bool:
            alpha = -1 * self._calculate_alpha(ma_i, va_i)
        else:
            alpha = self._calculate_alpha(ma_i, va_i)

        return norm.cdf(alpha)

    def _calculate_gaussian_pdf(self, ma_i, va_i):
        """
        calculate the ith layer's gaussian probability density function values
        """
        return norm.pdf(self._calculate_alpha(ma_i, va_i))

    def _calculate_gamma(self, cdf, pdf, alpha):
        """
        calculate the ith layer's gamma values
        """
        return np.array([p / c if a >= -30 else -a - (a ** -1) + (2 * (a ** -3)) for p, c, a in zip(pdf, cdf, alpha)])
        
    def _calculate_mz_i(self, ma_i, va_i, cdf, gamma, hidden_layers):
        """
        calculate the ith layer's output mariginal means
        """
        mz_i = cdf * (ma_i + ((va_i ** 0.5) * gamma))

        if hidden_layers:
            mz_i[-1] = 1
        
        return mz_i

    def _calculate_vz_i(self, ma_i, va_i, mz_i, cdf, minus_cdf, gamma, alpha, hidden_layers):
        """
        calculate the ith layer's output mariginal means
        """
        vz_i = (mz_i * (ma_i + ((va_i ** 0.5) * gamma)) * minus_cdf) \
                    + (cdf * va_i * (1 - (gamma ** 2) - (gamma * alpha)))
        
        if hidden_layers:
            vz_i[-1] = 0

        return vz_i

    def forward_propagation(self, feature_data_i, m, v, model_structure):
        """
        perform forward propagation to acquire all variables

        Args:
        feature_data_i (matrices of floats) - the current feature data
        target_data_i (matrices of floats) - the current target data
        m (matrices of floats) - the model weight's means
        v (matrices of floats) - the model weight's variances
        model_structure (matrices of floats) - list of the number of neurons on each layers
        """
        # create empty lists to store all variables
        ma, va, cdf, minus_cdf, pdf, gamma, alpha = [], [], [], [], [], [], []
        
        # the output marginal mean for the 0th layer is the feature data
        mz = [np.concatenate((feature_data_i, [[1]])).reshape(-1, 1)]
        
        # the output marginal variance for the 0th layer is a zero matrix
        vz = [np.zeros((model_structure[0], 1))]
            
        hidden_layers = True # initiate the variabel to be true

        # iterate over each layers in the model
        for i in range(len(model_structure)-1):
            if i == len(model_structure) - 2:
                hidden_layers = False
            ma.append(self._calculate_ma_i(mz[i], m[i]))
            va.append(self._calculate_va_i(mz[i], vz[i], m[i], v[i]))
            cdf.append(self._calculate_gaussian_cdf(ma[i], va[i], False))
            minus_cdf.append(self._calculate_gaussian_cdf(ma[i], va[i], True))
            pdf.append(self._calculate_gaussian_pdf(ma[i], va[i]))
            alpha.append(self._calculate_alpha(ma[i], va[i]))
            gamma.append(self._calculate_gamma(cdf[i], pdf[i], alpha[i]))
            
            if i < len(model_structure) - 2:
                mz.append(self._calculate_mz_i(ma[i], va[i], cdf[i], gamma[i], hidden_layers))
                vz.append(self._calculate_vz_i(ma[i], va[i], mz[i+1], cdf[i], minus_cdf[i], gamma[i], alpha[i], hidden_layers))
            else:
                mz.append(np.log(ma[-1]))
                vz.append(va[-1])
                
        return (ma, va, cdf, minus_cdf, pdf, gamma, alpha, mz, vz)
        
    def _binary_classification_output_activation_function(self, predictions):
        """
        transform the neuron's values on a the output layer using the probit function
        for all data that is labeled as 0 will be replaced with -1

        1 is the label for a fraudulent data
        0 is the label for a non-fraudulent data
        
        Args:
        predicitons (matrices of floats) - the predictions resulted from the model
        """
        fraud = (norm.cdf(-1 * predictions) <= norm.cdf(predictions)).astype(int) # based on the prediction, calculate which label have the higher probability 
        fraud[fraud == 0] = -1

        return fraud