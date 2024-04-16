from warnings import simplefilter
simplefilter('ignore')

import numpy as np

class bnn_probabilistic_back_propagation():
    def __init__(self, model_structure, target_data, learning_rate):
        self.model_structure = model_structure
        self.target_data = target_data
        self.learning_rate = learning_rate

    def _derivative_ma_i_over_mz_i_1(self, m_i):
        return m_i / (len(m_i) ** 0.5) 

    def _derivative_ma_i_over_vz_i_1(self):
        return 0

    def _derivative_va_i_over_mz_i_1(self, mz_i_1, v_i):
        return (2 * mz_i_1 * v_i) / len(v_i)

    def _derivative_va_i_over_vz_i_1(self, m_i, v_i):
        return ((m_i ** 2) + v_i) / len(v_i)

    def _extend_to_neuron_shape(self, array, target_shape):
        return np.array([array[0] for i in range(target_shape.shape[0])])

    def _derivative_ma_i_over_m_i(self, mz_i_1, m_i):
        d_ma_i_over_m_i = mz_i_1 / (len(mz_i_1)) ** 0.5

        return self._extend_to_neuron_shape(d_ma_i_over_m_i.T, m_i)

    def _derivative_ma_i_over_v_i(self):
        return 0

    def _derivative_va_i_over_m_i(self, vz_i_1, m_i):
        return (2 * self._extend_to_neuron_shape(vz_i_1.T, m_i) * m_i) / len(vz_i_1)

    def _derivative_va_i_over_v_i(self, mz_i_1, vz_i_1, m_i):
        d_va_i_over_v_i = ((mz_i_1 ** 2) + vz_i_1) / len(mz_i_1)

        return self._extend_to_neuron_shape(d_va_i_over_v_i.T, m_i)    
    
    def _derivative_alpha_i_over_ma_i(self, va_i):
        return 1 / (va_i ** 0.5)

    def _derivative_alpha_i_over_va_i(self, ma_i, va_i):
        return ma_i / (2 * (va_i ** 1.5))

    def _derivative_gamma_i_over_ma_i(self, gamma_i, alpha_i, ma_i, d_alpha_i_over_ma_i):
        return -1 * ((gamma_i * alpha_i) + (gamma_i ** 2)) * d_alpha_i_over_ma_i

    def _derivative_gamma_i_over_va_i(self, gamma_i, alpha_i, ma_i, d_alpha_i_over_va_i):
        return -1 * ((gamma_i * alpha_i) + (gamma_i ** 2)) * d_alpha_i_over_va_i
    
    def _derivative_mz_i_over_ma_i(self, ma_i, va_i, cdf_i, pdf_i, gamma_i, d_alpha_i_over_ma_i, d_gamma_i_over_ma_i):
        return (d_alpha_i_over_ma_i * pdf_i * (ma_i + ((va_i ** 0.5) * gamma_i))) \
                    + (cdf_i * (1 + ((va_i ** 0.5) * d_gamma_i_over_ma_i)))

    def _derivative_mz_i_over_va_i(self, ma_i, va_i, cdf_i, pdf_i, gamma_i, d_alpha_i_over_va_i, d_gamma_i_over_va_i):
        return (d_alpha_i_over_va_i * pdf_i * (ma_i + ((va_i ** 0.5) * gamma_i))) \
                    + (cdf_i * ((gamma_i / (2 * (va_i ** 0.5))) + ((va_i ** 0.5) * d_gamma_i_over_va_i)))

    def _derivative_vz_i_over_ma_i(self, ma_i, va_i, mz_i, pdf_i, cdf_i, minus_cdf_i, gamma_i, alpha_i, d_mz_i_over_ma_i, d_gamma_i_over_ma_i, d_alpha_i_over_ma_i):
        return (d_mz_i_over_ma_i * (ma_i + ((va_i ** 0.5) * gamma_i)) * minus_cdf_i) \
                    + mz_i * ((1 + ((va_i ** 0.5) * d_gamma_i_over_ma_i)) * minus_cdf_i \
                            - ((ma_i + ((va_i ** 0.5) * gamma_i)) * pdf_i * d_alpha_i_over_ma_i)) \
                    + pdf_i * d_alpha_i_over_ma_i * va_i * (1 - gamma_i ** 2 - (gamma_i * alpha_i)) \
                    - cdf_i * va_i * ((2 * gamma_i * d_gamma_i_over_ma_i) + (d_gamma_i_over_ma_i * alpha_i) + (gamma_i * d_alpha_i_over_ma_i))

    def _derivative_vz_i_over_va_i(self, ma_i, va_i, mz_i, pdf_i, cdf_i, minus_cdf_i, gamma_i, alpha_i, d_mz_i_over_va_i, d_gamma_i_over_va_i, d_alpha_i_over_va_i):
        return (d_mz_i_over_va_i * (ma_i + ((va_i ** 0.5) * gamma_i)) * minus_cdf_i) \
                    + mz_i * ((((1 / (2 * (va_i ** 0.5))) * gamma_i) + ((va_i ** 0.5) * d_gamma_i_over_va_i)) * minus_cdf_i \
                            - ((ma_i + ((va_i ** 0.5) * gamma_i)) * pdf_i * d_alpha_i_over_va_i)) \
                    + pdf_i * d_alpha_i_over_va_i * va_i * (1 - (gamma_i ** 2) - (gamma_i * alpha_i)) \
                    + cdf_i * ((1 - (gamma_i ** 2) - (gamma_i * alpha_i)) - (va_i * ((2 * gamma_i * d_gamma_i_over_va_i) + (d_gamma_i_over_va_i * alpha_i) + (gamma_i * d_alpha_i_over_va_i))))

    def _derivative_ma_i_over_ma_i_1(self, m_i_1, ma_i, va_i, mz_i, pdf_i, cdf_i, minus_cdf_i, gamma_i, alpha_i, d_alpha_i_over_ma_i, d_gamma_i_over_ma_i, d_mz_i_over_ma_i):
        return self._derivative_ma_i_over_mz_i_1(m_i_1) \
                        * self._derivative_mz_i_over_ma_i(ma_i, va_i, cdf_i, pdf_i, gamma_i, d_alpha_i_over_ma_i, d_gamma_i_over_ma_i) \
                    + self._derivative_ma_i_over_vz_i_1()  \
                        * self._derivative_vz_i_over_ma_i(ma_i, va_i, mz_i, pdf_i, cdf_i, minus_cdf_i, gamma_i, alpha_i, d_mz_i_over_ma_i, d_gamma_i_over_ma_i, d_alpha_i_over_ma_i)

    def _derivative_ma_i_over_va_i_1(self, m_i_1, ma_i, va_i, mz_i, pdf_i, cdf_i, minus_cdf_i, gamma_i, alpha_i, d_alpha_i_over_va_i, d_gamma_i_over_va_i, d_mz_i_over_va_i):
        return self._derivative_ma_i_over_mz_i_1(m_i_1) \
                        * self._derivative_mz_i_over_va_i(ma_i, va_i, cdf_i, pdf_i, gamma_i, d_alpha_i_over_va_i, d_gamma_i_over_va_i) \
                    + self._derivative_ma_i_over_vz_i_1() \
                        * self._derivative_vz_i_over_va_i(ma_i, va_i, mz_i, pdf_i, cdf_i, minus_cdf_i, gamma_i, alpha_i, d_mz_i_over_va_i, d_gamma_i_over_va_i, d_alpha_i_over_va_i)

    def _derivative_va_i_over_ma_i_1(self, m_i_1, v_i_1, ma_i, va_i, mz_i, pdf_i, cdf_i, minus_cdf_i, gamma_i, alpha_i, d_alpha_i_over_ma_i, d_gamma_i_over_ma_i, d_mz_i_over_ma_i):
        return self._derivative_va_i_over_mz_i_1(mz_i, v_i_1) \
                        * self._derivative_mz_i_over_ma_i(ma_i, va_i, cdf_i, pdf_i, gamma_i, d_alpha_i_over_ma_i, d_gamma_i_over_ma_i) \
                    + self._derivative_va_i_over_vz_i_1(m_i_1, v_i_1) \
                        * self._derivative_vz_i_over_ma_i(ma_i, va_i, mz_i, pdf_i, cdf_i, minus_cdf_i, gamma_i, alpha_i, d_mz_i_over_ma_i, d_gamma_i_over_ma_i, d_alpha_i_over_ma_i)

    def _derivative_va_i_over_va_i_1(self, m_i_1, v_i_1, ma_i, va_i, mz_i, pdf_i, cdf_i, minus_cdf_i, gamma_i, alpha_i, d_alpha_i_over_va_i, d_gamma_i_over_va_i, d_mz_i_over_va_i):
        return self._derivative_va_i_over_mz_i_1(mz_i, v_i_1) \
                        * self._derivative_mz_i_over_va_i(ma_i, va_i, cdf_i, pdf_i, gamma_i, d_alpha_i_over_va_i, d_gamma_i_over_va_i) \
                    + self._derivative_va_i_over_vz_i_1(m_i_1, v_i_1) \
                        * self._derivative_vz_i_over_va_i(ma_i, va_i, mz_i, pdf_i, cdf_i, minus_cdf_i, gamma_i, alpha_i, d_mz_i_over_va_i, d_gamma_i_over_va_i, d_alpha_i_over_va_i)

    def _derivative_logz_over_ma_L(self, ma_L, va_L):
        return (self.target_data[0, 0] - ma_L) / va_L

    def _derivative_logz_over_va_L(self, ma_L, va_L):
        return 0.5 * ((((self.target_data[0, 0] - ma_L) / va_L) ** 2) - (1 / va_L))

    def _derivative_logz_over_ma_i_1(self, d_logz_over_ma_i, d_logz_over_va_i, d_ma_i_over_ma_i_1, d_va_i_over_ma_i_1):
        return (d_ma_i_over_ma_i_1.T @ d_logz_over_ma_i) + (d_va_i_over_ma_i_1.T @ d_logz_over_va_i)

    def _derivative_logz_over_va_i_1(self, d_logz_over_ma_i, d_logz_over_va_i, d_ma_i_over_va_i_1, d_va_i_over_va_i_1):
        return (d_ma_i_over_va_i_1.T @ d_logz_over_ma_i) + (d_va_i_over_va_i_1.T @ d_logz_over_va_i)
    
    def _derivative_logz_over_m_i(self, d_logz_over_ma_i, d_logz_over_va_i, d_ma_i_over_m_i, d_va_i_over_m_i):
        return (d_logz_over_ma_i * d_ma_i_over_m_i) + (d_logz_over_va_i * d_va_i_over_m_i)

    def _derivative_logz_over_v_i(self, d_logz_over_ma_i, d_logz_over_va_i, d_ma_i_over_v_i, d_va_i_over_v_i):
        return (d_logz_over_ma_i * d_ma_i_over_v_i) + (d_logz_over_va_i * d_va_i_over_v_i) 
    
    def calculate_derivatives(self, m, v, ma, va, cdf, minus_cdf, pdf, gamma, alpha, mz, vz):
        n_layers = len(self.model_structure) - 1

        # Gradients of Input Marginal Mean and Variance over Output Marginal Mean and Variance
        d_ma_over_mz = [self._derivative_ma_i_over_mz_i_1(m[i]) for i in range(n_layers)]
        d_ma_over_vz = [self._derivative_ma_i_over_vz_i_1() for i in range(n_layers)]
        d_va_over_mz = [self._derivative_va_i_over_mz_i_1(m[i], v[i]) for i in range(n_layers)]
        d_va_over_vz = [self._derivative_va_i_over_vz_i_1(m[i], v[i]) for i in range(n_layers)]        

        # Gradients of Input Marginal Mean and Variance over Mean and Variance
        d_ma_over_m = [self._derivative_ma_i_over_m_i(mz[i], m[i]) for i in range(n_layers)]
        d_ma_over_v = [self._derivative_ma_i_over_v_i() for i in range(n_layers)]
        d_va_over_m = [self._derivative_va_i_over_m_i(vz[i], m[i]) for i in range(n_layers)]
        d_va_over_v = [self._derivative_va_i_over_v_i(mz[i], vz[i], m[i]) for i in range(n_layers)]

        # Gradients of Alpha and Lambda Value over Input Marginal Mean and Variance
        d_alpha_over_ma = [self._derivative_alpha_i_over_ma_i(va[i]) for i in range(n_layers)]
        d_alpha_over_va = [self._derivative_alpha_i_over_va_i(ma[i], va[i]) for i in range(n_layers)]
        d_gamma_over_ma = [self._derivative_gamma_i_over_ma_i(gamma[i], alpha[i], ma[i], d_alpha_over_ma[i]) for i in range(n_layers)]
        d_gamma_over_va = [self._derivative_gamma_i_over_va_i(gamma[i], alpha[i], ma[i], d_alpha_over_va[i]) for i in range(n_layers)]

        # Gradients of Output Marginals Over Input Marginals
        d_mz_over_ma = [self._derivative_mz_i_over_ma_i(ma[i], va[i], cdf[i], pdf[i], gamma[i], d_alpha_over_ma[i], d_gamma_over_ma[i]) for i in range(n_layers)]
        d_mz_over_va = [self._derivative_mz_i_over_va_i(ma[i], va[i], cdf[i], pdf[i], gamma[i], d_alpha_over_va[i], d_gamma_over_va[i]) for i in range(n_layers)]
        d_vz_over_ma = [self._derivative_vz_i_over_ma_i(ma[i], va[i], mz[i+1], pdf[i], cdf[i], minus_cdf[i], gamma[i], alpha[i], d_mz_over_ma[i], d_gamma_over_va[i], d_alpha_over_ma[i]) for i in range(n_layers)]
        d_vz_over_va = [self._derivative_vz_i_over_va_i(ma[i], va[i], mz[i+1], pdf[i], cdf[i], minus_cdf[i], gamma[i], alpha[i], d_mz_over_va[i], d_gamma_over_va[i], d_alpha_over_va[i]) for i in range(n_layers)]

        # Gradients of ith Layer Marginal Inputs over (i-1)th Layer Marginal Inputs
        d_ma_over_ma_1 = [self._derivative_ma_i_over_ma_i_1(m[i], ma[i], va[i], mz[i+1], pdf[i], cdf[i], minus_cdf[i], gamma[i], alpha[i], d_alpha_over_ma[i], d_gamma_over_ma[i], d_mz_over_ma[i]) for i in range(n_layers)] 
        d_ma_over_va_1 = [self._derivative_ma_i_over_va_i_1(m[i], ma[i], va[i], mz[i+1], pdf[i], cdf[i], minus_cdf[i], gamma[i], alpha[i], d_alpha_over_va[i], d_gamma_over_va[i], d_mz_over_va[i]) for i in range(n_layers)]
        d_va_over_ma_1 = [self._derivative_va_i_over_ma_i_1(m[i], v[i], ma[i], va[i], mz[i+1], pdf[i], cdf[i], minus_cdf[i], gamma[i], alpha[i], d_alpha_over_ma[i], d_gamma_over_ma[i], d_mz_over_ma[i]) for i in range(n_layers)]
        d_va_over_va_1 = [self._derivative_va_i_over_va_i_1(m[i], v[i], ma[i], va[i], mz[i+1], pdf[i], cdf[i], minus_cdf[i], gamma[i], alpha[i], d_alpha_over_ma[i], d_gamma_over_ma[i], d_mz_over_ma[i]) for i in range(n_layers)]

        # Gradients of log(Z) on Output Layer
        d_logz_over_ma = [self._derivative_logz_over_ma_L(ma[-1], va[-1])]
        d_logz_over_va = [self._derivative_logz_over_va_L(ma[-1], va[-1])]

        # Gradients of log(Z) on Hidden Layers
        for i, (d_ma_i_over_ma_i_1, d_ma_i_over_va_i_1, d_va_i_over_ma_i_1, d_va_i_over_va_i_1) in enumerate(zip(d_ma_over_ma_1[::-1], d_ma_over_va_1[::-1], d_va_over_ma_1[::-1], d_va_over_va_1[::-1])):
            d_logz_over_ma.append(self._derivative_logz_over_ma_i_1(d_logz_over_ma[i], d_logz_over_va[i], d_ma_i_over_ma_i_1, d_va_i_over_ma_i_1))
            d_logz_over_va.append(self._derivative_logz_over_va_i_1(d_logz_over_ma[i], d_logz_over_va[i], d_ma_i_over_va_i_1, d_va_i_over_va_i_1))
        d_logz_over_ma = d_logz_over_ma[:-1][::-1]
        d_logz_over_va = d_logz_over_va[:-1][::-1]

        # Gradients of log(Z) over Weight's Mean and Variance
        d_logz_over_m = [self._derivative_logz_over_m_i(d_logz_over_ma[i], d_logz_over_va[i], d_ma_over_m[i], d_va_over_m[i]) for i in range(n_layers)]
        d_logz_over_v = [self._derivative_logz_over_m_i(d_logz_over_ma[i], d_logz_over_va[i], d_ma_over_v[i], d_va_over_v[i]) for i in range(n_layers)]

        return (d_logz_over_m, d_logz_over_v)
    
    def _linear_learning_rate_scheduler(self, end_lr, total_epochs, current_epoch):
        self.learning_rate -= (((self.learning_rate - end_lr) / total_epochs) * current_epoch)
    
        return 
 
    def optimize_m(self, m, v, d_logz_over_m):
        return [m_ + (self.learning_rate * (v_ * d_m)) for m_, v_, d_m in zip(m, v, d_logz_over_m)]

    def optimize_v(self, m, v, d_logz_over_m, d_logz_over_v):
        return [v_ - (self.learning_rate * ((v_ ** 2) * ((d_m ** 2) - (2 * d_v)))) for m_, v_, d_m, d_v in zip(m, v, d_logz_over_m, d_logz_over_v)]
