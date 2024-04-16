from warnings import simplefilter
simplefilter('ignore')

import numpy as np
from scipy.stats import norm

class bnn_forward_propagation():    
    def __init__(self):
        return

    def _calculate_ma_i(self, mz_i_1, m_i):
        return (m_i @ mz_i_1) / (len(m_i) ** 0.5)

    def _calculate_va_i(self, mz_i_1, vz_i_1, m_i, v_i):
        return  (((m_i * m_i) @ vz_i_1) \
                    + (v_i @ (mz_i_1 * mz_i_1)) ) \
                        / (len(mz_i_1) ** 0.5)

    def _calculate_alpha(self, ma_i, va_i):
        return ma_i / (va_i ** 0.5)

    def _calculate_gaussian_cdf(self, ma_i, va_i, minus):
        if minus:
            alpha = -1 * self._calculate_alpha(ma_i, va_i)
        else:
            alpha = self._calculate_alpha(ma_i, va_i)
    
        return norm.cdf(alpha)

    def _calculate_gaussian_pdf(self, ma_i, va_i):
        alpha = self._calculate_alpha(ma_i, va_i)
        
        return norm.pdf(alpha)

    def _calculate_gamma(self, cdf, pdf):
        return pdf / cdf
        
    def _calculate_mz_i(self, ma_i, va_i, cdf, gamma):
        return cdf * (ma_i + ((va_i ** 0.5) * gamma))

    def _calculate_vz_i(self, ma_i, va_i, mz_i, cdf, minus_cdf, gamma, alpha):
        return (mz_i * (ma_i + ((va_i ** 0.5) * gamma)) * minus_cdf) \
                    + (cdf * va_i * (np.ones(len(ma_i)).reshape(-1, 1) - (gamma ** 0.5) - (gamma * alpha)))

    def forward_propagation(self, feature_data_i, m, v, model_structure):
        ma = []
        va = []
        cdf = []
        minus_cdf = []
        pdf = []
        gamma = []
        alpha = []
        mz = [feature_data_i / feature_data_i]
        vz = [np.array([[0]])]
        
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
    
    def predict(self, feature_data, mz, vz):
        return np.mean([np.random.normal(mz[-1][0, 0], vz[-1][0, 0] ** 0.5) for _ in range(250)]) * feature_data 