import numpy as np
from scipy.stats import laplace, poisson, binom 
import matplotlib.pyplot as plt

class DAME :
    def __init__(self,alpha,n, distrib):

    def random_response(v, alpha):
        p = np.exp(alpha / 6) / (1 + np.exp(alpha / 6))
        return np.where(np.random.rand(*v.shape) < p, v, 1 - v)

    def projection(x, lower, upper):
        return np.clip(x, lower, upper)

    def phi(a, alpha, n):
        c5 = 868.5
        c4 = 8
        return c5 / (n * alpha**2 * np.log(c4 * max(a * n * alpha**2, 1) / np.log(c4 * max(a * n * alpha**2, 1))))

    def compute_m_tilde(n, alpha, distrib, max_a=100):
        M = distrib  # Exemple : M suit une loi de Poisson avec paramètre n
        best_a = 1
        best_value = 0
        
        for a in range(1, max_a + 1):
            probability = (M.sf(a - 1)) ** 2  # P(M >= a)^2
            threshold = min(phi(a, alpha, n), 1)
            
            if probability >= threshold:
                best_a = a
                best_value = probability
        
        return best_a


    def dame_algorithm(n, alpha, distrib, m_dist, data_samples):
        m_tilde = compute_m_tilde(n, alpha, distrib, max_a=100)
        tau = np.sqrt(2 * np.log(8 * max(np.sqrt(m_tilde * n * alpha**2), 1)) / m_tilde)
        bins = np.arange(-1, 1 + 2 * tau, 2 * tau)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        votes = np.zeros(len(bin_centers))
        for u in range(n // 2):
            m_u = len(data_samples[u])
            if m_u >= m_tilde:
                mean_u = np.mean(data_samples[u])
                idx = np.digitize(mean_u, bins) - 1
                votes[max(0, idx - 1):min(len(bin_centers), idx + 2)] += 1
        
        private_votes = random_response(votes, alpha)
        j_hat = np.argmax(private_votes)
        s_j_hat = bin_centers[j_hat]
        L_j_hat, U_j_hat = max(-1, bins[j_hat] - 6 * tau), min(1, bins[j_hat + 1] + 6 * tau)
        
        estimates = []
        for u in range(n // 2, n):
            m_u = len(data_samples[u])
            mean_u = np.mean(data_samples[u])
            shrinked_mean = (np.sqrt(min(m_u, m_tilde)) / np.sqrt(m_tilde)) * (mean_u + ((np.sqrt(m_tilde) / np.sqrt(m_u)) - 1) * s_j_hat)
            projected_mean = projection(shrinked_mean, L_j_hat, U_j_hat)
            noisy_mean = projected_mean + (14 * tau / alpha) * laplace.rvs()
            estimates.append(noisy_mean)
        
        theta_hat = np.mean(estimates)
        return theta_hat


