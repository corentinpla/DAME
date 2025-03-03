import DAME

n = 10000  # Nombre d'utilisateurs
alpha = 0.5  # Paramètre de confidentialité
num_simulations = 100  # Nombre de simulations par valeur de lambda

lambda_values = np.linspace(0.1, 10, 10)  # Valeurs de lambda
risk_means = []
risk_lower = []
risk_upper = []

for lambda_poisson in lambda_values:
    risk_samples = []
    for _ in range(num_simulations):
        m_dist = np.random.poisson(lambda_poisson, n)
        m_dist = np.where(m_dist > 0, m_dist, 1)
        distrib = poisson(lambda_poisson)
        data_samples = [np.random.uniform(-1, 1, m) for m in m_dist]
        theta_hat = dame_algorithm(n, alpha, distrib, m_dist, data_samples)
        risk = np.mean((theta_hat - 0) ** 2)
        risk_samples.append(risk)
    
    mean_risk = np.mean(risk_samples)
    lower_bound = np.percentile(risk_samples, 2.5)
    upper_bound = np.percentile(risk_samples, 97.5)
    
    risk_means.append(mean_risk)
    risk_lower.append(lower_bound)
    risk_upper.append(upper_bound)

plt.figure(figsize=(8, 6))
plt.plot(lambda_values, risk_means, marker='o', linestyle='-', label='Moyenne du risque')
plt.fill_between(lambda_values, risk_lower, risk_upper, color='b', alpha=0.2, label='Intervalle de confiance 95%')
plt.xlabel(r'$\lambda$ (Poisson)')
plt.ylabel('Risque')
plt.title('Risque en fonction de $\lambda$ pour DAME avec intervalles de confiance')
plt.legend()
plt.grid(True)
plt.show()
