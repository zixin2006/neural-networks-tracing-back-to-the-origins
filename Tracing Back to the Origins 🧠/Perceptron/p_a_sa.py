# How test result_1 of Perceptron is derived

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# P(e,i)
def P_e_i(x, y, R, e, i):
    return comb(x, e) * R**e * (1 - R)**(x - e) * comb(y, i) * R**i * (1 - R)**(y - i)

# P_a
def P_a(x, y, R, theta):
    P_a_val = 0
    for e in range(theta, x + 1):
        for i in range(0, min(y, e - theta) + 1):
            P_a_val += P_e_i(x, y, R, e, i)
    return P_a_val


R_values = np.linspace(0, 0.5, 100)
theta_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Figure 1
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

x_y_combinations = [(10, 0), (7, 3), (5, 5), (3, 7), (1, 9)]
for x, y in x_y_combinations:
    P_a_vals = [P_a(x, y, R, 1) for R in R_values]
    axs[0].plot(R_values, P_a_vals, label=f'x={x}, y={y}')
axs[0].set_title('(a) Effect of Inhibitory-Excitatory Mixture, θ = 1')
axs[0].set_xlabel('Proportion of S-points illuminated (R)')
axs[0].set_ylabel('P_a')
axs[0].legend()

# Figure 2
x = 10
y = 0
for theta in theta_values:
    P_a_vals = [P_a(x, y, R, theta) for R in R_values]
    axs[1].plot(R_values, P_a_vals, label=f'θ={theta}')
axs[1].set_title('(b) Variation with θ for x=10, y=0')
axs[1].set_xlabel('Proportion of S-points illuminated (R)')
axs[1].set_ylabel('P_a')
axs[1].legend()

# Figure 3
combinations = [(50, 50, 1), (24, 25, 1), (50, 50, 2), (5, 5, 2), (50, 50, 4), (5, 5, 4)]
for x, y, theta in combinations:
    P_a_vals = [P_a(x, y, R, theta) for R in R_values]
    axs[2].plot(R_values, P_a_vals, label=f'x={x}, y={y}, θ={theta}')
axs[2].set_title('(c) Variation with x, y and θ')
axs[2].set_xlabel('Proportion of S-points illuminated (R)')
axs[2].set_ylabel('P_a')
axs[2].legend()

plt.tight_layout()
plt.show()