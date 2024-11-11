# How test result_2 of Perceptron is derived

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# P(e,i)
def P(e, i, x, y, R):
    return comb(x, e) * (R**e) * ((1-R)**(x-e)) * comb(y, i) * (R**i) * ((1-R)**(y-i))

# P_a
def P_a(theta, x, y, R):
    P_a_value = 0
    for e in range(theta, x+1):
        for i in range(0, min(y, e-theta)+1):
            P_a_value += P(e, i, x, y, R)
    return P_a_value

# P_c
def P_c(theta, x, y, R, L, G):
    P_c_value = 0
    for e in range(theta, x+1):
        for i in range(0, min(y, e-theta)+1):
            for le in range(0, e+1):
                for li in range(0, i+1):
                    for ge in range(0, x-e+1):
                        for gi in range(0, y-i+1):
                            if e - i - le + li + ge - gi >= theta:
                                P_c_value += (P(e, i, x, y, R) *
                                              comb(e, le) * (L**le) * ((1-L)**(e-le)) *
                                              comb(i, li) * (L**li) * ((1-L)**(i-li)) *
                                              comb(x-e, ge) * (G**ge) * ((1-G)**(x-e-ge)) *
                                              comb(y-i, gi) * (G**gi) * ((1-G)**(y-i-gi)))
    return P_c_value

# Figure 1

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
R_values = np.linspace(0, 0.5, 100)
combinations = [(10, 0, 1), (7, 3, 1), (7, 3, 2), (5, 5, 1), (3, 7, 1)]

for x, y, theta in combinations:
    P_c_values = [P_a(theta, x, y, R) for R in R_values]
    axs[0].plot(R_values, P_c_values, label=f'x={x}, y={y}, θ={theta}')
    axs[0].annotate(f'x={x}, y={y}, θ={theta}', xy=(R_values[-1], P_c_values[-1]), xytext=(5, 0), textcoords='offset points', ha='right', fontsize=8)

axs[0].set_xlabel('R (Retinal Area Illuminated)')
axs[0].set_ylabel('P_c')
axs[0].set_title('(a) P_c as a function of R, for nonoverlapping stimuli.')
axs[0].legend()
axs[0].grid(True)

# Figure 2
C_values = np.linspace(0, 1, 100)
theta_values = [3, 5, 7, 8]

for theta in theta_values:
    for R in [0.5, 0.2]:
        P_c_values = [(1 - L) * (1 - G)**theta for L, G in zip(C_values, C_values)]
        axs[1].plot(C_values, P_c_values, label=f'R={R}, θ={theta}, {"solid" if R == 0.5 else "dashed"}', linestyle='-' if R == 0.5 else '--')
        axs[1].annotate(f'θ={theta}', xy=(C_values[-1], P_c_values[-1]), xytext=(5, 0), textcoords='offset points', ha='right', fontsize=8)

axs[1].set_xlabel('C (Proportion of Overlap Between Stimuli)')
axs[1].set_ylabel('P_c')
axs[1].set_title('(b) P_c as a function of C. X = 10, Y = 0.')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()