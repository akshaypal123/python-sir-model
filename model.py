import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N
N = 1000
# Initial infected (I0) and removed (R0)
I0 = 5
R0 = 0
# Initial susceptible (S0)
S0 = N - I0 - R0
# Transmission rate (beta)
beta = 0.2
# Recovery rate
gamma = 1./10
# Time points
t = np.linspace(0, 160, 160)

def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dS_dt = -beta*S*I/N
    dI_dt = beta*S*I/N - gamma*I
    dR_dt = gamma*I
    return dS_dt, dI_dt, dR_dt

y0 = S0, I0, R0
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.set_ylim(0,1.2)
ax.set_xlim(xmin=0.0)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
