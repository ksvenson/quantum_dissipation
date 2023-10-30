from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

dim = 30
w0 = 1  # resonant frequency
gamma = 0.1  # dampening rate
a = destroy(dim)
ad = a.dag()
c_ops = [np.sqrt(gamma/2) * a]

plot_delta = 10*gamma
plot_U = 10*gamma

# TODO
# What does N represent in this case?
# Thermodynamic parameter
N = 10


def ham(delta, F, U):
    """
    Hamiltonian for a Kerr nonlinear resonator
    :param delta: Resonant-drive detuning
    :param F: Forcing amplitude
    :param U: Kerr nonlinearity
    :return: Hamiltonian
    """
    return -delta*ad*a + (U/N/2)*ad*ad*a*a + F/np.sqrt(N)*(ad + a)


def scaled_photon_number(delta, F, U):
    tlist = [0]
    H = ham(delta, F, U)
    rho_ss = steadystate(H, c_ops)
    e_ops = [ad*a]
    return mesolve(H, rho_ss, tlist, c_ops, e_ops).expect[0][0]/N


flist = np.linspace(0, 5, 100)
# Photon number plot
plt.figure()
plt.plot(flist, [scaled_photon_number(plot_delta, gamma*f, plot_U) for f in flist])
plt.ylabel(r'$\langle a^\dag a \rangle/N$')
plt.xlabel(r'$\tilde{F}/\gamma$')

# Gap plot
# plt.figure()

plt.show()
