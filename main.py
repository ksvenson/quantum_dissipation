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


def ham(delta, F, U, N):
    """
    Hamiltonian for a Kerr nonlinear resonator
    :param delta: Resonant-drive detuning
    :param F: Forcing amplitude, divided by sqrt(N)
    :param U: Kerr nonlinearity, divided by N
    :return: Hamiltonian
    """
    return -delta*ad*a + (U/N/2)*ad*ad*a*a + F*np.sqrt(N)*(ad + a)


def scaled_photon_number(delta, F, U, N):
    tlist = [0]
    H = ham(delta, F, U, N)
    rho_ss = steadystate(H, c_ops)
    e_ops = [ad*a]
    return mesolve(H, rho_ss, tlist, c_ops, e_ops).expect[0][0]/N


def evals(delta, F, U, N):
    H = ham(delta, F, U, N)
    liouv = liouvillian(H, c_ops)
    return np.linalg.eigvals(liouv)


def gap(delta, F, U, N):
    return np.sort(-np.real(evals(delta, F, U, N)))[1]


def first_fidelity(delta, F, U, N):
    tlist = [0]
    H = ham(delta, F, U, N)
    rho_ss = steadystate(H, c_ops)


flist = np.linspace(0, 5, 25)

# TODO
# What does N represent in this case?
# Thermodynamic parameter
for N in (1, 3, 10, 20):
    plt.figure(num='scaled_photon_number')
    plt.plot(flist, [scaled_photon_number(plot_delta, gamma*f, plot_U, N) for f in flist], label=rf'$N={N}$')
    plt.ylabel(r'$\langle a^\dag a \rangle/N$')
    plt.xlabel(r'$\tilde{F}/\gamma$')
    plt.legend()

    plt.figure(num='gap')
    plt.plot(flist, [gap(plot_delta, gamma*f, plot_U, N)/gamma for f in flist], label=rf'$N={N}$')
    plt.ylabel(r'$-\text{Re}(\lambda_1/\gamma)$')
    plt.xlabel(r'$\tilde{F}/\gamma$')
    plt.yscale('log')
    plt.legend()

plt.show()
