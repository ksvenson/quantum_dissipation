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
    return -delta*ad*a + (U/N/2)*ad*ad*a*a + F*np.sqrt(N)*(ad + a)


def scaled_photon_number(state):
    return np.trace(ad*a*state)/N


def liouv_gap(op):
    return np.abs(np.real(sp.sparse.linalg.eigs(op, k=1, which='SR', return_eigenvectors=False, tol=0.5)[0]))


def first_fidelity(delta, F, U, N):
    H = ham(delta, F, U, N)
    rho_ss = steadystate(H, c_ops)


if __name__ == '__main__':
    flist = np.linspace(0, 5, 100)

    for N in (1, 3, 10, 20):
        spn = []
        gap = []
        for f in flist:
            H = ham(plot_delta, gamma*f, plot_U, N)
            liouv = liouvillian(H, c_ops)
            rho_ss = steadystate(H, c_ops)
            spn.append(scaled_photon_number(rho_ss))
            gap.append(liouv_gap(liouv.data))
        spn = np.array(spn)
        gap = np.array(gap)

        # plt.figure(num='scaled_photon_number')
        # plt.plot(flist, spn, label=rf'$N={N}$')
        # plt.ylabel(r'$\langle a^\dag a \rangle/N$')
        # plt.xlabel(r'$\tilde{F}/\gamma$')
        # plt.legend()

        plt.figure(num='gap')
        plt.plot(flist, gap/gamma, label=rf'$N={N}$')
        plt.ylabel(r'$-\text{Re}(\lambda_1/\gamma)$')
        plt.xlabel(r'$\tilde{F}/\gamma$')
        plt.yscale('log')
        plt.legend()

    plt.show()
