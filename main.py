from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

dim = 60
w0 = 1  # resonant frequency
gamma = 0.1  # dampening rate
a = destroy(dim)
ad = a.dag()
c_ops = [np.sqrt(gamma/2) * a]

plot_delta = 10*gamma
plot_U = 10*gamma
flist = np.linspace(0, 5, 100)
Nlist = np.array([1, 3, 10, 20])


def ham(delta, F, U, N):
    return -delta*ad*a + (U/N/2)*ad*ad*a*a + F*np.sqrt(N)*(ad + a)


def scaled_photon_number(state):
    return np.trace(ad*a*state)/N


def liouv_gap(op):
    print('calculating eigenvalues')
    eigs = np.sort(np.abs(np.real(np.linalg.eigvals(op))))[1]
    return eigs
    # evals = sp.sparse.linalg.eigs(sp.linalg.fractional_matrix_power(op, 0.1), k=1, which='LR', tol=10**(-4), return_eigenvectors=False)
    # # eigs = sp.sparse.linalg.eigs(op.data, k=1, sigma=0, return_eigenvectors=False, tol=0.5)
    # print(evals)
    # idx = np.argmin(np.real(evals))
    # return evals[idx]


def get_evals(op):
    return np.linalg.eigvals(op)


def fidelity(s1, s2):
    sqrts1 = sp.linalg.sqrtm(s1)
    return np.trace(sp.linalg.sqrtm(sqrts1 * s2 * sqrts1))


def wigner():
    # rho+
    # rho -
    # rho steady state
    pass


def save_evals():

    return  # don't want to accidentally call this function

    np.save('flist', flist)
    np.save('Nlist', Nlist)

    result = np.zeros((len(Nlist), len(flist), dim**2), dtype=complex)
    for Nidx, N in enumerate(Nlist):
        for fidx, f in enumerate(flist):
            H = ham(plot_delta, gamma * f, plot_U, N)
            liouv = liouvillian(H, c_ops)
            result[Nidx, fidx] = get_evals(liouv)
            np.save(f'evals_plotdelta_{plot_delta}_plotU_{plot_U}', result)
            print(f'Done (N, f) = {(N, f)}')


if __name__ == '__main__':
    all_evals = np.load(f'evals_plotdelta_{plot_delta}_plotU_{plot_U}.npy')
    for Nidx, N in enumerate(Nlist):
        spn = []
        gap = []
        for fidx, f in enumerate(flist):
            H = ham(plot_delta, gamma*f, plot_U, N)
            liouv = liouvillian(H, c_ops)
            rho_ss = steadystate(H, c_ops)
            evals = all_evals[Nidx, fidx]

            spn.append(scaled_photon_number(rho_ss))
            gap.append(np.sort(-np.real(evals))[1])
        spn = np.array(spn)
        gap = np.array(gap)

        plt.figure(num='scaled_photon_number')
        plt.plot(flist, spn, label=rf'$N={N}$')
        plt.ylabel(r'$\langle a^\dag a \rangle/N$')
        plt.xlabel(r'$\tilde{F}/\gamma$')
        plt.legend()

        plt.figure(num='gap')
        plt.plot(flist, gap/gamma, label=rf'$N={N}$')
        plt.ylabel(r'$-\text{Re}(\lambda_1/\gamma)$')
        plt.xlabel(r'$\tilde{F}/\gamma$')
        plt.yscale('log')
        plt.legend()

    plt.show()
