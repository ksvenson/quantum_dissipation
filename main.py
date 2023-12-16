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


def wigner(state):
    # rho+
    # rho-
    # rho steady state
    pass


def save_eigen():
    np.save('flist', flist)
    np.save('Nlist', Nlist)
    vals = np.zeros((len(Nlist), len(flist), dim**2), dtype=complex)
    gap_vec = np.zeros((len(Nlist), len(flist), dim**2), dtype=complex)
    for Nidx, N in enumerate(Nlist):
        for fidx, f in enumerate(flist):
            H = ham(plot_delta, gamma * f, plot_U, N)
            liouv = liouvillian(H, c_ops)
            vals[Nidx, fidx], vecs = np.linalg.eig(liouv)
            gap_idx = np.argsort(np.real(vals[Nidx, fidx]))[-2]
            gap_vec[Nidx, fidx] = vecs[:, gap_idx]
            np.save(f'eigenvalues_plotdelta_{plot_delta}_plotU_{plot_U}', vals)
            np.save(f'eigenvectors_plotdelta_{plot_delta}_plotU_{plot_U}', gap_vec)
            print(f'Done (N, f) = {(N, f)}')


if __name__ == '__main__':
    all_evals = np.load(f'eigenvalues_plotdelta_{plot_delta}_plotU_{plot_U}.npy')
    all_gap_evecs = np.load(f'eigenvectors_plotdelta_{plot_delta}_plotU_{plot_U}.npy')
    for Nidx, N in enumerate(Nlist):
        spn = []
        gap = []
        fid = []
        for fidx, f in enumerate(flist):
            H = ham(plot_delta, gamma*f, plot_U, N)
            liouv = liouvillian(H, c_ops)
            rho_ss = steadystate(H, c_ops)
            evals = all_evals[Nidx, fidx]
            gap_eval = np.sort(np.real(evals))[-2]
            gap_evec = all_gap_evecs[Nidx, fidx]
            rho_1 = Qobj(inpt=np.reshape(gap_evec, (dim, dim)), isherm=True)
            rho_p = rho_1.trunc_neg()
            rho_m = (-rho_1).trunc_neg()
            xi = (rho_p + rho_m)/2

            spn.append(scaled_photon_number(rho_ss))
            gap.append(-gap_eval)
            fid.append(metrics.fidelity(rho_ss, xi))
            print(f'Done (N, f) = {(N, f)}')
        spn = np.array(spn)
        gap = np.array(gap)
        fid = np.array(fid)

        plt.figure(num='scaled_photon_number')
        plt.plot(flist, spn, label=rf'$N={N}$')
        plt.title('Scaled Photon Number')
        plt.ylabel(r'$\langle a^\dag a \rangle/N$')
        plt.xlabel(r'$\tilde{F}/\gamma$')
        plt.legend()

        plt.figure(num='gap')
        plt.plot(flist, gap/gamma, label=rf'$N={N}$')
        plt.title('Liouvillian Gap')
        plt.ylabel(r'$-\text{Re}(\lambda_1/\gamma)$')
        plt.xlabel(r'$\tilde{F}/\gamma$')
        plt.yscale('log')
        plt.legend()

        plt.figure(num='fidelity')
        plt.plot(flist, 1-fid, label=rf'$N={N}$')
        plt.title('Fidelity')
        plt.ylabel(r'$1-f(\hat{\rho}_{ss}, \hat{\xi})$')
        plt.xlabel(r'$\tilde{F}/\gamma$')
        plt.yscale('log')
        plt.legend()

    plt.show()
