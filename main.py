import pickle

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
flist = np.linspace(0, 5, 100)  # \tilde{F}/gamma
Nlist = np.array([1, 3, 10, 20])
w_x = np.linspace(-5, 5, 200)
w_y = np.linspace(-5, 5, 200)


def ham(delta, F, U, N):
    """
    Hamiltonian for the "driven-dissipative Kerr resonator"
    :param delta: Detuning
    :param F: Defined as \tilde{F} in paper
    :param U: Define as \tilde{U} in paper
    :param N: Thermodynamic parameter
    :return: Hamiltonian
    """
    return -delta*ad*a + (U/N/2)*ad*ad*a*a + F*np.sqrt(N)*(ad + a)


def scaled_photon_number(state):
    return np.trace(ad*a*state)/N


def plot_wigner(states, name):
    fig = plt.figure(figsize=(8, 12), constrained_layout=True, num=f'wigner_{name}')
    title = 'Wigner Distributions for '
    if name == 'rhoss':
        title += r'$\hat{\rho}_{ss}$'
    elif name == 'rhop':
        title += r'$\hat{\rho}_{1}^+$'
    elif name == 'rhom':
        title += r'$\hat{\rho}_{1}^-$'
    fig.suptitle(title)
    subfigs = fig.subfigures(nrows=4, ncols=1)
    for Nidx, N in enumerate(states):
        subfigs[Nidx].suptitle(f'$N={N}$')
        axes = subfigs[Nidx].subplots(nrows=1, ncols=3)
        for fidx, f in enumerate(states[N]):
            axes[fidx].contourf(w_x, w_y, states[N][f], 100)
            axes[fidx].set_title(rf'$f={round(f, 2)}$')
    plt.savefig(f'./figs/wigner_{name}.png')


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
    with open('all_wig_ss.pkl', 'rb') as file:
        all_wig_ss = pickle.load(file)
    with open('all_wig_p.pkl', 'rb') as file:
        all_wig_p = pickle.load(file)
    with open('all_wig_m.pkl', 'rb') as file:
        all_wig_m = pickle.load(file)

    plot_wigner(all_wig_ss, 'rhoss')
    plot_wigner(all_wig_p, 'rhop')
    plot_wigner(all_wig_m, 'rhom')

    plt.show()
    quit()

    all_evals = np.load(f'eigenvalues_plotdelta_{plot_delta}_plotU_{plot_U}.npy')
    all_gap_evals = np.sort(all_evals, axis=-1)[:, :, -2]
    crit_f_idx = np.argmax(np.real(all_gap_evals), axis=-1)
    all_gap_evecs = np.load(f'eigenvectors_plotdelta_{plot_delta}_plotU_{plot_U}.npy')
    all_wig_ss = {}
    all_wig_p = {}
    all_wig_m = {}
    for Nidx, N in enumerate(Nlist):
        spn = []
        gap = []
        fid = []
        wig_ss = {}
        wig_p = {}
        wig_m ={}
        for fidx, f in enumerate(flist):
            H = ham(plot_delta, gamma*f, plot_U, N)
            liouv = liouvillian(H, c_ops)
            rho_ss = steadystate(H, c_ops)
            gap_eval = all_gap_evals[Nidx, fidx]
            gap_evec = all_gap_evecs[Nidx, fidx]
            rho_1 = Qobj(inpt=np.reshape(gap_evec, (dim, dim)).T, isherm=True)
            rho_p = rho_1.trunc_neg()
            rho_m = (-rho_1).trunc_neg()
            xi = (rho_p + rho_m)/2

            if fidx == crit_f_idx[Nidx] or fidx == crit_f_idx[Nidx]-len(flist)//4 or fidx == crit_f_idx[Nidx]+len(flist)//4:
                wig_ss[f] = wigner(rho_ss, w_x, w_y)
                wig_p[f] = wigner(rho_p, w_x, w_y)
                wig_m[f] = wigner(rho_m, w_x, w_y)

            spn.append(scaled_photon_number(rho_ss))
            gap.append(-gap_eval)
            fid.append(metrics.fidelity(rho_ss, xi))
            print(f'Done (N, f) = {(N, f)}')
        spn = np.array(spn)
        gap = np.array(gap)
        fid = np.array(fid)

        all_wig_ss[N] = wig_ss
        all_wig_p[N] = wig_p
        all_wig_m[N] = wig_m

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

    plot_wigner(all_wig_ss, 'rhoss')
    plot_wigner(all_wig_p, 'rhop')
    plot_wigner(all_wig_m, 'rhom')

    with open('all_wig_ss.pkl', 'wb') as file:
        pickle.dump(all_wig_ss, file, pickle.HIGHEST_PROTOCOL)
    with open('all_wig_p.pkl', 'wb') as file:
        pickle.dump(all_wig_p, file, pickle.HIGHEST_PROTOCOL)
    with open('all_wig_m.pkl', 'wb') as file:
        pickle.dump(all_wig_m, file, pickle.HIGHEST_PROTOCOL)

    plt.show()
