import numpy as np

from scipy.linalg import svd as csvd
from fishbonett.fbpca import pca as rsvd
from opt_einsum import contract as einsum
from numpy import exp
from copy import deepcopy as dcopy
from scipy.sparse import kron as skron
import scipy
from fishbonett.stuff import drude, temp_factor, _c
from fishbonett.legendre_discretization import get_vn_squared
from fishbonett.lanczos import lanczos


def eye(d):
    if d == [] or None:
        return None
    elif type(d) is int or d is str:
        return np.eye(int(d))
    elif type(d) is list or np.ndarray:
        return np.eye(*d)


def kron(a, b):
    if a is None or b is None:
        return None
    if type(a) is list and type(b) is list:
        return skron(*a, *b, format='csc')
    if type(a) is list and type(b) is not list:
        return skron(*a, b, format='csc')
    if type(a) is not list and type(b) is list:
        return skron(a, *b, format='csc')
    else:
        return skron(a, b, format='csc')


def svd(A, b, full_matrices=False):
    dim = min(A.shape[0], A.shape[1])
    b = min(b, dim)
    if b >= 0:
        # print("CSVD", A.shape, b)
        # cs = csvd(A, full_matrices=False)
        print("RRSVD", A.shape, b)
        rs = rsvd(A, b, True, n_iter=2, l=2 * b)
        # print("Difference", diffsnorm(A, *B))
        # print(cs[1] - rs[1])
        return rs
    else:
        return csvd(A, full_matrices=False)


def calc_U(H, dt):
    """Given the H_bonds, calculate ``U_bonds[i] = expm(-dt*H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means 'imaginary time' evolution!
    """
    return scipy.linalg.expm(-dt * 1j * H)


class SpinBoson:

    def __init__(self, pd, h1e, he_dy, sd, domain):
        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.len_boson = len(self.pd_boson)
        self.domain = domain
        self.he_dy = he_dy
        self.h1e = h1e
        # BEGIN discretization
        Vn = []
        coef = []
        for i, j in enumerate(sd):
            w_list, V_list = get_vn_squared(J=j, n=self.len_boson, domain=self.domain, ncap=20000)
            V_list = np.sqrt(V_list/np.pi)
            _, P = lanczos(np.diag(w_list), V_list)
            sign = np.sign(P[0, :])
            P = P.dot(np.diag(sign))
            Vn.append(V_list)
            coef.append(P)
        self.freq = w_list
        self.Vn = np.array(Vn)
        self.coef = coef
        # END discretization
        self.phase = lambda lam, t, delta: (np.exp(-1j * lam * (t + delta)) - np.exp(-1j * lam * t)) / (-1j * lam)
        # self.phase_func = lambda lam, t: np.exp(-1j * lam * (t))

    def get_h2(self, t, delta, inc_sys=True):
        print("Geting h2")
        freq = self.freq
        e = self.phase
        Vn = self.Vn
        phase_factor = np.array([e(w, t, delta) for w in freq])
        print("Geting d's")
        coef = np.array(self.coef)
        d_nt = [einsum('ik,k,k->i', Vn, coef[0, :, n], phase_factor) for n in range(len(freq)) ]
        # d_nt = [einsum('ik,k,k->i', Vn, [1]*self.len_boson, phase_factor) for n in range(len(freq))]
        # idx = np.argsort(np.abs(freq))
        # print(idx)
        # d_nt = np.array(d_nt)[idx]
        d_nt = d_nt[::-1]

        h2 = []
        he_dy = self.he_dy
        for i, dt in enumerate(d_nt):
            d1 = self.pd_boson[i]
            d2 = self.pd_spin
            c1 = _c(d1)
            dtc = np.array(dt).conj()
            coup = 0
            for j, dy in enumerate(he_dy):
                a = dt[j] * c1 + dtc[j] * c1.T
                coup += kron(dt[j] * c1 + dtc[j] * c1.T, dy)
            h2.append((coup, d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_spin
        site = delta * kron(np.eye(d1), self.h1e)
        if inc_sys is True:
            print(h2[-1][0].shape, site.shape)
            h2[-1] = (h2[-1][0] + site, d1, d2)
        else:
            h2[-1] = (h2[-1][0], d1, d2)
        return h2

    def get_u(self, t, dt, mode='normal', factor=1, inc_sys=True):
        H = self.get_h2(t, dt, inc_sys)
        U1 = dcopy(H)
        U2 = dcopy(U1)
        for i, h_d1_d2 in enumerate(H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h.toarray() / factor, 1)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            # print(u)
            u1 = u.reshape([r0, s0, r1, s1])
            u2 = np.transpose(u1, [1, 0, 3, 2])
            U1[i] = u1
            U2[i] = u2
            print("Exponential", i, r0 * s0, r1 * s1)
        return U1, U2


if __name__ == '__main__':
    import numpy as np
    from backwardSpinBoson import SpinBoson
    from fishbonett.spinBosonMPS import SpinBoson1D
    from fishbonett.stuff import sigma_x, sigma_z, temp_factor, drude, lorentzian
    from time import time
    import sys

    bath_length = 200
    phys_dim = 20
    threshold = 1e-3
    coup = 4
    bond_dim = 1000
    tmp = 2
    bath_freq = 4

    a = [phys_dim] * bath_length

    pd = a[::-1] + [2]

    g = 500 #+ bath_freq * 500

    temp = 226.00253972894595 * 0.5 * tmp
    j1 = lambda w: drude(w, lam=coup * 78.539815 / 2, gam=bath_freq * 4 * 19.634953749999998) * temp_factor(temp,w)
    j1 = lambda w: lorentzian(60, w, coup * 78.539815, omega=bath_freq * 4 * 19.634953749999998) * temp_factor(temp,w)
    j2 = lambda w: drude(w, lam=coup * 78.539815 / 2, gam=bath_freq * 4 * 19.634953749999998) * temp_factor(temp,w)
    j2 = lambda w: lorentzian(60, w, coup * 78.539815, omega=bath_freq * 4 * 19.634953749999998) * temp_factor(temp, w)

    eth = SpinBoson(pd, h1e=100*sigma_x, he_dy=[0.1*sigma_z, 1*sigma_x], sd=[j1, j2], domain=[-g, g])
    # print(eth.Vn)
    # exit()
    etn = SpinBoson1D(pd)
    etn.B[-1][0, 1, 0] = 0
    etn.B[-1][0, 0, 0] = 1

    dt = 0.001 / 1 / 2
    num_steps = 100 * 1 * 2

    p = []

    t = 0.
    for tn in range(num_steps):
        U1, U2 = eth.get_u(2 * tn * dt, 2 * dt, mode='normal', factor=2)

        t0 = time()
        etn.U = U1
        for j in range(bath_length - 1, 0, -1):
            print("j==", j, tn)
            etn.update_bond(j, bond_dim, threshold, swap=1)

        etn.update_bond(0, bond_dim, threshold, swap=0)
        etn.update_bond(0, bond_dim, threshold, swap=0)
        t1 = time()
        t = t + t1 - t0

        # U1, U2 = eth.get_u((2*tn+1) * dt, dt, mode='reverse')

        t0 = time()
        etn.U = U2
        for j in range(1, bath_length):
            print("j==", j, tn)
            etn.update_bond(j, bond_dim, threshold, swap=1)

        theta = etn.get_theta1(bath_length)  # c.shape vL i vR
        rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
        pop = np.einsum('ij,ji', rho, sigma_z)
        p = p + [pop]

        t1 = time()
        t = t + t1 - t0

    # t1 = time()
    pop = [x.real for x in p]
    print("population", pop)
    print(t)
    pop = np.array(pop)
