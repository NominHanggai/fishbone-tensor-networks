import numpy as np

from scipy.linalg import svd as csvd
from fishbonett.fbpca import pca as rsvd
from opt_einsum import contract as einsum
from copy import deepcopy as dcopy
from scipy.sparse import kron as skron
import sympy
import scipy
from fishbonett.stuff import temp_factor


def _c(dim: int):
    op = np.zeros((dim, dim))
    for i in range(dim - 1):
        op[i, i + 1] = np.sqrt(i + 1)
    return op


def kron(a, b):
    return skron(a, b, format='csc')


def svd(A, b, full_matrices=False):
    dim = min(A.shape[0], A.shape[1])
    b = min(b, dim)
    if b >= 0:
        print("RRSVD", A.shape, b)
        return rsvd(A, b, True, n_iter=2, l=2 * b)
    else:
        return csvd(A, full_matrices=False)


def calc_U(H, dt):
    return scipy.linalg.expm(-dt * 1j * H)


class SpinBoson:
    def __init__(self, pd, coup_mat, freq, temp, betaOmega=2.):
        def g_state(dim):
            tensor = np.zeros(dim)
            tensor[(0,) * len(dim)] = 1.
            return tensor

        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.B = [g_state([1, d, 1]) for d in pd]
        self.S = [np.ones([1]) for _ in pd]
        self.U = [np.zeros(0) for _ in pd[1:]]
        self.H = [np.zeros(0) for _ in pd[1:]]

        self.len_boson = len(self.pd_boson)
        self.betaOmega = betaOmega
        self.h1e = np.eye(self.pd_spin)
        self.temp = temp
        freq = np.array(freq)
        self.freq = np.concatenate((-freq, freq))
        self.coup_mat = [mat * np.sqrt(np.abs(temp_factor(temp, self.freq[n]))) for n, mat in
                         enumerate(coup_mat + coup_mat)]
        index = np.abs(self.freq).argsort()[::-1]
        self.freq = self.freq[index]
        self.heating_op = [scipy.linalg.expm(2*betaOmega*np.sign(self.freq[i]) * _c(d).T @ _c(d)) for i, d in enumerate(self.pd_boson)]
        self.heating_op = [op / np.linalg.norm(op) for op in self.heating_op]
        self.coup_mat_np = np.array(self.coup_mat)[index]

    def get_theta1(self, i: int):
        return np.tensordot(np.diag(self.S[i]), self.B[i], [1, 0])

    def get_theta2(self, i: int):
        j = (i + 1)
        return np.tensordot(self.get_theta1(i), self.B[j], [2, 0])

    def get_rdm(self):
        theta = self.get_theta1(0)
        rho = einsum('PiQ,ij,PjL->QL', theta, self.heating_op[0], theta.conj())
        for i in range(1, self.len_boson):
            rho = einsum('PQ, PiK, ij, QjL->KL', rho, self.B[i], self.heating_op[i], self.B[i].conj())
            # rho = rho/einsum('KK', rho)
        rho = einsum('PQ,PiL,QjL->ij', rho, self.B[-1], self.B[-1].conj())
        return rho

    def split_truncate_theta(self, theta, i: int, chi_max: int, eps: float):
        (chi_left_on_left, phys_left,
         phys_right, chi_right_on_right) = theta.shape
        theta = np.reshape(theta, [chi_left_on_left * phys_left,
                                   phys_right * chi_right_on_right])
        A, S, B = svd(theta, chi_max, full_matrices=False)
        chivC = min(chi_max, np.sum(S > eps))
        print("Error Is", np.sum(S > eps), chi_max, S[chivC:] @ S[chivC:], chivC)
        # keep the largest `chivC` singular values
        piv = np.argsort(S)[::-1][:chivC]
        A, S, B = A[:, piv], S[piv], B[piv, :]
        S = S / np.linalg.norm(S)
        # A: {vL*i, chivC} -> vL i vR=chivC
        A = np.reshape(A, [chi_left_on_left, phys_left, chivC])
        # B: {chivC, j*vR} -> vL==chivC j vR
        B = np.reshape(B, [chivC, phys_right, chi_right_on_right])
        # vL [vL'] * [vL] i vR -> vL i vR
        A = np.tensordot(np.diag(self.S[i] ** (-1)), A, [1, 0])
        # vL i [vR] * [vR] vR -> vL i vR
        A = np.tensordot(A, np.diag(S), [2, 0])
        self.S[i + 1] = S
        self.B[i] = A
        self.B[i + 1] = B

    def update_bond(self, i: int, chi_max: int, eps: float, swap):
        theta = self.get_theta2(i)
        U_bond = self.U[i]
        # i j [i*] [j*], vL [i] [j] vR
        print(theta.shape, U_bond.shape)
        if swap == 1:
            print("swap: on")
            Utheta = einsum('ijkl,PklQ->PjiQ', U_bond, theta)

        elif swap == 0:
            print("swap: off")
            Utheta = einsum('ijkl,PklQ->PijQ', U_bond, theta)
        else:
            print(swap)
            raise ValueError
        self.split_truncate_theta(Utheta, i, chi_max, eps)

    def get_h2(self, delta):
        print("Getting h2")
        freq = self.freq
        mat_list = self.coup_mat_np
        h2 = []
        for i, mat in enumerate(mat_list):
            d1 = self.pd_boson[i]
            d2 = self.pd_spin
            c1 = _c(d1)
            w = freq[i]
            annih = np.exp(self.betaOmega*np.sign(w))
            creat = np.exp(-1*self.betaOmega*np.sign(w))
            coup = np.kron(annih*c1 + creat*c1.T, mat)
            site = np.kron(w * c1.T @ c1, np.eye(d2))
            h2.append((delta * (coup + site), d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_spin
        site = delta * np.kron(np.eye(d1), self.h1e)
        h2[-1] = (h2[-1][0] + site, d1, d2)
        return h2

    def get_u(self, dt):
        self.H = self.get_h2(dt)
        U1 = dcopy(self.H)
        U2 = dcopy(U1)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h, 1)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            u1 = u.reshape([r0, s0, r1, s1])
            u2 = np.transpose(u1, [1, 0, 3, 2])
            U1[i] = u1
            U2[i] = u2
            print("Exponential", i, r0 * s0, r1 * s1)
        return U1, U2

if __name__ == '__main__':
    bath_length = 6
    pd = [60]*bath_length +[2]

    N = 3
    SB = 35
    delta = 0.08679 / 2 * 8065.540106923572
    freq = np.linspace(0.08679, 0.09919, N) * 8065.540106923572
    gamma = np.sqrt((delta * SB / 4) / np.sum(1 / freq))
    gamma = np.repeat(gamma, N)
    coup_mat = [x*np.diag([1, -1]) for x in gamma]

    etn = SpinBoson(pd=pd, coup_mat=coup_mat, freq=freq, temp=300, betaOmega=0.04) #
    from fishbonett.stuff import sigma_x
    from time import time
    etn.h1e = delta * sigma_x
    threshold = 1e-3
    bond_dim = 10000
    dt = 0.001 / 5
    num_steps = 200

    p = []
    U1, U2 = etn.get_u(dt)
    t0 = time()
    for tn in range(num_steps):
        etn.U = U1
        for j in range(bath_length - 1, 0, -1):
            print("j==", j, tn)
            etn.update_bond(j, bond_dim, threshold, swap=1)

        etn.update_bond(0, bond_dim, threshold, swap=0)
        etn.update_bond(0, bond_dim, threshold, swap=0)

        etn.U = U2
        for j in range(1, bath_length):
            print("j==", j, tn)
            etn.update_bond(j, bond_dim, threshold, swap=1)
        print("Getting RDM")
        rho = etn.get_rdm()
        rho = rho/np.trace(rho)
        p = p + [np.abs(rho[0, 0])]
    t1 = time()
    print(t1-t0)
    pop = [x for x in p]
    print("population", pop)


