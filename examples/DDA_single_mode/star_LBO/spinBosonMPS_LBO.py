import numpy as np

from scipy.linalg import svd as csvd
from fishbonett.fbpca import pca as rsvd
from opt_einsum import contract as einsum
from scipy.sparse import kron as skron
import scipy


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


def _to_list(x):
    """
    Converts x to [x] if x is a np.ndarray. If x is None,
    convert x(=None) to []. If x is already a list of a
    np.ndarray return x itself. Else if x is not a list of
    just one np.ndarray, raise TypeError.
    :param x: an np.array or a list of one np.ndarray
    :type x:
    :return:
    :rtype:
    """
    if x is None:
        return []
    elif x is list:
        return x
    else:
        return [x]


try:
    import cupy as cp

    CUPY_SUCCESS = True
    from fishbonett.rsvd_cupy import rsvd as cursvd


    def cusvd(A, b, full_matrices=False):
        dim = min(A.shape[0], A.shape[1])
        b = min(b, dim)
        # print("CSVD", A.shape, b)
        # cs = csvd(A, full_matrices=False)
        print("RRSVD", A.shape, b)
        rs = cursvd(A, b, True, n_iter=2, l=2 * b)
        # print("Difference", diffsnorm(A, *B))
        # print(cs[1] - rs[1])
        return rs


    mempool = cp.get_default_memory_pool()
except ImportError:
    print("CuPy is not imported. Will use CPUs")
    CUPY_SUCCESS = False


class SpinBoson1D:

    def __init__(self, pd):
        def g_state(dim):
            tensor = np.zeros(dim)
            tensor[(0,) * len(dim)] = 1.
            return tensor
        self.pre_factor = 1.5
        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.B = [g_state([1, d, 1]) for d in pd]
        self.S = [np.ones([1], np.float) for d in pd]
        self.U = [np.zeros(0) for d in pd[1:]]
        self.R = [np.eye(d) for d in pd]

    def get_theta1(self, i: int):
        theta1_proj = np.tensordot(np.diag(self.S[i]), self.B[i], [1, 0])
        return einsum('aIb,IK->aKb', theta1_proj, self.R[i])

    def get_theta2(self, i: int):
        j = i + 1
        # shape: i, M, N, j
        return einsum('aIb,bJc,JL->aILc', self.get_theta1(i), self.B[j], self.R[j])

    def split_truncate_theta(self, theta, i: int, chi_max: int, eps: float, eps_LBO: float, gpu=False):
        if gpu is False or CUPY_SUCCESS is False:
            w_A, v_A = np.linalg.eigh(einsum('aIJb,aKJb->IK', theta, theta.conj()))

            # print(einsum('aIJb,aKJb->IK', theta, theta.conj())- einsum('aIJb,aKJb->IK', theta, theta.conj()).T.conj())
            print("w_A", w_A)
            chivC = np.sum(w_A > eps_LBO)
            piv = np.argsort(w_A)[::-1][:chivC]
            # if len(w_A) !=3:
            self.R[i] = v_A.T.conj()[piv, :]
            # else:
                # self.R[i] = np.eye(3)
            print('A chivC', chivC)
            w_B, v_B = np.linalg.eigh(einsum('aIJb, aIKb->JK', theta, theta.conj()))
            print("w_B", w_B)
            chivC = np.sum(w_B > eps_LBO)
            print('B chivC', chivC)
            piv = np.argsort(w_B)[::-1][:chivC]
            # if len(w_B) !=3:
            self.R[i+1] = v_B.T.conj()[piv, :]
            # else:
            # self.R[i + 1] = np.eye(3)

            theta = einsum('aIJb,IK,JL->aKLb', theta, self.R[i].T.conj(), self.R[i+1].T.conj())

            (chi_left_on_left, phys_left,
             phys_right, chi_right_on_right) = theta.shape
            theta = np.reshape(theta, [chi_left_on_left * phys_left,
                                       phys_right * chi_right_on_right])

            chi_try = int(self.pre_factor * len(self.S[i + 1])) + 10
            A, S, B = svd(theta, chi_try, full_matrices=False)
            chivC = min(chi_max, np.sum(S > eps), chi_try)
            while chivC == chi_try and chi_try < min(*theta.shape):
                print(f"Expanding chi_try by {self.pre_factor}")
                chi_try = int(round(self.pre_factor * chi_try))
                A, S, B = svd(theta, chi_try, full_matrices=False)
                chivC = min(chi_max, np.sum(S > eps), chi_try)

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
        elif gpu is True and CUPY_SUCCESS is True:
            print("GPU running")
            (chi_left_on_left, phys_left,
             phys_right, chi_right_on_right) = theta.shape
            theta = cp.array(theta)
            theta = cp.reshape(theta, [chi_left_on_left * phys_left,
                                       phys_right * chi_right_on_right])
            mempool.free_all_blocks()

            chi_try = int(self.pre_factor * len(self.S[i + 1])) + 10
            A, S, B = cusvd(theta, chi_try, full_matrices=False)
            chivC = min(chi_max, cp.sum(S > eps).item(), chi_try)
            while chivC == chi_try and chi_try < min(*theta.shape):
                print(f"Expanding chi_try by {self.pre_factor}")
                chi_try = int(round(self.pre_factor * chi_try))
                A, S, B = cusvd(theta, chi_try, full_matrices=False)
                chivC = min(chi_max, cp.sum(S > eps).item(), chi_try)

            del theta
            mempool.free_all_blocks()
            print("Error Is", cp.sum(S > eps).item(), chi_max, S[chivC:] @ S[chivC:], chivC)
            # keep the largest `chivC` singular values
            piv = cp.argsort(S)[::-1][:chivC]
            A, S, B = A[:, piv], S[piv], B[piv, :]
            S = S / cp.linalg.norm(S)
            # A: {vL*i, chivC} -> vL i vR=chivC
            A = cp.reshape(A, [chi_left_on_left, phys_left, chivC])
            # B: {chivC, j*vR} -> vL==chivC j vR
            B = cp.reshape(B, [chivC, phys_right, chi_right_on_right])
            # vL [vL'] * [vL] i vR -> vL i vR
            A = cp.tensordot(cp.diag(self.S[i] ** (-1)), A, [1, 0])
            # vL i [vR] * [vR] vR -> vL i vR
            A = cp.tensordot(A, cp.diag(S), [2, 0])
            self.S[i + 1] = S.get()
            del S
            self.B[i] = A.get()
            del A
            self.B[i + 1] = B.get()
            del B
            mempool.free_all_blocks()

    def update_bond(self, i: int, chi_max: int, eps: float, eps_LBO: float, swap, toarray=False, gpu=False):
        if not gpu or CUPY_SUCCESS is False:
            theta = self.get_theta2(i)
            if toarray is True:
                d1, d2, u_bond = self.U[i]
                u_bond = u_bond.toarray()
                u_bond = u_bond.reshape([d1, d2, d1, d2])
                # i j [i*] [j*], vL [i] [j] vR
            else:
                u_bond = self.U[i]
            print(theta.shape, u_bond.shape)
            if swap == 1:
                print("swap: on")
                utheta = einsum('ijkl,PklQ->PjiQ', u_bond, theta)
            elif swap == 0:
                print("swap: off")
                utheta = einsum('ijkl,PklQ->PijQ', u_bond, theta)
            else:
                print(swap)
                raise ValueError
            self.split_truncate_theta(utheta, i, chi_max, eps, eps_LBO)
        else:
            theta = cp.array(self.get_theta2(i))
            u_bond = cp.array(self.U[i])
            # i j [i*] [j*], vL [i] [j] vR
            print(theta.shape, u_bond.shape)
            if swap == 1:
                print("swap: on")
                utheta = einsum('ijkl,PklQ->PjiQ', u_bond, theta)
            elif swap == 0:
                print("swap: off")
                utheta = einsum('ijkl,PklQ->PijQ', u_bond, theta)
            else:
                print(swap)
                raise ValueError
            self.split_truncate_theta(utheta, i, chi_max, eps, gpu=True)
