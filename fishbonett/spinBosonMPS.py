import numpy as np

from scipy.linalg import svd as csvd
from fbpca import pca as rsvd
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

        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.B = [g_state([1, d, 1]) for d in pd]
        self.S = [np.ones([1], np.float) for d in pd]
        self.U = [np.zeros(0) for d in pd[1:]]

    def get_theta1(self, i: int):
        return np.tensordot(np.diag(self.S[i]), self.B[i], [1, 0])

    def get_theta2(self, i: int):
        j = (i + 1)
        return np.tensordot(self.get_theta1(i), self.B[j], [2, 0])

    def split_truncate_theta(self, theta, i: int, chi_max: int, eps: float, gpu=False):
        if gpu is False or CUPY_SUCCESS is False:
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
        elif gpu is True and CUPY_SUCCESS is True:
            print("1 USED", mempool.used_bytes() * 1e-6)
            print("1 TOTAl", mempool.total_bytes() * 1e-6)
            print("GPU running")
            (chi_left_on_left, phys_left,
             phys_right, chi_right_on_right) = theta.shape
            theta = cp.array(theta)
            print("2 USED", mempool.used_bytes() * 1e-6)
            print("2 TOTAl", mempool.total_bytes() * 1e-6)
            theta = cp.reshape(theta, [chi_left_on_left * phys_left,
                                       phys_right * chi_right_on_right])
            print("3 USED", mempool.used_bytes() * 1e-6)
            print("3 TOTAl", mempool.total_bytes() * 1e-6)
            mempool.free_all_blocks()
            print("4 USED", mempool.used_bytes() * 1e-6)
            print("4 TOTAl", mempool.total_bytes() * 1e-6)
            A, S, B = cusvd(theta, chi_max, full_matrices=False)
            print("5 USED", mempool.used_bytes() * 1e-6)
            print("5 TOTAl", mempool.total_bytes() * 1e-6)
            del theta
            mempool.free_all_blocks()
            print("6 USED", mempool.used_bytes() * 1e-6)
            print("6 TOTAl", mempool.total_bytes() * 1e-6)
            chivC = min(chi_max, cp.sum(S > eps).item())
            print("Error Is", cp.sum(S > eps), chi_max, S[chivC:] @ S[chivC:], chivC)
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

    def update_bond(self, i: int, chi_max: int, eps: float, swap, gpu=False):
        if not gpu or CUPY_SUCCESS is False:
            theta = self.get_theta2(i)
            u_bond = self.U[i]
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
            self.split_truncate_theta(utheta, i, chi_max, eps)
        else:
            print("-1 USED", mempool.used_bytes())
            print("-1 TOTAl", mempool.total_bytes())
            theta = cp.array(self.get_theta2(i))
            print("-2 USED", mempool.used_bytes())
            print("-2 TOTAl", mempool.total_bytes())
            u_bond = cp.array(self.U[i])
            print("-3 USED", mempool.used_bytes())
            print("-3 TOTAl", mempool.total_bytes())
            # i j [i*] [j*], vL [i] [j] vR
            # mempool.free_all_blocks()
            utheta = einsum('ijkl,PklQ->PjiQ', u_bond, theta)
            print("-5 USED", mempool.used_bytes())
            print("-5 TOTAl", mempool.total_bytes())
            del theta, u_bond
            mempool.free_all_blocks()
            print("-6 USED", mempool.used_bytes())
            print("-6 TOTAl", mempool.total_bytes())
            utheta = cp.transpose(utheta, [2, 0, 1, 3])  # vL i j vR
            print("-7 USED", mempool.used_bytes())
            print("-7 TOTAl", mempool.total_bytes())
            mempool.free_all_blocks()
            print("-8 USED", mempool.used_bytes())
            print("-8 TOTAl", mempool.total_bytes())
            self.split_truncate_theta(utheta, i, chi_max, eps, gpu=True)
            mempool.free_all_blocks()
            print("-9 USED", mempool.used_bytes())
            print("-9 TOTAl", mempool.total_bytes())
