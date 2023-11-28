from fishbonett.fbpca import pca as rsvd
from scipy.linalg import svd as csvd
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm as sparseExpm
from scipy.sparse import kron as skron
import numpy as np
from fishbonett.legendre_discretization import get_vn_squared
from fishbonett.lanczos import lanczos
import fishbonett.recurrence_coefficients as rc

def calc_U(H, dt):
    """Given the H_bonds, calculate ``U_bonds[i] = expm(-dt*H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means 'imaginary time' evolution!
    """
    return expm(-dt * 1j * H)

def calc_u_sp(H, dt):
    """Given the H_bonds, calculate ``U_bonds[i] = expm(-dt*H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means 'imaginary time' evolution!
    """
    H_sparse = csc_matrix(H)
    return sparseExpm(-dt * 1j * H_sparse)

def svd(A, b, full_matrices=False):
    """

    Args:
        A (): a matrix to be SVDed
        b (): the desired number of singular values
        full_matrices (): an option for svd in scipy. See scipy's doc

    Returns:
        see scipy's the return value of svd

    """
    dim = min(A.shape[0], A.shape[1])
    b = min(b, dim)
    if b >= 0:
        rs = rsvd(A, b, True, n_iter=2, l=2 * b)
        return rs
    else:
        return csvd(A, full_matrices=False)

def kron(a, b):
    if a is None or b is None:
        raise Exception("Can't kron none")
    if type(a) is list and type(b) is list:
        return skron(*a, *b, format='csc')
    if type(a) is list and type(b) is not list:
        return skron(*a, b, format='csc')
    if type(a) is not list and type(b) is list:
        return skron(a, *b, format='csc')
    else:
        return skron(a, b, format='csc')

def c_(dim: int):
    op = np.zeros((dim, dim))
    for i in range(dim - 1):
        op[i, i + 1] = np.sqrt(i + 1)
    return op

def get_bath_nn_paras(sd, n, domain):
    w_list, v_list = get_vn_squared(sd, n=n, domain=domain)
    v_list = np.sqrt(v_list / np.pi)
    k0 = np.linalg.norm(v_list)
    tri_mat, P = lanczos(np.diag(w_list), v_list)
    k_list = np.array([k0] + list(np.diagonal(tri_mat, -1)))
    return w_list, k_list

def get_coupling(sd, n, domain, g=1, ncap=20000):
    alphaL, betaL = rc.recurrenceCoefficients(
        n - 1, lb=domain[0], rb=domain[1], j=sd, g=g, ncap=ncap
    )
    w_list = g * np.array(alphaL)
    k_list = g * np.sqrt(np.array(betaL))
    k_list[0] = k_list[0] / g
    return w_list, k_list