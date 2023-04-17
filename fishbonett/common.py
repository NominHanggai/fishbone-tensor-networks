from fishbonett.fbpca import pca as rsvd
from scipy.linalg import svd as csvd
from scipy.linalg import expm


def calc_U(H, dt):
    """Given the H_bonds, calculate ``U_bonds[i] = expm(-dt*H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means 'imaginary time' evolution!
    """
    return expm(-dt * 1j * H)


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
