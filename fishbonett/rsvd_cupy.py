import cupy as cp
import numpy as np
from cupyx.scipy.linalg import lu
from time import time

mp = cp.get_default_memory_pool()

def mult(A, B):
    return A.dot(B)

def rsvd(A, k=6, raw=False, n_iter=2, l=None):
    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert k > 0
    assert k <= min(m, n)
    assert n_iter >= 0
    assert l >= k

    if cp.isrealobj(A):
        isreal = True
    else:
        isreal = False

    # Promote the types of integer data to float data.
    dtype = (A * 1.0).dtype
    mp.free_all_blocks()
    #
    # SVD A directly if l >= m/1.25 or l >= n/1.25.
    #
    if l >= m / 1.25 or l >= n / 1.25:
        t0 = time()
        (U, s, Va) = cp.linalg.svd(A, full_matrices=False)
        t1 = time()
        print(f'FULL SVD. TIME {t1-t0}')
        #
        # Retain only the leftmost k columns of U, the uppermost
        # k rows of Va, and the first k entries of s.
        #
        return U[:, :k], s[:k], Va[:k, :]
 
    if m >= n:
 
        #
        # Apply A to a random matrix, obtaining Q.
        #
        if isreal:
            Q = cp.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
                .astype(dtype)
            Q = mult(A, Q)
            mp.free_all_blocks()
        if not isreal:
            Q = cp.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
                .astype(dtype)
            Q += 1j * cp.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
                .astype(dtype)
            mp.free_all_blocks()
            Q = mult(A, Q)
            mp.free_all_blocks()
        #
        # Form a matrix Q whose columns constitute a
        # well-conditioned basis for the columns of the earlier Q.
        #
        if n_iter == 0:
            (Q, _) = cp.linalg.qr(Q, mode='reduced')
            mp.free_all_blocks()
        if n_iter > 0:
            (Q, _) = lu(Q, permute_l=True)
            mp.free_all_blocks()
 
        #
        # Conduct normalized power iterations.
        #
        for it in range(n_iter):
 
            Q = mult(Q.conj().T, A).conj().T
            mp.free_all_blocks()
 
            (Q, _) = lu(Q, permute_l=True)
            mp.free_all_blocks()
 
            Q = mult(A, Q)
            mp.free_all_blocks()
 
            if it + 1 < n_iter:
                (Q, _) = lu(Q, permute_l=True)
                mp.free_all_blocks()
            else:
                (Q, _) = cp.linalg.qr(Q, mode='reduced')
                mp.free_all_blocks()
 
        #
        # SVD Q'*A to obtain approximations to the singular values
        # and right singular vectors of A; adjust the left singular
        # vectors of Q'*A to approximate the left singular vectors
        # of A.
        #
        QA = mult(Q.conj().T, A)
        del A
        mp.free_all_blocks()
        t0 = time()
        (R, s, Va) = cp.linalg.svd(QA, full_matrices=False)
        mp.free_all_blocks()
        t1 = time()
        print(f'rSVD. TIME {t1-t0}')
        U = Q.dot(R)
        del Q, R
        mp.free_all_blocks()
 
        #
        # Retain only the leftmost k columns of U, the uppermost
        # k rows of Va, and the first k entries of s.
        #
        return U[:, :k], s[:k], Va[:k, :]
 
    if m < n:
 
        #
        # Apply A' to a random matrix, obtaining Q.
        #
        if isreal:
            R = cp.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                .astype(dtype)
        if not isreal:
            R = cp.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                .astype(dtype)
            R += 1j * cp.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                .astype(dtype)

 
        Q = mult(R, A).conj().T
        del R
        mp.free_all_blocks()
 
        #
        # Form a matrix Q whose columns constitute a
        # well-conditioned basis for the columns of the earlier Q.
        #
        if n_iter == 0:
            (Q, _) = cp.linalg.qr(Q, mode='reduced')
            del _
            mp.free_all_blocks()
        if n_iter > 0:
            (Q, _) = lu(Q, permute_l=True)
            del _
            mp.free_all_blocks()
 
        #
        # Conduct normalized power iterations.
        #
        for it in range(n_iter):
 
            Q = mult(A, Q)
            mp.free_all_blocks()
            (Q, _) = lu(Q, permute_l=True)
            del _
            mp.free_all_blocks()
 
            Q = mult(Q.conj().T, A).conj().T
            mp.free_all_blocks()

            if it + 1 < n_iter:
                (Q, _) = lu(Q, permute_l=True)
                del _
                mp.free_all_blocks()
            else:
                (Q, _) = cp.linalg.qr(Q, mode='reduced')
                del _
                mp.free_all_blocks()
 
        #
        # SVD A*Q to obtain approximations to the singular values
        # and left singular vectors of A; adjust the right singular
        # vectors of A*Q to approximate the right singular vectors
        # of A.
        #
        t0 = time()
        (U, s, Ra) = cp.linalg.svd(mult(A, Q), full_matrices=False)
        del A
        mp.free_all_blocks()
        t1 = time()
        print(f'rSVD. TIME {t1-t0}')
        Va = Ra.dot(Q.conj().T)
 
        #
        # Retain only the leftmost k columns of U, the uppermost
        # k rows of Va, and the first k entries of s.
        #
        return U[:, :k], s[:k], Va[:k, :]

if __name__ == "__main__":
    spdup = []
    import cProfile as cpf
    for i in range(29,31):
        dim_l = 100*i
        dim = (dim_l,dim_l)
        trunc = 300
        A = cp.random.rand(*dim)
        t0 = time()
        cpf.run('rsvd(A, trunc, raw=True, l=2*trunc)')
        #rsvd(A, trunc, n_iter=4, raw=True, l=2*trunc)
        t1 = time()
        #cp.linalg.svd(A, full_matrices=False)
        t2 = time()
        A =A.get()
        t3 = time()
        #spy.linalg.svd(A, full_matrices=False)
        t4 = time()
        #pca(A, trunc, raw=True, n_iter=4, l=2*trunc)
        cpf.run('pca(A, trunc, raw=True, l=2*trunc)')
        t5 = time()
        
        #print(f"cp rSVD {t1-t0}", f"fbpca {t5-t4}")
        spdup.append((dim_l, (t5-t4)/(t1-t0)))
        print(dim_l, f"np SVD {t4-t3}",f"cp SVD {t2-t1}", f"cp rSVD {t1-t0}", f"fbpca {t5-t4}")
    print(*[f"{{{x[0]},{x[1]}}}" for x in spdup],sep=', ') 
