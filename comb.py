import numpy as np

from ..toycodes.a_mps import SimpleMPS as mps

d = 10;
L = 10
B = np.zeros((5, d, 5))
S = np.ones([1], np.float)
B_List = [B.copy() for i in range(L)]
S_List = [S.copy() for i in range(L)]
a = mps(B_List, S_List, "finite")


class SimpleTTPS:
    """ Simple Tree Like Tensor-Product States
        
                         ⋮
        --d33--d32--d31--E3--V3--b31--b32--b33--
                         |
        --d23--d22--d21--E2--V2--b21--b22--b23--
                         |
        --d13--d12--d11--E1--V1--b11--b12--b13--
                         |
        --d03--d02--d01--E0--V0--b01--b02--b03--
                         ⋮ 
    """

    def __init__(self, ele_list, vib_list, e_bath_list, v_bath_list):
        self.e = ele_list
        self.v = v_bath_list
        self.eb = e_bath_list
        self.vb = v_bath_list
        self.eL = len(ele_list)
        self.vL = len(vib_list)

    def get_theta1(self, idx):
        """Calculate effective single-site wave function on sites i in mixed canonical form.

        The returned array has legs ``vL, i, vR`` (as one of the Bs).
        """
        (i, n) = idx

        print(self.Ss[i], self.Bs[i])
        return np.tensordot(np.diag(self.Ss[i]), self.Bs[i], [1, 0])  # vL [vL'], [vL] i vR

    def get_theta2(self, tuple):
        """Calculate effective two-site wave function on sites i,j=(i+1) in mixed canonical form.

        The returned array has legs ``vL, i, j, vR``.
        tuple: the tuple that stores the bond matrix. For example, 
               the E0--V0 bond is labeled as (0,0),
               the E0--E1 bond is (0,1),
               the E1--V1 bond is (1,0)
               the V1--b11 bond is (1,1), 
               the E1--d11 bond is (1,-1), etc.
        """
        # j = (i + 1) % self.L
        i, j, n = tuple
        if n == 0:
            return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])
        # return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])  # vL i [vR], [vL] j vR
        if n != 0:
            assert abs(j - i) == 1
            return np.tensordot(self.get_theta1(i), self.Bs[j], [2, 0])

    # def get_dtheta2(self, tuple):
    # def get_ltheta2(self, tuple):
    # def get_rtheta2(self, tuple):


def example_TEBD_gs_tf_ising_finite(L, g):
    print("finite TEBD, imaginary time evolution, transverse field Ising")
    print("L={L:d}, g={g:.2f}".format(L=L, g=g))
    import toycodes.a_mps as a_mps
    import toycodes.b_model as b_model
    M = b_model.TFIModel(L=L, J=1., g=g, bc='finite')

    psi = a_mps.init_FM_MPS(M.L, M.d, M.bc)

    for dt in [0.1]:
        print("Hey1", psi.Bs, "\n", psi.Ss)
        U_bonds = calc_U_bonds(M.H_bonds, dt)
        update_bond(psi, 1, U_bonds[0], chi_max=40, eps=1.e-10)
        # run_TEBD(psi, U_bonds, N_steps=500, chi_max=30, eps=1.e-10)
        E = np.sum(psi.bond_expectation_value(M.H_bonds))
        print("dt = {dt:.5f}: E = {E:.13f}".format(dt=dt, E=E))
    # psi = a_mps.init_FM_MPS(M.L, M.d, M.bc)
    print('Hey', psi.Bs, "\n", psi.Ss)
    print("Site S\n", psi.Ss, "\nSite B\n", psi.Bs, "\nSite B\n", )
    print("final bond dimensions: ", psi.get_chi())
    mag_x = np.sum(psi.site_expectation_value(M.sigmax))
    mag_z = np.sum(psi.site_expectation_value(M.sigmaz))
    print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
    print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))
    if L < 20:  # compare to exact result
        from toycodes.tfi_exact import finite_gs_energy
        E_exact = finite_gs_energy(L, 1., g)
        print("Exact diagonalization: E = {E:.13f}".format(E=E_exact))
        print("relative error: ", abs((E - E_exact) / E_exact))
    return E, psi, M


example_TEBD_gs_tf_ising_finite(L=3, g=1.)
