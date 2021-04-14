import numpy as np
from scipy.optimize import curve_fit
# from fishbonett.starSpinBoson import SpinBoson, SpinBoson1D
from fishbonett.backwardSpinBoson import SpinBoson, SpinBoson1D
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude1, lemmer
from scipy.linalg import expm
from time import time

bath_length = 60
phys_dim = 30
bond_dim = 60
a = [np.ceil(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
a = [int(x) for x in a]

a = [phys_dim]*bath_length
print(a)

pd = a[::-1] + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.


# spectral density parameters
g = 1000
eth.domain = [-g, g]
temp = 129.2425
j = lambda w: lemmer(w, lam=333.564*2, k=4.16955, wm=333.564) * temp_factor(temp,w)

eth.sd = j

eth.he_dy = (sigma_z)/2
eth.h1e = 0*sigma_z/2 + (-333.564)*sigma_x / 2

eth.build(g=1., ncap=20000)
# print(eth.w_list)
# print(eth.k_list)

# U_one = eth.get_u(dt=0.002, t=0.2)

# ~ 0.5 ps ~ 0.1T
p = []


threshold = 1e-5
dt = 0.0006
num_steps = 300

s_dim = np.empty([0,0])

t = 0.
for tn in range(num_steps):
    U1, U2 = eth.get_u(2*tn*dt, dt,mode='normal')

    t0 = time()
    etn.U = U1
    for j in range(bath_length-1,0,-1):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold, swap=1)

    etn.update_bond(0, bond_dim, threshold, swap=0)
    etn.update_bond(0, bond_dim, threshold, swap=0)
    t1 = time()
    t = t + t1 - t0

    U1, U2 = eth.get_u((2*tn+1) * dt, dt, mode='reverse')

    t0 = time()
    etn.U = U2
    for j in range(1, bath_length):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold,swap=1)

    dim = [len(s) for s in etn.S]
    s_dim = np.append(s_dim, dim)

    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    pop = np.einsum('ij,ji', rho, sigma_z)
    p = p + [pop]
    t1 = time()
    t = t + t1 - t0

# t1 = time()
pop = [x.real for x in p]
print("population", pop)
print(t)
s_dim.astype('float32').tofile('heatmap_JCP.dat')