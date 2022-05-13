# fishbonett 🐟

## About
**fishbonett** <')+++< is a Python package for propagating the dynamics of multi-site vibronic model systems that have
fishbone-like configurations. The motivation of the fishbone-like configurations of this specific type
of models is very simple; we would like to study the dynamics of electron-transfer or excitation-energy-transfer
systems, in which each electronic or vibrational site is coupled to a bath.

Additional useful tools in the [`utility`](https://github.com/Mulliken/fishbone-tensor-networks/tree/main/utility) directory
----------------------------------------------------------------------------------------------------------------------------
1. `boys_localization.py`. A script that realizes multi-state diabatization through the Boys localization method. Ref: J. Chem. Phys. 129, 244101 (2008).
1. `legendre_discretization.py`. A tool that uses Legendre polynomials to discretize continuous spectral densities. Ref: Phys. Rev. B 92, 155126 (2015).
2. `golden_rule_rate.py`. A tool to calculate the Fermi's golden rule reaction rate, given a spectral density, an electronic coupling, and a free energy.
3. `transfer_tensor_method.py`. A tool that uses the transfer-tensor method to predict long-time dynamics of open quantum systems. Ref: Phys. Rev. Lett. 112, 110401 (2014).
