from fishbonett.model import FishBoneH
import numpy as np
a = [3, 3, 3]
b = [2]
pd = np.array([[a, b, b, a], [a, b, b, []]], dtype=object)
# TODO handle the case above
tri = FishBoneH(pd)
tri.domain = [-1, 1]
print("_sd", tri.sd)
print("evL",tri._evL)
print("evL",tri._ebL)
print("!",tri._evL)
tri.build_coupling()
print(tri.k_list)
#print(tri.w_list)