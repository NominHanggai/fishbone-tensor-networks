c = np.einsum('LIlj,LilJ->IijJ', t[:,:,0,0,:,:,0,0], t[:,:,0,0,:,:,0,0].conj())

