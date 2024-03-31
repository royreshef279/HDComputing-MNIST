import numpy as np
from scipy.spatial.distance import hamming


def gen_hv(seed,D):
    rng = np.random.default_rng(seed)
    hv = rng.choice([0, 1], size=D).astype(np.int16)
    return hv

def gen_im(seed,D,N):
    rng = np.random.default_rng(seed)
    im = []
    for n in range(N):
        im.append(gen_hv(rng,D))
    im = np.vstack(im)
    return im

