import numpy as np

"""
    decompose affine matrix into start, step and direction cosines
"""
def decompose(aff):
    (u,s,vh) = np.linalg.svd(aff[0:3,0:3])
    # remove scaling
    dir_cos = u @ vh
    step  = np.diag( np.linalg.inv(dir_cos) @ aff[0:3,0:3])
    start = np.squeeze(np.asarray(np.linalg.inv(dir_cos) @ aff[0:3,3]))
    return start, step, dir_cos


"""
create affine matrix from start, step and direction cosines
"""
def compose(start, step, dir_cos):
    aff=np.eye(4)

    aff[0:3,0:3] = dir_cos @ np.diag(step)
    aff[0:3,3]   = dir_cos @ start

    return aff
