import numpy as np
import torch

def takeFiniteDifference1_2(A, k, delta):
    s = np.shape(A)

    if isinstance(A, np.ndarray):
        B = np.zeros_like(A)
    elif isinstance(A, torch.Tensor):
        B = torch.zeros_like(A)
    else:
        raise ValueError("Unsupported array type")

    if s[k] >= 3:
        if k == 0:
            B[1:-1, :, :, :] = (A[2:, :, :, :] - A[:-2, :, :, :]) / (2 * delta[k])
            B[0, :, :, :] = B[1, :, :, :]
            B[-1, :, :, :] = B[-2, :, :, :]
        elif k == 1:
            B[:, 1:-1, :, :] = (A[:, 2:, :, :] - A[:, :-2, :, :]) / (2 * delta[k])
            B[:, 0, :, :] = B[:, 1, :, :]
            B[:, -1, :, :] = B[:, -2, :, :]
        elif k == 2:
            B[:, :, 1:-1, :] = (A[:, :, 2:, :] - A[:, :, :-2, :]) / (2 * delta[k])
            B[:, :, 0, :] = B[:, :, 1, :]
            B[:, :, -1, :] = B[:, :, -2, :]
        elif k == 3:
            B[:, :, :, 1:-1] = (A[:, :, :, 2:] - A[:, :, :, :-2]) / (2 * delta[k])
            B[:, :, :, 0] = B[:, :, :, 1]
            B[:, :, :, -1] = B[:, :, :, -2]

    return B

