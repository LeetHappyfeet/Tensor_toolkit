def takeFiniteDifference2_2(A, k1, k2, delta):
    s = np.shape(A)

    if isinstance(A, np.ndarray):
        B = np.zeros_like(A)
    elif isinstance(A, torch.Tensor):
        B = torch.zeros_like(A)
    else:
        raise ValueError("Unsupported array type")

    if s[k1] >= 3 and s[k2] >= 3:
        if k1 == k2:
            if k1 == 0:
                B[1:-1, :, :, :] = (A[2:, :, :, :] - 2 * A[1:-1, :, :, :] + A[:-2, :, :, :]) / (delta[k1] ** 2)
                B[0, :, :, :] = B[1, :, :, :]
                B[-1, :, :, :] = B[-2, :, :, :]
            elif k1 == 1:
                B[:, 1:-1, :, :] = (A[:, 2:, :, :] - 2 * A[:, 1:-1, :, :] + A[:, :-2, :, :]) / (delta[k1] ** 2)
                B[:, 0, :, :] = B[:, 1, :, :]
                B[:, -1, :, :] = B[:, -2, :, :]
            elif k1 == 2:
                B[:, :, 1:-1, :] = (A[:, :, 2:, :] - 2 * A[:, :, 1:-1, :] + A[:, :, :-2, :]) / (delta[k1] ** 2)
                B[:, :, 0, :] = B[:, :, 1, :]
                B[:, :, -1, :] = B[:, :, -2, :]
            elif k1 == 3:
                B[:, :, :, 1:-1] = (A[:, :, :, 2:] - 2 * A[:, :, :, 1:-1] + A[:, :, :, :-2]) / (delta[k1] ** 2)
                B[:, :, :, 0] = B[:, :, :, 1]
                B[:, :, :, -1] = B[:, :, :, -2]
        else:
            kL = max(k1, k2)
            kS = min(k1, k2)

            x1 = slice(2, s[kS])
            x0 = slice(1, s[kS] - 1)
            x_1 = slice(None, s[kS] - 2)

            y1 = slice(2, s[kL])
            y0 = slice(1, s[kL] - 1)
            y_1 = slice(None, s[kL] - 2)

            if kS == 0:
                if kL == 1:
                    B[x0, y0, :, :] = 1 / (2 ** 2 * delta[kL] * delta[kS]) * (
                            A[x_1, y_1, :, :] - A[x_1, y1, :, :] - A[x1, y_1, :, :] + A[x1, y1, :, :])

                elif kL == 2:
                    B[x0, :, y0, :] = 1 / (2 ** 2 * delta[kL] * delta[kS]) * (
                            A[x_1, :, y_1, :] - A[x_1, :, y1, :] - A[x1, :, y_1, :] + A[x1, :, y1, :])

                elif kL == 3:
                    B[x0, :, :, y0] = 1 / (2 ** 2 * delta[kL] * delta[kS]) * (
                            A[x_1, :, :, y_1] - A[x_1, :, :, y1] - A[x1, :, :, y_1] + A[x1, :, :, y1])

            elif kS == 1:
                if kL == 2:
                    B[:, x0, y0, :] = 1 / (2 ** 2 * delta[kL] * delta[kS]) * (
                            A[:, x_1, y_1, :] - A[:, x_1, y1, :] - A[:, x1, y_1, :] + A[:, x1, y1, :])

                elif kL == 3:
                    B[:, x0, :, y0] = 1 / (2 ** 2 * delta[kL] * delta[kS]) * (
                            A[:, x_1, :, y_1] - A[:, x_1, :, y1] - A[:, x1, :, y_1] + A[:, x1, :, y1])

            elif kS == 2:
                if kL == 3:
                    B[:, :, x0, y0] = 1 / (2 ** 2 * delta[kL] * delta[kS]) * (
                            A[:, :, x_1, y_1] - A[:, :, x_1, y1] - A[:, :, x1, y_1] + A[:, :, x1, y1])

    return B
