import sympy as sp

def analyticalEnergyTensor(g, coords):
    # Calculate inverse metric tensor
    gi = g.inv()

    # Compute the Christoffel symbols
    Christoffel = [[[sp.S(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                Christoffel[i][j][k] = sum(
                    0.5 * gi[i, d] * (g[d, j].diff(coords[k]) + g[d, k].diff(coords[j]) - g[j, k].diff(coords[d]))
                    for d in range(4)
                )

    # Compute the Ricci tensor
    Ricci = sp.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            Ricci[i, j] = sum(
                Christoffel[a][i][j].diff(coords[a]) - Christoffel[a][i][a].diff(coords[j]) +
                sum(
                    Christoffel[a][i][j] * Christoffel[b][a][b] - Christoffel[a][i][b] * Christoffel[b][j][a]
                    for b in range(4)
                )
                for a in range(4)
            )

    # Compute the Ricci Scalar
    Ricci_Scalar = sum(Ricci[i, j] * gi[i, j] for i in range(4) for j in range(4))

    # Compute the Stress-Energy Tensor
    T = sp.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            T[i, j] = (Ricci[i, j] - 0.5 * Ricci_Scalar * g[i, j])

    # Convert to Contravariant
    T_out = sp.zeros(4, 4)
    for i in range(4):
        for j in range(4):
            T_out[i, j] = sum(
                T[a, b] * gi[a, i] * gi[b, j]
                for a in range(4)
                for b in range(4)
            )

    return T_out
