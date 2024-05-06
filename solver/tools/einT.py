def einT2(R_munu, R, gl):
    # Initialize the Einstein tensor
    E = [[0 for _ in range(4)] for _ in range(4)]
    
    # Calculate the Einstein tensor components
    for mu in range(4):
        for nu in range(4):
            E[mu][nu] = R_munu[mu][nu] - 0.5 * gl[mu][nu] * R
    
    return E

# Example usage:
# R_munu = [[R11, R12, R13, R14], [R21, R22, R23, R24], [R31, R32, R33, R34], [R41, R42, R43, R44]]
# R = Ricci scalar
# gl = [[g11, g12, g13, g14], [g21, g22, g23, g24], [g31, g32, g33, g34], [g41, g42, g43, g44]] (inverse metric tensor)
# E = einT2(R_munu, R, gl)
