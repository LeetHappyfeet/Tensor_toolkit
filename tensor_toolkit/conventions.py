"""Authoritative tensor conventions for the supported reference pipeline."""

METRIC_SIGNATURE = (-1, 1, 1, 1)
METRIC_SIGNATURE_NAME = "-+++"
DIMENSIONS = 4
DTYPE_NAME = "float64"

# R^rho_{ sigma mu nu} = d_mu Gamma^rho_{nu sigma}
#                        - d_nu Gamma^rho_{mu sigma}
#                        + Gamma^rho_{mu lambda} Gamma^lambda_{nu sigma}
#                        - Gamma^rho_{nu lambda} Gamma^lambda_{mu sigma}
RIEMANN_CONVENTION = "d_mu_Gamma_nu-d_nu_Gamma_mu+Gamma_mu_Gamma_nu-Gamma_nu_Gamma_mu"
RICCI_CONTRACTION = "R_sigma_nu = R^rho_{ sigma rho nu}"

__all__ = [
    "METRIC_SIGNATURE",
    "METRIC_SIGNATURE_NAME",
    "DIMENSIONS",
    "DTYPE_NAME",
    "RIEMANN_CONVENTION",
    "RICCI_CONTRACTION",
]
