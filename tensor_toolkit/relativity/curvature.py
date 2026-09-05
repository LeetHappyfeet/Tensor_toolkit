"""Observer-independent and observer-dependent curvature diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .frames import ObserverFrame, normalize_timelike


def lower_riemann(metric: np.ndarray, riemann_mixed: np.ndarray) -> np.ndarray:
    metric = np.asarray(metric, dtype=np.float64)
    riemann_mixed = np.asarray(riemann_mixed, dtype=np.float64)
    if metric.shape != (4, 4) or riemann_mixed.shape != (4, 4, 4, 4):
        raise ValueError("expected point metric (4,4) and Riemann (4,4,4,4)")
    return np.einsum("ar,rsmn->asmn", metric, riemann_mixed)


def ricci_from_point_riemann(riemann_mixed: np.ndarray) -> np.ndarray:
    riemann_mixed = np.asarray(riemann_mixed, dtype=np.float64)
    if riemann_mixed.shape != (4, 4, 4, 4):
        raise ValueError("Riemann tensor must have shape (4,4,4,4)")
    return np.einsum("rsrn->sn", riemann_mixed)


def ricci_scalar_point(metric: np.ndarray, ricci: np.ndarray) -> float:
    inverse = np.linalg.inv(np.asarray(metric, dtype=np.float64))
    return float(np.einsum("mn,mn->", inverse, ricci))


def ricci_square(metric: np.ndarray, ricci: np.ndarray) -> float:
    inverse = np.linalg.inv(np.asarray(metric, dtype=np.float64))
    return float(np.einsum("ma,nb,mn,ab->", inverse, inverse, ricci, ricci))


def kretschmann_scalar(metric: np.ndarray, riemann_mixed: np.ndarray) -> float:
    metric = np.asarray(metric, dtype=np.float64)
    inverse = np.linalg.inv(metric)
    lower = lower_riemann(metric, riemann_mixed)
    raised = np.einsum(
        "ae,bf,cg,dh,efgh->abcd",
        inverse,
        inverse,
        inverse,
        inverse,
        lower,
        optimize=True,
    )
    return float(np.einsum("abcd,abcd->", lower, raised))


@dataclass(frozen=True)
class CurvatureDiagnostics:
    ricci_scalar: float
    ricci_square: float
    kretschmann: float


def curvature_diagnostics(metric: np.ndarray, riemann_mixed: np.ndarray) -> CurvatureDiagnostics:
    ricci = ricci_from_point_riemann(riemann_mixed)
    return CurvatureDiagnostics(
        ricci_scalar=ricci_scalar_point(metric, ricci),
        ricci_square=ricci_square(metric, ricci),
        kretschmann=kretschmann_scalar(metric, riemann_mixed),
    )


def tidal_tensor(
    metric: np.ndarray,
    riemann_mixed: np.ndarray,
    frame: ObserverFrame,
) -> np.ndarray:
    """Return the observer-frame electric/tidal Riemann tensor E_ij."""

    metric = np.asarray(metric, dtype=np.float64)
    lower = lower_riemann(metric, riemann_mixed)
    u = normalize_timelike(metric, frame.four_velocity)
    spatial = frame.spatial_basis
    return np.einsum(
        "im,a,jn,b,manb->ij",
        spatial,
        u,
        spatial,
        u,
        lower,
        optimize=True,
    )


def tidal_eigensystem(tidal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tidal = np.asarray(tidal, dtype=np.float64)
    if tidal.shape != (3, 3):
        raise ValueError("tidal tensor must have shape (3, 3)")
    return np.linalg.eigh(0.5 * (tidal + tidal.T))


def geodesic_deviation_acceleration(tidal: np.ndarray, separation) -> np.ndarray:
    tidal = np.asarray(tidal, dtype=np.float64)
    separation = np.asarray(separation, dtype=np.float64)
    if tidal.shape != (3, 3):
        raise ValueError("tidal tensor must have shape (3, 3)")
    if separation.shape != (3,):
        raise ValueError("separation must have shape (3,)")
    return -tidal @ separation
