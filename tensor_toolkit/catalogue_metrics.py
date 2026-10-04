"""Additional prescribed spacetime metrics, independent of VTK.

All evaluate() methods return covariant (-,+,+,+) arrays shaped
(4, 4, *broadcast_shape). SI length charts use x0=ct in metres; the
cosmological/wave charts are geometrized. Explicit chart domains are
checked before attempting finite differences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
import numpy as np

from tensor_toolkit.constants import GRAVITATIONAL_CONSTANT as G, SPEED_OF_LIGHT as C
from tensor_toolkit.spacetime import (
    CATALOGUE_REFERENCE, CoordinateChart, SpacetimeDefinition,
)

EPSILON_0 = 8.8541878128e-12  # vacuum permittivity, F/m


def _shape(grid):
    return np.broadcast_shapes(*(np.shape(item) for item in grid))


def _empty(grid):
    return np.zeros((4, 4, *_shape(grid)), dtype=np.float64)


def _mass_length(mass_kg):
    mass = float(mass_kg)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("mass_kg must be finite and positive")
    return G * mass / C**2


@dataclass(frozen=True)
class KerrBoyerLindquistMetric:
    """Kerr exterior, (ct, areal r, theta, phi); section 2.14.1, eq. 2.14.1.

    spin is signed a/m = cJ/(GM^2), |spin| < 1. Chart excludes the
    outer horizon and polar axis. This is NOT Cartesian geometry.
    """
    mass_kg: float
    spin: float = 0.5
    name: str = "Kerr (Boyer-Lindquist)"
    coordinates: tuple[str, str, str, str] = ("ct", "r", "theta", "phi")
    definition: ClassVar[SpacetimeDefinition] = SpacetimeDefinition(
        "Kerr", CoordinateChart(("ct", "r", "theta", "phi"),
        ("m", "m", "rad", "rad"), "spherical",
        "r > m+sqrt(m^2-a^2), 0 < theta < pi"),
        "exact", CATALOGUE_REFERENCE + ", section 2.14.1")

    def evaluate(self, coordinate_grid):
        _, r, theta, _ = coordinate_grid
        m = _mass_length(self.mass_kg)
        s = float(self.spin)
        if not np.isfinite(s) or abs(s) >= 1:
            raise ValueError("Kerr spin requires |a/m| < 1")
        a = m * s
        r, theta = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(theta, dtype=float))
        if (np.any(~np.isfinite(r)) or np.any(~np.isfinite(theta)) or
                np.any(r <= m + np.sqrt(m*m-a*a)) or
                np.any(theta <= 0) or np.any(theta >= np.pi)):
            raise ValueError("Boyer-Lindquist grid must lie outside horizon and polar axis")
        sn2 = np.sin(theta)**2
        sigma = r*r + a*a * np.cos(theta)**2
        delta = r*r - 2*m*r + a*a
        g = _empty(coordinate_grid)
        g[0, 0] = -(1 - 2*m*r/sigma)
        g[0, 3] = g[3, 0] = -2*m*a*r*sn2/sigma
        g[1, 1] = sigma/delta
        g[2, 2] = sigma
        g[3, 3] = (r*r+a*a+2*m*a*a*r*sn2/sigma)*sn2
        return g


@dataclass(frozen=True)
class KerrSchildMetric:
    """Exterior Kerr in Cartesian Kerr-Schild coordinates (ct,x,y,z).

    This chart is independently specified; the supplied Catalogue of
    Spacetimes gives the Boyer-Lindquist form in section 2.14, not this
    Cartesian Kerr-Schild representation. z is the spin axis.
    """
    mass_kg: float
    spin: float = 0.5
    name: str = "Kerr (Cartesian Kerr-Schild)"
    coordinates: tuple[str, str, str, str] = ("ct", "x", "y", "z")
    definition: ClassVar[SpacetimeDefinition] = SpacetimeDefinition(
        "Kerr", CoordinateChart(("ct", "x", "y", "z"),
        ("m", "m", "m", "m"), "kerr-schild-cartesian",
        "exterior only: Kerr-Schild radial coordinate r > r_plus"),
        "exact", "Kerr-Schild Cartesian representation (Catalogue section 2.14 is Boyer-Lindquist)")

    def evaluate(self, coordinate_grid):
        _, x, y, z = coordinate_grid
        m = _mass_length(self.mass_kg)
        s = float(self.spin)
        if not np.isfinite(s) or abs(s) >= 1:
            raise ValueError("Kerr spin requires |a/m| < 1")
        a = m*s
        x, y, z = np.broadcast_arrays(np.asarray(x, dtype=float),
                                       np.asarray(y, dtype=float), np.asarray(z, dtype=float))
        rho2 = x*x + y*y + z*z
        r2 = 0.5*(rho2-a*a + np.sqrt((rho2-a*a)**2+4*a*a*z*z))
        r = np.sqrt(np.maximum(r2, 0))
        if np.any(~np.isfinite(r)) or np.any(r <= m + np.sqrt(m*m-a*a)):
            raise ValueError("Kerr-Schild experiment is restricted to exterior r > r_plus")
        den = r*r + a*a
        l = (np.ones_like(r), (r*x+a*y)/den, (r*y-a*x)/den, z/r)
        H = m*r**3/(r**4+a*a*z*z)
        g = _empty(coordinate_grid)
        for mu in range(4):
            g[mu, mu] = -1.0 if mu == 0 else 1.0
            for nu in range(4):
                g[mu, nu] += 2*H*l[mu]*l[nu]
        return g


@dataclass(frozen=True)
class ReissnerNordstromMetric:
    """Exterior Reissner-Nordstrom metric in areal spherical (ct,r,theta,phi).

    Electric charge uses ordinary SI coulombs with q^2=GQ^2/(4*pi*eps0*c^4).
    The Catalogue's charge parameter is normalized differently; see docs.
    """
    mass_kg: float
    charge_coulombs: float = 0.0
    name: str = "Reissner-Nordstrom (areal spherical)"
    coordinates: tuple[str, str, str, str] = ("ct", "r", "theta", "phi")
    definition: ClassVar[SpacetimeDefinition] = SpacetimeDefinition(
        "Reissner-Nordstrom", CoordinateChart(("ct", "r", "theta", "phi"),
        ("m", "m", "rad", "rad"), "spherical",
        "subextremal; r > outer horizon; 0 < theta < pi"),
        "exact", CATALOGUE_REFERENCE + ", section 2.20")

    @property
    def charge_length_squared(self):
        q = float(self.charge_coulombs)
        if not np.isfinite(q):
            raise ValueError("charge must be finite")
        return G*q*q/(4*np.pi*EPSILON_0*C**4)

    def evaluate(self, coordinate_grid):
        _, r, theta, _ = coordinate_grid
        m = _mass_length(self.mass_kg)
        q2 = self.charge_length_squared
        if q2 >= m*m:
            raise ValueError("exterior RN reference requires a strictly subextremal source")
        r, theta = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(theta, dtype=float))
        if (np.any(~np.isfinite(r)) or np.any(~np.isfinite(theta)) or
                np.any(r <= m + np.sqrt(m*m-q2)) or
                np.any(theta <= 0) or np.any(theta >= np.pi)):
            raise ValueError("RN grid must lie outside outer horizon and polar axis")
        f = 1-2*m/r+q2/r**2
        g = _empty(coordinate_grid)
        g[0, 0] = -f
        g[1, 1] = 1/f
        g[2, 2] = r*r
        g[3, 3] = r*r*np.sin(theta)**2
        return g


@dataclass(frozen=True)
class FLRWMetric:
    """Homogeneous isotropic FLRW in stereographic Cartesian spatial chart.

    ds²=-dt² + a(t)² (dx²+dy²+dz²)/(1+k*r²/4)²; a(t)=a0 exp(H t).
    k=0,+1,-1; dimensionless geometrized spatial coordinates, k=-1 r<2.
    The chosen scale-factor law is a prescribed cosmology, not a matter solver.
    """
    hubble: float = 0.1
    curvature: int = 0
    scale0: float = 1.0
    name: str = "FLRW (stereographic)"
    coordinates: tuple[str, str, str, str] = ("t", "x", "y", "z")
    definition: ClassVar[SpacetimeDefinition] = SpacetimeDefinition(
        "FLRW", CoordinateChart(("t", "x", "y", "z"),
        ("geometrized",)*4, "cartesian", "a(t)>0; k=-1 requires x²+y²+z²<4"),
        "prescribed", CATALOGUE_REFERENCE + ", section 2.9 (stereographic spatial transform)")

    def evaluate(self, coordinate_grid):
        t, x, y, z = coordinate_grid
        if self.curvature not in (-1, 0, 1):
            raise ValueError("FLRW curvature must be -1, 0, or +1")
        if not np.isfinite(self.hubble) or not np.isfinite(self.scale0) or self.scale0 <= 0:
            raise ValueError("FLRW scale and Hubble parameters must be finite; scale0 positive")
        r2 = np.asarray(x)**2 + np.asarray(y)**2 + np.asarray(z)**2
        denominator = 1+self.curvature*r2/4
        if np.any(denominator <= 0):
            raise ValueError("open FLRW stereographic chart requires spatial radius < 2")
        scale = self.scale0*np.exp(float(self.hubble)*np.asarray(t, dtype=float))
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("FLRW scale factor overflow or invalid")
        spatial = (scale/denominator)**2
        g = _empty(coordinate_grid)
        g[0, 0] = -1
        for i in (1, 2, 3):
            g[i, i] = spatial
        return g


@dataclass(frozen=True)
class LinearizedPlaneWaveMetric:
    """Weak TT plane wave propagating in +z, NOT the exact sandwich wave in
    Catalogue section 2.19. Geometrized chart and small amplitudes.
    """
    plus: float = 0.03
    cross: float = 0.0
    angular_frequency: float = 1.0
    phase: float = 0.0
    name: str = "Linearized plane gravitational wave (TT)"
    coordinates: tuple[str, str, str, str] = ("t", "x", "y", "z")
    definition: ClassVar[SpacetimeDefinition] = SpacetimeDefinition(
        "linearized-plane-wave", CoordinateChart(("t", "x", "y", "z"),
        ("geometrized",)*4, "cartesian", "weak-field |h| << 1"),
        "approximate", "Linearized TT perturbation; not the exact sandwich wave of Catalogue section 2.19")

    def evaluate(self, coordinate_grid):
        t, _, _, z = coordinate_grid
        p, c, omega, phase = map(float, (self.plus, self.cross, self.angular_frequency, self.phase))
        if not np.all(np.isfinite([p, c, omega, phase])) or np.hypot(p, c) >= 0.1:
            raise ValueError("linearized wave requires finite parameters and sqrt(plus²+cross²) < 0.1")
        u = np.asarray(t)-np.asarray(z)
        hp = p*np.cos(omega*u)
        hc = c*np.cos(omega*u+phase)
        g = _empty(coordinate_grid)
        g[0, 0] = -1
        g[1, 1] = 1+hp
        g[2, 2] = 1-hp
        g[1, 2] = g[2, 1] = hc
        g[3, 3] = 1
        return g


@dataclass(frozen=True)
class WeakField1PNMetric:
    """Static isotropic Schwarzschild weak-field expansion in (ct,x,y,z).

    g00=-(1-2U/c²+2U²/c^4), gij=(1+2U/c²)delta_ij.
    This is a truncated 1PN/reference metric, NOT exact Schwarzschild.
    """
    mass_kg: float
    minimum_radius_m: float = 0.0
    name: str = "Weak-field 1PN (static spherical)"
    coordinates: tuple[str, str, str, str] = ("ct", "x", "y", "z")
    definition: ClassVar[SpacetimeDefinition] = SpacetimeDefinition(
        "weak-field-1pn", CoordinateChart(("ct", "x", "y", "z"),
        ("m",)*4, "isotropic-cartesian",
        "r > minimum_radius_m and GM/(rc²) < 0.05"),
        "approximate", "Standard static Schwarzschild 1PN expansion; supplemental to the Catalogue")

    def evaluate(self, coordinate_grid):
        _, x, y, z = coordinate_grid
        m = _mass_length(self.mass_kg)
        r = np.sqrt(np.asarray(x)**2+np.asarray(y)**2+np.asarray(z)**2)
        lower = float(self.minimum_radius_m)
        if not np.isfinite(lower) or lower < 0 or np.any(r <= lower) or np.any(m/r >= 0.05):
            raise ValueError("1PN grid requires r > minimum_radius_m and GM/(rc²) < 0.05")
        eps = m/r
        g = _empty(coordinate_grid)
        g[0, 0] = -(1-2*eps+2*eps*eps)
        for i in (1, 2, 3):
            g[i, i] = 1+2*eps
        return g


__all__ = ["KerrBoyerLindquistMetric", "KerrSchildMetric", "ReissnerNordstromMetric",
           "FLRWMetric", "LinearizedPlaneWaveMetric", "WeakField1PNMetric", "EPSILON_0"]
