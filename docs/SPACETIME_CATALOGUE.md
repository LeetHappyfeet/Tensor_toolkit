# Spacetime catalogue — first physics expansion (Development)

Tensor Toolkit's authoritative metrics implement `Metric.evaluate(sparse_4d_grid)` and
return covariant `float64` values, index-first with signature (-,+,+,+).
`tensor_toolkit.spacetime.describe_spacetime(metric)` supplies immutable chart,
coordinate-unit, validity-domain, physical-source classification, and reference
metadata. The metadata is persisted with every authoritative `ExperimentResult`.
Coordinate labels and units are not interchangeable: `t` in geometrized
coordinates is not automatically `ct` in SI metres.

## Reference and interpretation

Thomas Mueller et al., *Catalogue of Spacetimes*, arXiv:0904.4184v3:
https://arxiv.org/pdf/0904.4184

The PDF documents the metric conventions and coordinates in sections 1–2.
Relevant direct sources are Minkowski (2.1), Schwarzschild isotropic (2.2.3),
Alcubierre (2.3), FRW (2.9), Kerr Boyer–Lindquist (2.14.1),
Reissner–Nordstroem (2.20), and de Sitter (2.21).

**Source fidelity:** the supplement's Cartesian Kerr–Schild representation
is a standard alternative coordinate chart of Kerr, NOT the Boyer–Lindquist
formula printed in section 2.14. The `LinearizedPlaneWaveMetric` is the
usual weak TT perturbation, NOT the exact sandwich wave of section 2.19.
The static weak-field 1PN metric is another separately derived,
truncated approximation, not a metric table copied from the PDF.

## Built-in experiment IDs

| ID | Class | Coordinates | Meaning |
|----|-------|-------------|---------|
| minkowski | MinkowskiMetric | t,x,y,z (geometrized) | flat reference |
| de-sitter | DeSitterFlatMetric | t,x,y,z (geometrized) | flat-slicing de Sitter |
| alcubierre | AlcubierreMetric | t,x,y,z (geometrized) | prescribed warp geometry |
| schwarzschild | SchwarzschildIsotropicMetric | ct,x,y,z (metres) | exterior static spherical |
| kerr | KerrSchildMetric | ct,x,y,z (metres) | exterior Kerr–Schild Cartesian |
| kerr-bl | KerrBoyerLindquistMetric | ct,r,theta,phi (m,m,rad,rad) | exterior Boyer–Lindquist Kerr |
| reissner-nordstrom | ReissnerNordstromMetric | ct,r,theta,phi (m,m,rad,rad) | subextremal charged exterior |
| schwarzschild-interior | ConstantDensityInteriorMetric | ct,r,theta,phi (m,m,rad,rad) | constant-density static star interior |\n| weak-field-1pn | WeakField1PNMetric | ct,x,y,z (metres) | static spherical 1PN truncation |
| flrw | FLRWMetric (k=0) | t,x,y,z (geometrized) | flat spatial stereographic chart |
| flrw-closed | FLRWMetric (k=+1) | t,x,y,z (geometrized) | positive spatial curvature |
| flrw-open | FLRWMetric (k=-1) | t,x,y,z (geometrized) | negative curvature; r<2 patch |
| plane-wave | LinearizedPlaneWaveMetric | t,x,y,z (geometrized) | weak plus/cross TT wave |

Default compact-object grids deliberately avoid origin, poles and horizons;
they are NOT centered symmetric [-extent,+extent] grids. The CLI accepts
`--points` on these coordinate charts, but refuses generic `--extent` since
that option would silently change the chart to an invalid centered box.
Construct `Experiment(metric, axes=..., outputs=...)` in Python for custom
domain-aware grids.

Examples:

```bash
tensor-toolkit list
tensor-toolkit run schwarzschild --fields metric --output results/schwarzschild
tensor-toolkit run kerr --fields metric --output results/kerr-ks
tensor-toolkit run kerr-bl --fields metric --output results/kerr-bl
tensor-toolkit run reissner-nordstrom --fields metric
tensor-toolkit run flrw --fields metric einstein
tensor-toolkit run plane-wave --fields metric
```

`tensor-toolkit visualize` retains its Cartesian volume GUI and does not
expose the spherical/offset exterior grids as x/y/z volumes.

## Coordinate and physics cautions

- Kerr `spin` is dimensionless signed `a/m`, with `m=GM/c²` and
  `|spin|<1`. The reference implementations restrict Kerr grids to
  the exterior (even though Kerr–Schild coordinates can extend further).
  Boyer–Lindquist excludes polar axis and horizon. A zero-spin
  Kerr–Schild chart is Schwarzschild **Kerr–Schild**, not Schwarzschild
  isotropic coordinates; equal component-by-component comparison of
  the two different charts is invalid.
- Interior Schwarzschild models a static homogeneous-density star with
  surface radius R > 9GM/(4c²), in an areal spherical chart. Its metric
  matches the exterior Schwarzschild areal chart at the surface. It is
  analytical and is not a TOV solver or evolving matter source.
- Reissner–Nordstroem uses physical SI charge `Q` in coulombs and
  `q²=GQ²/(4*pi*epsilon0*c^4)`, rather than silently reusing a differently
  normalized charge symbol from the Catalogue. The registered example
  is subextremal and outside the outer horizon.
- FLRW uses an explicitly chosen scale factor `a(t)=a0 exp(Ht)` and
  spatial `a²/(1+k*r²/4)²`. Providing a scale factor alone does **not**
  solve for its material sources. `k=-1` requires stereographic radius <2.
- The weak-field metric is `g00=-(1-2u+2u²)`,
  `gij=(1+2u) delta_ij`, `u=GM/(rc²)`. It requires `u<0.05`;
  results near that upper cutoff are still approximate and require
  convergence/accuracy checks.
- The TT wave assumes `sqrt(plus²+cross²)<0.1`. It is a first-order
  perturbative metric; nonlinear Einstein-tensor residuals at O(h²) must
  not be misinterpreted as the wave's exact material source.
- Metrics are **prescribed**. Deriving `T_mu_nu` from a sampled metric
  does not automatically evolve material fields or determine their stability.
- Numerical derivatives near any validity-domain boundary require the
  entire finite-difference halo, not just the requested center, to lie
  within the permitted chart. Pick safely separated experiment axes.

## Physical-object composition and 1PN model

`Body(name, mass, position, velocity, radius=0, properties=PhysicalProperties(...))`
keeps the existing point-mass API. Optional independent components are
`RotationProperties`, `MaterialProperties`,
`ElectromagneticProperties` and `PropulsionProperties`.
Their presence does not turn on spin evolution, charge forces, collision
response, variable mass, constitutive matter physics or any other force by
itself. These are intentionally future solver inputs, not fictitious physics.

`CentralBody1PN(primary="star")` is a velocity-dependent reference
DynamicsModel for one **stationary** positive-mass primary and passive probes.
It adds the standard leading test-particle 1PN acceleration in the selected
central rest chart to Newtonian gravity:

```text
a = -mu*r/r^3 + mu/(c^2*r^3) * [(4*mu/r - v.v)*r + 4*(r.v)*v]
mu = G*M
```

Use `simulate(system, ..., method="rk4", dynamics=CentralBody1PN(primary="star"))`.
This is not general relativistic N-body backreaction or a full EIH integrator.
It rejects moving primaries, multiple massive bodies, surface penetration and
states outside its stated weak-field/slow-motion range. Standard Newtonian
energy diagnostics must not be read as conserved 1PN energy; model-aware
diagnostics are scheduled for the next pass.

## Deliberately deferred

This first pass does not implement adaptive integrators, dense output,
full multi-body 1PN, propagating fuel mass, source/matter evolution,
spacetime interpolation caches, or ADM/BSSN evolution. Those are separate
physics and numerical-validation milestones.
