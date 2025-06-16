try:
    import sympy as sp
except Exception:  # no sympy
    sp = None

try:
    from analyticalEnergyTensor import analyticalEnergyTensor
except Exception:
    analyticalEnergyTensor = None

try:
    from solver.met2den import met2den
except Exception:
    met2den = None


class SimulationEngine:
    def __init__(self, metric_tensor, dt, duration, coords=None):
        self.metric = metric_tensor
        self.dt = dt
        self.duration = duration
        self.coords = coords
        self.objects = []
        self.time = 0.0
        self.stress_energy = None

    def add_object(self, obj):
        self.objects.append(obj)

    def _compute_stress_energy(self):
        if sp is not None and isinstance(self.metric, getattr(sp, 'Matrix', ())):
            if self.coords is not None and analyticalEnergyTensor is not None:
                try:
                    return analyticalEnergyTensor(self.metric, self.coords)
                except Exception:
                    return None
        if met2den is not None:
            try:
                return met2den(self.metric)
            except Exception:
                return None
        return None

    def step(self):
        self.stress_energy = self._compute_stress_energy()
        for obj in self.objects:
            if hasattr(obj, 'apply'):
                self.metric = obj.apply(self.metric, self.dt)
        self.time += self.dt
        return self.metric

    def run(self):
        steps = int(self.duration / self.dt)
        for _ in range(steps):
            self.step()
        return self.metric
