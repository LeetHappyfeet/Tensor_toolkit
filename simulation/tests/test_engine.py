import unittest

from simulation.engine import SimulationEngine


class DummyObject:
    def apply(self, metric, dt):
        metric[0][0] += dt
        return metric


class TestSimulationEngine(unittest.TestCase):
    def test_step_updates_metric(self):
        g = [[-1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        engine = SimulationEngine(g, dt=0.1, duration=0.1)
        engine.add_object(DummyObject())
        initial = engine.metric[0][0]
        engine.step()
        self.assertAlmostEqual(engine.metric[0][0], initial + 0.1)


if __name__ == '__main__':
    unittest.main()
