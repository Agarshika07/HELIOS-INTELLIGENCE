import unittest
from helios.server import Engine

class ScenarioEngineTests(unittest.TestCase):
    def run_scenario(self, name, ticks=100):
        e = Engine(); e.reset(name); e.running = True
        for _ in range(ticks): e.tick()
        return e.out

    def test_flat_progresses(self):
        o = self.run_scenario('flat')
        self.assertGreater(o['position_m'], 1)
        self.assertEqual(o['state'], 'MOVING NORMALLY')

    def test_soft_sand_reduces_progress(self):
        flat = self.run_scenario('flat'); sand = self.run_scenario('soft_sand')
        self.assertLess(sand['speed_mps'], flat['speed_mps'])
        self.assertEqual(sand['state'], 'MOVING WITH RESISTANCE')

    def test_deep_sand_stuck_detection(self):
        o = self.run_scenario('deep_sand')
        self.assertTrue(o['stuck_suspected'])
        self.assertIn('STOP', o['action'])

    def test_steep_slope_holds(self):
        o = self.run_scenario('steep_slope')
        self.assertEqual(o['state'], 'SLOPE PROGRESS AT RISK')

    def test_obstacle_stops(self):
        o = self.run_scenario('obstacle')
        self.assertEqual(o['state'], 'OBSTACLE BLOCKING PATH')
        self.assertEqual(o['speed_mps'], 0)

    def test_sensor_fault_requests_hold(self):
        o = self.run_scenario('sensor_fault')
        self.assertFalse(o['quality']['ok'])
        self.assertIn('HOLD POSITION', o['action'])

if __name__ == '__main__': unittest.main()
