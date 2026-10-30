import unittest
import numpy as np
from helios.core import Helios, dataset, simulate, featurize, validate, FEATS

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y, _ = dataset(range(0, 3)); cls.model = Helios().fit(X, y)

    def test_feature_shape(self):
        self.assertEqual(featurize(simulate("stable", 1)).shape[1], len(FEATS))

    def test_missing_channel_raises(self):
        d = simulate("stable", 1); d.pop("gx")
        with self.assertRaises(ValueError): validate(d)

    def test_saturation_flagged(self):
        d = simulate("stable", 1); d["az"][:31] = 20
        _, q = validate(d); self.assertFalse(q["ok"])

    def test_irregular_timestamps_flagged(self):
        d = simulate("stable", 1)
        d["t"] = np.cumsum(np.random.default_rng(0).uniform(0.001, 0.08, len(d["t"])))
        _, q = validate(d); self.assertFalse(q["ok"])

    def test_bad_quality_forces_safe_action(self):
        x = featurize(simulate("stable", 2))[0]
        r = self.model.infer(x, dict(ok=False, missing=0, saturated=0, irregular=1))
        self.assertIn("sensor verification", r["action"])

    def test_impact_not_continue(self):
        r = self.model.infer(featurize(simulate("irregular", 3))[0], dict(ok=True))
        self.assertNotEqual(r["action"], "Continue monitoring")

    def test_stable_reasonable(self):
        r = self.model.infer(featurize(simulate("stationary", 4))[0], dict(ok=True))
        self.assertIn(r["level"], ("NORMAL", "CAUTION"))

    def test_reproducible(self):
        a = featurize(simulate("vibration", 5)); b = featurize(simulate("vibration", 5))
        self.assertTrue(np.allclose(a, b))

if __name__ == "__main__": unittest.main()
