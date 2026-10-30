"""Train + evaluate on synthetic data, save model and metrics. Run from repo root: python -m scripts.train_model"""
import json, joblib
from pathlib import Path
from sklearn.metrics import f1_score
from helios.core import Helios, dataset
X, y, _ = dataset(range(0, 3)); Xt, yt, _ = dataset(range(100, 104))
m1 = Helios().fit(X, y); Xn, yn, _ = dataset(range(3, 5))
import numpy as np
m2 = Helios().fit(np.vstack([X, Xn]), np.concatenate([y, yn]))
res = {"data": "SYNTHETIC ONLY - not real-world accuracy",
       "v1_macro_f1": f1_score(yt, m1.clf.predict(Xt), average="macro"), "v2_macro_f1": f1_score(yt, m2.clf.predict(Xt), average="macro"),
       "predictions_changed": int((m1.clf.predict(Xt) != m2.clf.predict(Xt)).sum()), "test_windows": len(yt)}
out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps(res, indent=2)); joblib.dump(m2, out / "model_v2.joblib"); print(res)
