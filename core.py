"""HELIOS INTELLIGENCE V1.2 core. Synthetic data only until labelled hardware recordings exist."""
import argparse
import csv
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CFG = dict(fs=50, win=50, step=25, min_conf=0.55, anomaly_pct=2, saturation_g=15.5,
           risk=dict(caution=0.35, high=0.7))
CLASSES = ["stable", "vibration", "impact", "irregular", "stationary"]
FEATS = ["a_mean", "a_var", "g_mean", "g_var", "jerk_rms", "peak_a", "dom_freq"]
CHANNELS = ["t", "ax", "ay", "az", "gx", "gy", "gz"]


def _arrays(data):
    """Normalize a mapping of channel names to one-dimensional NumPy arrays."""
    if not hasattr(data, "keys"):
        raise TypeError("sensor data must be a mapping of channel names to arrays")
    return {str(k): np.asarray(v, dtype=float).reshape(-1) for k, v in data.items()}


def validate(data):
    d = _arrays(data)
    missing_channels = [c for c in CHANNELS if c not in d]
    if missing_channels:
        raise ValueError(f"missing channels: {missing_channels}")
    lengths = {len(d[c]) for c in CHANNELS}
    if len(lengths) != 1:
        raise ValueError("sensor channels must have equal lengths")
    if not len(d["t"]):
        raise ValueError("sensor recording is empty")
    order = np.argsort(np.nan_to_num(d["t"], nan=np.inf), kind="stable")
    d = {k: v[order] for k, v in d.items()}
    xyz = np.column_stack([d[c] for c in ("ax", "ay", "az")])
    finite_all = np.column_stack([d[c] for c in CHANNELS])
    missing = float(np.isnan(finite_all).mean())
    saturated = float((np.nanmax(np.abs(xyz), axis=1) > CFG["saturation_g"]).mean())
    intervals = np.diff(d["t"])
    irregular = float((np.abs(intervals - 1 / CFG["fs"]) > 0.5 / CFG["fs"]).mean()) if len(intervals) else 0.0
    q = dict(missing=missing, saturated=saturated, irregular=irregular,
             ok=missing < .05 and saturated < .02 and irregular < .1)
    # Linear interpolation with endpoint fill, implemented without Pandas.
    for c in CHANNELS:
        v = d[c].copy()
        good = np.isfinite(v)
        if good.any():
            idx = np.arange(len(v))
            v[~good] = np.interp(idx[~good], idx[good], v[good])
        d[c] = v
    return d, q


def windows(data):
    d = _arrays(data)
    n = len(d["t"])
    for s in range(0, n - CFG["win"] + 1, CFG["step"]):
        sl = slice(s, s + CFG["win"])
        a = np.linalg.norm(np.column_stack([d[c][sl] for c in ("ax", "ay", "az")]), axis=1)
        g = np.linalg.norm(np.column_stack([d[c][sl] for c in ("gx", "gy", "gz")]), axis=1)
        jerk = np.diff(a) * CFG["fs"]
        sp = np.abs(np.fft.rfft(a - a.mean()))
        yield [a.mean(), a.var(), g.mean(), g.var(), np.sqrt((jerk ** 2).mean()),
               np.abs(a - a.mean()).max(), np.fft.rfftfreq(len(a), 1 / CFG["fs"])[sp.argmax()]]


def featurize(data):
    rows = list(windows(data))
    return np.asarray(rows, dtype=float).reshape((-1, len(FEATS)))


def simulate(label, seed, n=600):
    r = np.random.default_rng(seed)
    t = np.arange(n) / CFG["fs"]
    p = dict(stable=(.03, .02, 0), vibration=(.2, .1, 0), impact=(.05, .05, .03),
             irregular=(.15, .5, 0), stationary=(.005, .003, 0))[label]
    d = dict(t=t, ax=r.normal(0, p[0], n), ay=r.normal(0, p[0], n),
             az=1 + r.normal(0, p[0], n), gx=r.normal(0, p[1], n),
             gy=r.normal(0, p[1], n), gz=r.normal(0, p[1], n))
    if p[2]:
        for i in r.choice(n, 12, replace=False): d["az"][i] += r.uniform(2, 4)
    if label == "irregular":
        wave = np.sin(t * r.uniform(1, 3)) * 0.6
        for c in ("gx", "gy", "gz"): d[c] += wave
    return d


def dataset(seeds):
    X, y, grp = [], [], []
    for s in seeds:
        for c in CLASSES:
            f = featurize(simulate(c, s * 10 + CLASSES.index(c)))
            X.extend(f); y.extend([c] * len(f)); grp.extend([f"{c}-{s}"] * len(f))
    return np.asarray(X), np.asarray(y), np.asarray(grp)


class Helios:
    def fit(self, X, y):
        self.clf = RandomForestClassifier(200, random_state=0).fit(X, y)
        base = X[np.isin(y, ["stable", "stationary"])]
        self.mu, self.sd = base.mean(0), base.std(0) + 1e-9
        self.iso = IsolationForest(random_state=0).fit(base)
        self.thr = np.percentile(self.iso.score_samples(base), CFG["anomaly_pct"])
        return self

    def infer(self, x, quality):
        p = self.clf.predict_proba([x])[0]; k = p.argmax()
        state = self.clf.classes_[k] if p[k] >= CFG["min_conf"] else "uncertain"
        z = (x - self.mu) / self.sd
        sc = float(-self.iso.score_samples([x])[0]); alert = -sc < self.thr
        top = [FEATS[i] for i in np.argsort(-np.abs(z))[:2]]
        risk = float(np.clip(0.4 * min(abs(z[1]) / 10, 1) + 0.3 * min(abs(z[3]) / 10, 1) + 0.3 * min(abs(z[5]) / 10, 1), 0, 1))
        lvl = "HIGH RISK" if risk >= CFG["risk"]["high"] else "CAUTION" if risk >= CFG["risk"]["caution"] else "NORMAL"
        if not quality["ok"] or state == "uncertain": action = "Pause for sensor verification / repeat measurement"
        elif lvl == "HIGH RISK": action = "Stop and request human inspection"
        elif lvl == "CAUTION" or alert: action = "Reduce commanded speed (recommendation only)"
        else: action = "Continue monitoring"
        why = (f"State '{state}' (p={p[k]:.2f}). Anomaly score {sc:.2f} ({'ALERT' if alert else 'within baseline'}); "
               f"largest deviations: {top[0]} (z={z[FEATS.index(top[0])]:.1f}), {top[1]} (z={z[FEATS.index(top[1])]:.1f}). "
               f"Risk {risk:.2f} = rule-based score, NOT a failure probability. "
               f"{'Data quality poor - alert may be a sensor issue. ' if not quality['ok'] else ''}"
               "Signals do not establish terrain type or wheel slip.")
        return dict(state=state, confidence=float(p[k]), anomaly=sc, alert=bool(alert), risk=risk, level=lvl, action=action, explanation=why)


def evaluate(m, X, y, tag):
    pr = m.clf.predict(X)
    print(f"\n[{tag}] SYNTHETIC macro-F1={f1_score(y, pr, average='macro'):.3f}")
    print(classification_report(y, pr, zero_division=0)); print(confusion_matrix(y, pr, labels=CLASSES)); return pr


def read_csv(path):
    columns = {c: [] for c in CHANNELS}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in CHANNELS:
                columns[c].append(float(row[c]) if row.get(c, "").strip() else float("nan"))
    return {k: np.asarray(v, dtype=float) for k, v in columns.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); a = ap.parse_args()
    Xtr, ytr, _ = dataset(range(0, 3)); Xte, yte, _ = dataset(range(100, 104))
    base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    print("baseline macro-F1:", round(f1_score(yte, base.predict(Xte), average="macro"), 3))
    m1 = Helios().fit(Xtr, ytr); p1 = evaluate(m1, Xte, yte, "v1 initial")
    Xn, yn, _ = dataset(range(3, 5))
    m2 = Helios().fit(np.vstack([Xtr, Xn]), np.concatenate([ytr, yn])); p2 = evaluate(m2, Xte, yte, "v2 retrained")
    print("predictions changed on same test set:", int((p1 != p2).sum()), "of", len(yte))
    if a.csv:
        clean, q = validate(read_csv(a.csv)); print("quality:", q)
        print(json.dumps(m2.infer(featurize(clean)[-1], q), indent=2))
