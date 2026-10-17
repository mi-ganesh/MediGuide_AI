"""Shared model logic used by BOTH train_model.py and app.py.

Symptom state vector: 1 = has it, 0 = does not have it, -1 = unknown.
"""
import re
import numpy as np

UNKNOWN, NO, YES = -1, 0, 1


# ---------------------------------------------------------------- helpers
def display_label(symptom: str) -> str:
    """'regurgitation.1' (duplicate column in the CSV) -> 'Regurgitation (variant 2)'."""
    m = re.search(r"\.(\d+)$", symptom)
    text = f"{symptom[:m.start()]} (variant {int(m.group(1)) + 1})" if m else symptom
    return text[:1].upper() + text[1:]


def entropy(p, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log2(p)).sum(axis=axis)


def encode_tristate(X):
    """Unknown/yes/no -> two binary blocks [is_yes | is_no]. Unknown = all zeros."""
    X = np.asarray(X)
    return np.hstack([X == YES, X == NO]).astype(np.float32)


def apply_temperature(P, T):
    Q = np.power(np.clip(P, 1e-12, 1.0), 1.0 / T)
    return Q / Q.sum(axis=1, keepdims=True)


def log_loss_np(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_temperature(P, y):
    """Find T that makes probabilities honest (T<1 sharpens, T>1 softens)."""
    best_t, best = 1.0, np.inf
    for t in np.linspace(0.3, 2.5, 45):
        nll = log_loss_np(apply_temperature(P, t), y)
        if nll < best:
            best_t, best = float(t), nll
    return best_t, best


def topk_accuracy(P, y, k):
    top = np.argsort(P, axis=1)[:, -k:]
    return float((top == y[:, None]).any(axis=1).mean())


# ------------------------------------------------- partial-input simulation
def mask_user_like(X, rng, pos_range=(1, 6), neg_range=(0, 12)):
    """Mimic a real user: names a few symptoms they HAVE, denies a few, rest unknown."""
    X = np.asarray(X, dtype=np.int8)
    out = np.full(X.shape, UNKNOWN, dtype=np.int8)
    step = 20000
    for start in range(0, len(X), step):
        x = X[start:start + step]
        o = out[start:start + step]  # view
        for value, (lo, hi) in ((YES, pos_range), (NO, neg_range)):
            scores = np.where(x == value, rng.random(x.shape), 2.0)
            ranks = scores.argsort(axis=1).argsort(axis=1)
            k = rng.integers(lo, hi + 1, size=len(x))[:, None]
            o[(x == value) & (ranks < k)] = value
    return out


def mask_random(X, rng, low=0.3, high=0.95):
    """Hide a random share (30%-95%) of ALL symptom values."""
    X = np.asarray(X, dtype=np.int8).copy()
    rate = rng.uniform(low, high, size=(len(X), 1))
    X[rng.random(X.shape) < rate] = UNKNOWN
    return X


# ------------------------------------------------------------------ model
class DiseaseModel:
    """Calibrated blend of one or more classifiers + adaptive question logic."""

    def __init__(self, symptoms, classes, members, weights, temperature,
                 symptom_prob, prevalence):
        self.symptoms = list(symptoms)
        self.classes = list(classes)
        self.members = members            # list of (kind, estimator); kind in {"raw","tri"}
        self.weights = list(weights)
        self.temperature = float(temperature)
        self.symptom_prob = np.asarray(symptom_prob, dtype=np.float32)  # (classes, symptoms)
        self.prevalence = np.asarray(prevalence, dtype=np.float32)
        self.index = {s: i for i, s in enumerate(self.symptoms)}

    # -- probabilities
    def raw_proba(self, X):
        X = np.asarray(X, dtype=np.int8)
        total = 0.0
        for (kind, est), w in zip(self.members, self.weights):
            feats = encode_tristate(X) if kind == "tri" else X.astype(np.float32)
            total = total + w * est.predict_proba(feats)
        return total

    def predict_proba(self, X):
        return apply_temperature(self.raw_proba(X), self.temperature)

    def state_from(self, confirmed, denied=()):
        state = np.full((1, len(self.symptoms)), UNKNOWN, dtype=np.int8)
        for s in confirmed:
            state[0, self.index[s]] = YES
        for s in denied:
            state[0, self.index[s]] = NO
        return state

    # -- adaptive questioning
    def next_question(self, state, probs, answered, n_candidates=40):
        """Pick the unanswered symptom with the highest expected information gain."""
        P = self.symptom_prob
        p_yes_all = probs @ P
        variance = probs @ (P ** 2) - p_yes_all ** 2   # how much diseases disagree on it
        variance[list(answered)] = -1.0
        cand = np.argsort(variance)[::-1][:n_candidates]
        cand = cand[variance[cand] > 1e-6]
        if cand.size == 0:
            return None
        m = len(cand)
        rows = np.arange(m)
        yes_states = np.repeat(state, m, axis=0)
        yes_states[rows, cand] = YES
        no_states = np.repeat(state, m, axis=0)
        no_states[rows, cand] = NO
        h = entropy(self.predict_proba(np.vstack([yes_states, no_states])))  # one batch call
        p_yes = np.clip(p_yes_all[cand], 0.0, 1.0)
        gain = entropy(probs) - (p_yes * h[:m] + (1.0 - p_yes) * h[m:])
        best = int(np.argmax(gain))
        return {
            "index": int(cand[best]),
            "symptom": self.symptoms[int(cand[best])],
            "info_gain": float(gain[best]),
            "p_yes": float(p_yes[best]),
        }
