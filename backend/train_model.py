"""Advanced training script for the Diseases & Symptoms model.

Usage (from the backend folder):
    python train_model.py                      # good default (10-25 min)
    python train_model.py --fast               # quick smoke test (1-2 min)
    python train_model.py --search             # also tries several configs (slow)
    python train_model.py --csv path/to/file.csv

Put Diseases_and_Symptoms_dataset.csv in backend/data/ (or pass --csv).

What is improved compared with the notebook
  1. Training mimics real users: a few symptoms present, a few denied, the rest
     unknown (the notebook only hid a flat 30% at random).
  2. Models are size-limited. An unbounded Random Forest with 100 classes can
     reach gigabytes; this one is capped and the file is compressed.
  3. Two model families (Random Forest + MLP) are trained and blended; the best
     blend is chosen on a held-out validation set that looks like real input.
  4. Probabilities are calibrated (temperature scaling), so "80% sure" means it.
  5. Evaluation reports accuracy for 1, 2, 3, 5 symptoms - not only full rows.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from engine import (DiseaseModel, apply_temperature, encode_tristate, fit_temperature,
                    log_loss_np, mask_random, mask_user_like, topk_accuracy)

BASE = Path(__file__).parent
TARGET_NAMES = ("diseases", "disease", "prognosis", "diagnosis", "condition")


# ------------------------------------------------------------------ data
def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    target = next(c for c in df.columns if c.lower() in TARGET_NAMES)
    y = df[target].astype(str).str.strip().str.lower()
    X = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce").fillna(0)
    X = (X > 0).astype(np.int8)
    empty = X.columns[X.sum() == 0]
    if len(empty):
        print(f"Dropping {len(empty)} symptom columns that are never present")
        X = X.drop(columns=empty)
    return X, y


def augment(X, y, rng, user_copies, random_copies):
    xs, ys = [X], [y]
    for _ in range(user_copies):
        xs.append(mask_user_like(X, rng)); ys.append(y)
    for _ in range(random_copies):
        xs.append(mask_random(X, rng)); ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


# ---------------------------------------------------------------- models
def configs(args):
    if args.fast:
        return ([dict(n_estimators=40, min_samples_leaf=3, max_leaf_nodes=800)],
                [dict(hidden_layer_sizes=(256,), alpha=1e-4, max_iter=12)])
    if args.search:
        return ([dict(n_estimators=150, min_samples_leaf=2, max_leaf_nodes=2500),
                 dict(n_estimators=150, min_samples_leaf=5, max_leaf_nodes=1500),
                 dict(n_estimators=200, min_samples_leaf=3, max_leaf_nodes=2000, max_features=0.15)],
                [dict(hidden_layer_sizes=(512, 256), alpha=1e-4, max_iter=40),
                 dict(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=40),
                 dict(hidden_layer_sizes=(1024, 512), alpha=1e-3, max_iter=40)])
    return ([dict(n_estimators=120, min_samples_leaf=3, max_leaf_nodes=2000)],
            [dict(hidden_layer_sizes=(512, 256), alpha=1e-4, max_iter=40)])


def fit_rf(cfg, Xa, ya, seed):
    cfg = dict(cfg)
    est = RandomForestClassifier(max_features=cfg.pop("max_features", "sqrt"),
                                 random_state=seed, n_jobs=-1, **cfg)
    return est.fit(Xa.astype(np.float32), ya)


def fit_mlp(cfg, Xa, ya, seed):
    est = MLPClassifier(activation="relu", batch_size=512, early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=4,
                        random_state=seed, **cfg)
    return est.fit(encode_tristate(Xa), ya)


def member_proba(kind, est, X):
    feats = encode_tristate(X) if kind == "tri" else np.asarray(X, dtype=np.float32)
    return est.predict_proba(feats)


# ------------------------------------------------------------ evaluation
def report(model, X, y, seed):
    rows = []
    for k in (1, 2, 3, 5):
        Xp = mask_user_like(X, np.random.default_rng(seed + k), pos_range=(k, k), neg_range=(0, 0))
        P = model.predict_proba(Xp)
        rows.append((f"{k} symptom{'s' if k > 1 else ''} given", topk_accuracy(P, y, 1), topk_accuracy(P, y, 3)))
    P = model.predict_proba(X)
    rows.append(("all symptoms known", topk_accuracy(P, y, 1), topk_accuracy(P, y, 3)))
    print(f"\n{'Input':<24}{'Top-1':>9}{'Top-3':>9}")
    for name, a1, a3 in rows:
        print(f"{name:<24}{a1:>9.1%}{a3:>9.1%}")
    macro = f1_score(y, P.argmax(1), average="macro", zero_division=0)
    print(f"Macro F1 (all symptoms known): {macro:.4f}")
    return {"rows": [dict(input=r[0], top1=r[1], top3=r[2]) for r in rows], "macro_f1_full": macro}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(BASE / "data" / "Diseases_and_Symptoms_dataset.csv"))
    ap.add_argument("--out", default=str(BASE / "model" / "model.joblib"))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    X_df, y_raw = load_dataset(args.csv)
    symptoms = list(X_df.columns)
    classes = sorted(y_raw.unique())
    y = pd.Categorical(y_raw, categories=classes).codes.astype(np.int64)
    X = X_df.to_numpy(dtype=np.int8)
    print(f"Rows: {len(X)}  Symptoms: {len(symptoms)}  Diseases: {len(classes)}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=args.seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.10, stratify=y_tr, random_state=args.seed)

    user_c, rand_c = (1, 1) if args.fast else (2, 1)
    Xa, ya = augment(X_tr, y_tr, rng, user_c, rand_c)
    print(f"Training rows after augmentation: {len(Xa)}")

    # validation set that looks like real usage (few known symptoms)
    Xv = mask_user_like(X_va, np.random.default_rng(args.seed + 99))

    rf_cfgs, mlp_cfgs = configs(args)
    best = {}
    for family, kind, cfgs, fit in (("rf", "raw", rf_cfgs, fit_rf), ("mlp", "tri", mlp_cfgs, fit_mlp)):
        for cfg in cfgs:
            print(f"\nTraining {family} {cfg} ...")
            est = fit(cfg, Xa, ya, args.seed)
            assert list(est.classes_) == list(range(len(classes))), "a disease is missing from the training split"
            P = member_proba(kind, est, Xv)
            T, nll = fit_temperature(P, y_va)
            acc3 = topk_accuracy(apply_temperature(P, T), y_va, 3)
            print(f"  validation: log-loss {nll:.3f}  top-3 {acc3:.1%}")
            if family not in best or nll < best[family]["nll"]:
                best[family] = dict(kind=kind, est=est, P=P, nll=nll, cfg=cfg)

    # choose the best blend of the two families
    rf, mlp = best["rf"], best["mlp"]
    choice = None
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):          # weight of the RF
        P = w * rf["P"] + (1 - w) * mlp["P"]
        T, nll = fit_temperature(P, y_va)
        print(f"Blend RF weight {w:.2f}: log-loss {nll:.3f}")
        if choice is None or nll < choice[2]:
            choice = (w, T, nll)
    w, T, _ = choice
    members, weights = [], []
    if w > 0:
        members.append((rf["kind"], rf["est"])); weights.append(w)
    if w < 1:
        members.append((mlp["kind"], mlp["est"])); weights.append(1 - w)
    print(f"\nChosen: RF weight {w:.2f}, MLP weight {1 - w:.2f}, temperature {T:.2f}")

    model = DiseaseModel(
        symptoms=symptoms, classes=classes, members=members, weights=weights, temperature=T,
        symptom_prob=np.stack([X_tr[y_tr == c].mean(axis=0) for c in range(len(classes))]),
        prevalence=X_tr.mean(axis=0),
    )

    metrics = report(model, X_te, y_te, args.seed)
    metrics.update(rf_weight=w, temperature=T, rf_config=rf["cfg"], mlp_config=mlp["cfg"],
                   train_minutes=round((time.time() - t0) / 60, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out, compress=3)
    metrics["model_mb"] = round(out.stat().st_size / 1e6, 1)
    (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    template = {c: {"description": "", "precautions": [], "specialist": ""} for c in classes}
    (BASE / "disease_info.template.json").write_text(json.dumps(template, indent=2))
    print(f"\nSaved {out} ({metrics['model_mb']} MB) in {metrics['train_minutes']} min")


if __name__ == "__main__":
    main()
