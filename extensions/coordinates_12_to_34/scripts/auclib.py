# The classifier of App. J (gbc/analyze.py): HistGradientBoostingClassifier with library defaults
# (sklearn 1.5.2), StratifiedKFold(5, shuffle=True), out-of-fold ROC-AUC and McFadden pseudo-R^2 with the
# training-fold base rate as the null model, metrics averaged over the 5 folds. Repetition s uses split seed s and
# classifier random_state s (s = 0 is exactly App. J).
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def ll(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cv_oof(X, y, seed, groups=None):
    if groups is None:
        split = StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    else:
        split = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups)
    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1, dtype=np.int8)
    folds = []
    for k, (tr, te) in enumerate(split):
        clf = HistGradientBoostingClassifier(random_state=seed).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        oof[te], fold_id[te] = p, k
        p0 = np.full(len(te), y[tr].mean())
        folds.append({"auc": float(roc_auc_score(y[te], p)), "mcfadden": 1 - ll(y[te], p) / ll(y[te], p0),
                      "brier": float(np.mean((p - y[te]) ** 2)), "n_test": int(len(te)),
                      "base_rate_train": float(y[tr].mean()), "n_iter": int(clf.n_iter_)})
    return oof, fold_id, folds


def summarize(folds):
    return {k: [float(np.mean([f[k] for f in folds])), float(np.std([f[k] for f in folds]))]
            for k in ["auc", "mcfadden", "brier"]}
