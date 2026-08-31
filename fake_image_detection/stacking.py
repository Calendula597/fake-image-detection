from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy(dtype=np.float64) / max(len(x), 1)


def oof_lr(X: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=4000, solver="liblinear")
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return oof


def fit_predict_lr(Xtr, y, Xte):
    Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-8)
    Xte = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    clf = LogisticRegression(C=1.0, max_iter=4000, solver="liblinear")
    clf.fit(Xtr, y)
    return clf.predict_proba(Xte)[:, 1]


def nested_stack_oof(M: np.ndarray, y: np.ndarray, seed: int = 2026) -> Dict[str, np.ndarray]:
    """Level-2 嵌套堆叠，返回 LR / ET / GB 的 OOF 预测。"""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof_lr2 = np.zeros(len(y), dtype=np.float64)
    oof_et = np.zeros(len(y), dtype=np.float64)
    oof_gb = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(M, y):
        Rtr = np.stack([rank01(M[tr, j]) for j in range(M.shape[1])], axis=1)
        Rva = np.zeros_like(M[va], dtype=np.float64)
        for j in range(M.shape[1]):
            Rva[:, j] = np.searchsorted(np.sort(M[tr, j]), M[va, j], side="right") / len(tr)
        Xtr = np.hstack([M[tr], Rtr])
        Xva = np.hstack([M[va], Rva])
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xva_s = sc.transform(Xva)

        lr = LogisticRegression(C=0.5, max_iter=4000, solver="liblinear")
        lr.fit(Xtr_s, y[tr])
        oof_lr2[va] = lr.predict_proba(Xva_s)[:, 1]

        et = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        et.fit(Xtr, y[tr])
        oof_et[va] = et.predict_proba(Xva)[:, 1]

        gb = GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=seed)
        gb.fit(Xtr, y[tr])
        oof_gb[va] = gb.predict_proba(Xva)[:, 1]
    return {"lr2": oof_lr2, "et": oof_et, "gb": oof_gb}


def fit_stack_predict(Mtr: np.ndarray, y: np.ndarray, Mte: np.ndarray, seed: int = 2026) -> Dict[str, np.ndarray]:
    """在全部训练 OOF 上拟合元学习器并预测测试集。"""
    Rtr = np.stack([rank01(Mtr[:, j]) for j in range(Mtr.shape[1])], axis=1)
    Rte = np.zeros_like(Mte, dtype=np.float64)
    for j in range(Mtr.shape[1]):
        Rte[:, j] = np.searchsorted(np.sort(Mtr[:, j]), Mte[:, j], side="right") / len(Mtr)
    Xtr = np.hstack([Mtr, Rtr])
    Xte = np.hstack([Mte, Rte])
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    lr = LogisticRegression(C=0.5, max_iter=4000, solver="liblinear")
    lr.fit(Xtr_s, y)
    p_lr = lr.predict_proba(Xte_s)[:, 1]

    et = ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    et.fit(Xtr, y)
    p_et = et.predict_proba(Xte)[:, 1]

    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=seed)
    gb.fit(Xtr, y)
    p_gb = gb.predict_proba(Xte)[:, 1]
    return {"lr2": p_lr, "et": p_et, "gb": p_gb}


def optimize_rank_weights(M_oof: np.ndarray, y: np.ndarray, prior: Optional[np.ndarray] = None, n_iter: int = 50000, seed: int = 2026) -> Tuple[float, np.ndarray]:
    """在 OOF 上搜索最优 rank 加权权重（两阶段：粗采样 + 局部细化）。"""
    O = np.stack([rank01(M_oof[:, j]) for j in range(M_oof.shape[1])], axis=1)
    rng = np.random.default_rng(seed)
    if prior is None:
        prior = np.ones(M_oof.shape[1])
    best_auc, best_w = -1.0, None
    # 阶段 1：粗采样
    for _ in range(n_iter):
        w = rng.dirichlet(prior)
        auc = roc_auc_score(y, O @ w)
        if auc > best_auc:
            best_auc, best_w = auc, w.copy()
    # 阶段 2：在最优点附近细化
    for scale in (0.1, 0.05, 0.02):
        for _ in range(n_iter // 5):
            w = best_w + rng.normal(0, scale, len(best_w))
            w = np.clip(w, 0, None)
            s = w.sum()
            if s < 1e-12:
                continue
            w = w / s
            auc = roc_auc_score(y, O @ w)
            if auc > best_auc:
                best_auc, best_w = auc, w.copy()
    return best_auc, best_w


def blend_l2(stack_oof: Dict[str, np.ndarray], rank_pred: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """在 L2 输出和 L1 rank 结果之间做网格搜索（粗网格防过拟合）。"""
    S = np.stack(
        [rank01(stack_oof["lr2"]), rank01(stack_oof["et"]), rank01(stack_oof["gb"]), rank01(rank_pred)],
        axis=1,
    )
    best_auc, best_w = -1.0, None
    for a in np.linspace(0, 1, 11):
        for b in np.linspace(0, 1 - a, 11):
            for c in np.linspace(0, 1 - a - b, 11):
                d = 1 - a - b - c
                ww = np.array([a, b, c, d])
                auc = roc_auc_score(y, S @ ww)
                if auc > best_auc:
                    best_auc, best_w = auc, ww.copy()
    return best_auc, best_w
