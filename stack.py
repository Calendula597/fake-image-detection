from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fake_image_detection.feature_extractor import load_features
from fake_image_detection.paths import resolve_code_root, sample_csv, test_csv
from fake_image_detection.stacking import (
    blend_l2,
    fit_predict_lr,
    fit_predict_mlp,
    fit_stack_predict,
    nested_stack_oof,
    oof_lr,
    oof_mlp,
    optimize_rank_weights,
    rank01,
)

CR = resolve_code_root()
FEAT = CR / "outputs" / "features"
PRED = CR / "outputs" / "predictions"
PRED.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path):
    d = pd.read_csv(path)
    if "probability" in d.columns:
        d = d.rename(columns={"probability": "score"})
    return d.set_index("id")["score"]


def _deg_paths(tag, size, crop, n_max=3):
    """查找某成员的退化训练特征文件（deg0/deg1/...）。"""
    paths = []
    for r in range(n_max):
        p = FEAT / f"{tag}_{size}_{crop}_deg{r}_sample.npy"
        if p.exists():
            paths.append(p)
    return paths


def _fit_mlp_deg(Xs, y, Xt, deg_paths):
    """在 [干净 + 退化] 特征上训练测试头（退化匹配训练，NTIRE 思路）。OOF 仍用干净特征。"""
    from fake_image_detection.stacking import fit_predict_mlp
    if not deg_paths:
        return fit_predict_mlp(Xs, y, Xt)
    Xs_aug = [Xs] + [np.load(p).astype(np.float32) for p in deg_paths]
    X_aug = np.concatenate(Xs_aug, 0)
    y_aug = np.concatenate([y] * len(Xs_aug), 0)
    return fit_predict_mlp(X_aug, y_aug, Xt)


def build_members(lab_y, test_ids, lab_df=None):
    """从 features/ 构建 Level-1 成员。"""
    members = []

    # Community Forensics
    for tag, size in [("cf384", 384), ("cf224", 224)]:
        for crop in ("crop", "resize"):
            pair = (
                load_features(FEAT, f"commfor_{tag}", size, crop, "sample"),
                load_features(FEAT, f"commfor_{tag}", size, crop, "test"),
            )
            if pair[0] is None or pair[1] is None:
                continue
            Xs, Xt = pair
            oof = oof_mlp(Xs, lab_y)
            te = _fit_mlp_deg(Xs, lab_y, Xt, _deg_paths(f"commfor_{tag}", size, crop))
            print(f"[L1] CF {tag}_{crop} AUC={roc_auc_score(lab_y, oof):.4f} (MLP,deg)")
            members.append((f"cf_{tag}_{crop}", oof, te))

    # CLIP (clipH / clipBigG / clipH378)
    for size in (224, 378):
        for crop in ("crop", "resize"):
            for tag in ("clipH", "clipBigG", "clipH378"):
                pair = (
                    load_features(FEAT, tag, size, crop, "sample"),
                    load_features(FEAT, tag, size, crop, "test"),
                )
                if pair[0] is None or pair[1] is None:
                    continue
                Xs, Xt = pair
                oof = oof_mlp(Xs, lab_y)
                te = _fit_mlp_deg(Xs, lab_y, Xt, _deg_paths(tag, size, crop))
                print(f"[L1] {tag}_{size}_{crop} AUC={roc_auc_score(lab_y, oof):.4f} (MLP,deg)")
                members.append((f"{tag}_{size}_{crop}", oof, te))

    # DINOv3 (dinoL / dinoB / dinoH)
    for tag in ("dinoL", "dinoB", "dinoH"):
        for crop in ("crop", "resize"):
            pair = (
                load_features(FEAT, tag, 224, crop, "sample"),
                load_features(FEAT, tag, 224, crop, "test"),
            )
            if pair[0] is None or pair[1] is None:
                continue
            Xs, Xt = pair
            oof = oof_mlp(Xs, lab_y)
            te = _fit_mlp_deg(Xs, lab_y, Xt, _deg_paths(tag, 224, crop))
            print(f"[L1] {tag}_224_{crop} AUC={roc_auc_score(lab_y, oof):.4f} (MLP,deg)")
            members.append((f"{tag}_224_{crop}", oof, te))

    # CO-SPY (语义 SigLIP + 伪影 VAE)
    for tag, size in (("cospySem", 384), ("cospyArt", 224)):
        pair = (
            load_features(FEAT, tag, size, "crop", "sample"),
            load_features(FEAT, tag, size, "crop", "test"),
        )
        if pair[0] is None or pair[1] is None:
            continue
        Xs, Xt = pair
        oof = oof_mlp(Xs, lab_y)
        te = fit_predict_mlp(Xs, lab_y, Xt)
        print(f"[L1] {tag}_{size}_crop AUC={roc_auc_score(lab_y, oof):.4f} (MLP)")
        members.append((f"{tag}_{size}_crop", oof, te))

    # CLIP-L (特殊命名)
    for sname, tname, tag in [
        ("clip_vitl14_sample.npy", "clip_vitl14_test.npy", "clipL224"),
        ("clipL_336_336_sample.npy", "clipL_336_336_test.npy", "clipL336"),
    ]:
        sp, tp = FEAT / sname, FEAT / tname
        if not sp.exists() or not tp.exists():
            continue
        Xs = np.load(sp).astype(np.float32)
        Xt = np.load(tp).astype(np.float32)
        oof = oof_mlp(Xs, lab_y)
        te = fit_predict_mlp(Xs, lab_y, Xt)
        print(f"[L1] {tag} AUC={roc_auc_score(lab_y, oof):.4f} (MLP)")
        members.append((tag, oof, te))

    # FSD (描述子) + AIDE (特征) + DIFT (扩散UNet特征) — 机制不同的新信号
    for sname, tname, tag, method in [
        ("fsd_desc_sample.npy", "fsd_desc_test.npy", "fsd_desc", "mlp"),
        ("aide_256_sample.npy", "aide_256_test.npy", "aide", "mlp"),
        ("dift_mid_sample.npy", "dift_mid_test.npy", "dift", "mlp"),
    ]:
        sp, tp = FEAT / sname, FEAT / tname
        if not sp.exists() or not tp.exists():
            continue
        Xs = np.load(sp).astype(np.float32)
        Xt = np.load(tp).astype(np.float32)
        if Xs.ndim > 2:
            Xs = Xs.reshape(len(Xs), -1)
        if Xt.ndim > 2:
            Xt = Xt.reshape(len(Xt), -1)
        oof = oof_mlp(Xs, lab_y)
        te = fit_predict_mlp(Xs, lab_y, Xt)
        print(f"[L1] {tag} AUC={roc_auc_score(lab_y, oof):.4f} (MLP)")
        members.append((tag, oof, te))

    # Qwen VLM 零样本分数（语义推理机制，1 维标量，直接用，无需训练）
    qz_s, qz_t = FEAT / "qwen_zs_sample.npy", FEAT / "qwen_zs_test.npy"
    if qz_s.exists() and qz_t.exists():
        oof = np.load(qz_s).astype(np.float64).reshape(-1)
        te = np.load(qz_t).astype(np.float64).reshape(-1)
        print(f"[L1] qwen_zs AUC={roc_auc_score(lab_y, oof):.4f} (VLM zeroshot)")
        members.append(("qwen_zs", oof, te))

    # B-Free 去偏差检测分数（真 vs SD假 训练的 LR，跨生成器机制，1 维标量，直接用）
    bf_s, bf_t = FEAT / "bfree_sample.npy", FEAT / "bfree_test.npy"
    if bf_s.exists() and bf_t.exists():
        oof = np.load(bf_s).astype(np.float64).reshape(-1)
        te = np.load(bf_t).astype(np.float64).reshape(-1)
        print(f"[L1] bfree AUC={roc_auc_score(lab_y, oof):.4f} (B-Free debiased)")
        members.append(("bfree", oof, te))

    # DRCT
    for tag, crop in [
        ("convb_sdv14", "resize"),
        ("convb_genimage", "resize"),
        ("convb_sdv14", "crop"),
        ("convb_genimage", "crop"),
    ]:
        pair = (
            load_features(FEAT, f"drct_{tag}", 224, crop, "sample"),
            load_features(FEAT, f"drct_{tag}", 224, crop, "test"),
        )
        if pair[0] is None or pair[1] is None:
            continue
        Xs, Xt = pair
        oof = oof_lr(Xs, lab_y)
        te = fit_predict_lr(Xs, lab_y, Xt)
        print(f"[L1] drct_{tag}_{crop} AUC={roc_auc_score(lab_y, oof):.4f}")
        members.append((f"drct_{tag}_{crop}", oof, te))

    # Metadata 指纹特征
    meta_path = CR / "artifacts" / "metadata_fingerprints_sem.csv"
    if meta_path.exists():
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.model_selection import StratifiedKFold as SKF
        meta = pd.read_csv(meta_path)
        cols = [c for c in meta.columns if c not in {"id", "split"}]
        train_m = meta[meta["split"] == "train"].merge(lab_df, on="id")
        test_m = meta[meta["split"] == "test"]
        Xm = train_m[cols].replace([np.inf, -np.inf], np.nan).fillna(-1).to_numpy()
        Tm = test_m[cols].replace([np.inf, -np.inf], np.nan).fillna(-1).to_numpy()
        skf = SKF(5, shuffle=True, random_state=42)
        oof = np.zeros(len(lab_y), dtype=np.float64)
        for tr, va in skf.split(Xm, lab_y):
            clf = ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=2, max_features=0.8,
                class_weight="balanced", n_jobs=-1, random_state=2026,
            )
            clf.fit(Xm[tr], lab_y[tr])
            oof[va] = clf.predict_proba(Xm[va])[:, 1]
        clf = ExtraTreesClassifier(
            n_estimators=1000, min_samples_leaf=2, max_features=0.8,
            class_weight="balanced", n_jobs=-1, random_state=2026,
        )
        clf.fit(Xm, lab_y)
        te = clf.predict_proba(Tm)[:, 1]
        print(f"[L1] meta AUC={roc_auc_score(lab_y, oof):.4f}")
        members.append(("meta", oof, te))

        # 尺寸先验：生成器原生尺寸（512²/768²/1024² 等）在训练集上高度偏向 AI。
        # LOO 查表防止自身泄漏；未见尺寸按 精确尺寸→取整到8的倍数→(宽高比,面积档,mult8) 逐级回退。
        ym = train_m["label"].to_numpy() if "label" in train_m.columns else lab_y
        assert len(ym) == len(lab_y)
        tr_sizes = list(zip(train_m["width"].to_numpy(), train_m["height"].to_numpy()))
        te_sizes = list(zip(test_m["width"].to_numpy(), test_m["height"].to_numpy()))

        def _q8(s):
            return (round(s[0] / 8) * 8, round(s[1] / 8) * 8)

        def _bucket(s):
            w, h = s
            ar = w / max(h, 1)
            arb = "sq" if 0.9 <= ar <= 1.1 else ("wide" if ar > 1.1 else "tall")
            area = w * h
            sc = "s" if area < 200000 else ("m" if area < 800000 else "l")
            return (arb, sc, int(w % 8 == 0 and h % 8 == 0))

        levels = (lambda s: s, _q8, _bucket)
        cnts = [dict() for _ in levels]
        for s, l in zip(tr_sizes, ym):
            for cnt, fn in zip(cnts, levels):
                k = fn(s)
                a, b = cnt.get(k, (0, 0))
                cnt[k] = (a + (1 - int(l)), b + int(l))

        def _lookup(s, l=None):
            for cnt, fn in zip(cnts, levels):
                k = fn(s)
                n0, n1 = cnt.get(k, (0, 0))
                if l is not None:  # LOO：去掉自身
                    n0, n1 = n0 - (1 - int(l)), n1 - int(l)
                if n0 + n1 >= 3:
                    return (n1 + 0.5) / (n0 + n1 + 1.0)
            return float(ym.mean())

        oof = np.array([_lookup(s, l) for s, l in zip(tr_sizes, ym)])
        te = np.array([_lookup(s) for s in te_sizes])
        n_coarse = sum(1 for s in te_sizes if cnts[0].get(s, (0, 0))[0] + cnts[0].get(s, (0, 0))[1] < 3)
        print(f"[L1] sizeprior AUC={roc_auc_score(ym, oof):.4f} test_coarse_fallback={n_coarse}/{len(te_sizes)}")
        members.append(("sizeprior", oof, te))

    # 多生成器扩增头：[1000 训练图 + COCO真(0) + 各生成器假图(1)] 上训练检测头。
    # 覆盖 FLUX/SD14/Midjourney/wukong/glide/BigGAN/ADM（ADM 和 MJ 是 CF384 最弱切片）。
    # OOF：每个 fold 训练 = 训练折 + 全部 aug 样本（aug 永不进验证折，无泄漏）。
    for tag, feat_tag, size in (("cf384", "commfor_cf384", 384), ("clipH378", "clipH378", 378)):
        pair = (
            load_features(FEAT, feat_tag, size, "crop", "sample"),
            load_features(FEAT, feat_tag, size, "crop", "test"),
        )
        aug_parts = [
            (p.stem.replace(f"aug_{tag}_", ""), np.load(p).astype(np.float32))
            for p in sorted(FEAT.glob(f"aug_{tag}_*.npy"))
        ]
        aug_parts = [(n, X) for n, X in aug_parts if len(X) > 0]
        if pair[0] is None or pair[1] is None or len(aug_parts) < 2:
            continue
        import torch
        from sklearn.model_selection import StratifiedKFold as SKF
        from fake_image_detection.stacking import _predict_mlp, _train_mlp

        Xs, Xt = pair
        X_aug = np.concatenate([X for _, X in aug_parts], 0)
        y_aug = np.concatenate([
            np.zeros(len(X)) if n == "coco" else np.ones(len(X)) for n, X in aug_parts
        ])
        gen_names = [n for n, _ in aug_parts]

        def _norm(X):
            return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        Xsn, Xan = _norm(Xs), _norm(X_aug)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        skf = SKF(5, shuffle=True, random_state=42)
        oof = np.zeros(len(lab_y), dtype=np.float64)
        for tr, va in skf.split(Xsn, lab_y):
            Xtr = np.concatenate([Xsn[tr], Xan], 0)
            ytr = np.concatenate([lab_y[tr], y_aug], 0)
            head = _train_mlp(Xtr, ytr, device, seed=42)
            oof[va] = _predict_mlp(head, Xsn[va], device)
        te = fit_predict_mlp(
            np.concatenate([Xs, X_aug], 0), np.concatenate([lab_y, y_aug], 0), Xt
        )
        print(f"[L1] fluxaug_{tag} AUC={roc_auc_score(lab_y, oof):.4f} (MLP, +{len(y_aug)} aug from {gen_names})")
        members.append((f"fluxaug_{tag}", oof, te))

    # NPR 特征（像素域，低相关性）
    npr_s, npr_t = FEAT / "npr_256_sample.npy", FEAT / "npr_256_test.npy"
    if npr_s.exists() and npr_t.exists():
        Xs = np.load(npr_s).astype(np.float32)
        Xt = np.load(npr_t).astype(np.float32)
        oof = oof_lr(Xs, lab_y)
        te = fit_predict_lr(Xs, lab_y, Xt)
        print(f"[L1] npr_256 AUC={roc_auc_score(lab_y, oof):.4f}")
        members.append(("npr_256", oof, te))

    # SRM/Bayar 高通残差取证特征（RAPID 思路，ET 头优于 LR）
    srm_s, srm_t = FEAT / "srm_sample.npy", FEAT / "srm_test.npy"
    if srm_s.exists() and srm_t.exists():
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.model_selection import StratifiedKFold as SKF
        Xs = np.load(srm_s).astype(np.float32)
        Xt = np.load(srm_t).astype(np.float32)
        skf = SKF(5, shuffle=True, random_state=42)
        oof = np.zeros(len(lab_y), dtype=np.float64)
        for tr, va in skf.split(Xs, lab_y):
            clf = ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=2026)
            clf.fit(Xs[tr], lab_y[tr])
            oof[va] = clf.predict_proba(Xs[va])[:, 1]
        clf = ExtraTreesClassifier(n_estimators=800, n_jobs=-1, random_state=2026)
        clf.fit(Xs, lab_y)
        te = clf.predict_proba(Xt)[:, 1]
        print(f"[L1] srm AUC={roc_auc_score(lab_y, oof):.4f}")
        members.append(("srm", oof, te))

    return members


def load_external_predictions(test_ids):
    """加载已有的预测 CSV 作为融合成分。"""
    specs = [
        ("ft384", "commfor_ft384_tta.csv"),
        ("st_cf384_crop", "commfor_cf384_384_crop.csv"),
        ("st_cf384_resize", "commfor_cf384_384_resize.csv"),
        ("st_cf224_crop", "commfor_cf224_224_crop.csv"),
        ("st_round3", "st_round3.csv"),
        ("clip_st", "clip_clipL_336_336.csv"),
        ("clipH_st", "clipH_224_st.csv"),
        ("clipH_crop_st", "clipH_224_crop_st.csv"),
        ("clipH_ft", "clipH_224_ft_tta.csv"),
        ("clipH_ft0", "clipH_224_ft0_tta.csv"),
        ("clipBigG_st", "clipBigG_224_st.csv"),
        ("clipBigG_ft", "clipBigG_224_ft_tta.csv"),
        ("clipBigG_ft0", "clipBigG_224_ft0_tta.csv"),
        ("full_cnx", "full_convnext_large.csv"),
        ("univfd", "univfd_zeroshot.csv"),
        ("main970", "ensemble_ft_cf_main.csv"),
        ("clipH378_st", "clipH378_378_st.csv"),
        ("clipH378_224_st", "clipH378_378_st.csv"),
        ("cnx_xxl_st", "clip_cnx_xxl_st.csv"),
        ("cnx_xxl_st2", "cnx_xxl_st.csv"),
        ("cf_hard", "commfor_ft384_hard.csv"),
        ("bigg_hard", "clipBigG_224_ft_hard.csv"),
        ("meta_soft", "meta_fingerprint_oof.csv"),
    ]
    out = {}
    for name, rel in specs:
        p = PRED / rel
        if not p.exists():
            continue
        s = load_csv(p).reindex(test_ids).to_numpy(dtype=np.float64)
        if np.isnan(s).any():
            continue
        out[name] = s
        print(f"[csv] {name}")
    return out


def blend_recipes(stack_base, csv_test):
    """内置 prior378 等配方。"""
    def blend(parts):
        total = sum(parts.values())
        acc = None
        for name, w in parts.items():
            if name == "stack_base":
                r = rank01(stack_base)
            elif name in csv_test:
                r = rank01(csv_test[name])
            else:
                continue
            term = w / total * r
            acc = term if acc is None else acc + term
        return acc

    recipes = {
        "stack_only": {"stack_base": 1.0},
        "stack_ft": {"stack_base": 0.7, "ft384": 0.3},
        "stack_ft_st": {"stack_base": 0.6, "ft384": 0.22, "st_cf384_crop": 0.1, "clip_st": 0.08},
        "stack_main970": {"stack_base": 0.45, "main970": 0.4, "ft384": 0.15},
        "stack_clipH": {"stack_base": 0.4, "main970": 0.25, "ft384": 0.12, "clipH_st": 0.12, "clip_st": 0.06, "st_cf384_crop": 0.05},
        "stack_bigg": {"stack_base": 0.38, "main970": 0.22, "ft384": 0.12, "clipBigG_ft": 0.14, "clipH_ft": 0.08, "clipBigG_st": 0.06},
        "stack_diverse": {"stack_base": 0.4, "main970": 0.2, "ft384": 0.1, "clipBigG_ft": 0.1, "clipH_ft": 0.08, "clip_st": 0.06, "full_cnx": 0.06},
        "stack_push": {"stack_base": 0.42, "main970": 0.22, "ft384": 0.12, "clipBigG_ft": 0.1, "clipH_ft": 0.08, "clipH_st": 0.06},
        "stack_clean984": {"stack_base": 0.5, "main970": 0.22, "ft384": 0.12, "clipBigG_ft": 0.08, "clipH_st": 0.08},
        "stack_cf_anchor": {"stack_base": 0.35, "main970": 0.35, "ft384": 0.15, "st_cf384_crop": 0.08, "st_cf384_resize": 0.07},
        "stack_hard": {"stack_base": 0.48, "main970": 0.2, "ft384": 0.1, "cf_hard": 0.1, "bigg_hard": 0.07, "clipH_st": 0.05},
        "stack_prior378": {"stack_base": 0.45, "main970": 0.2, "ft384": 0.1, "clipBigG_ft": 0.08, "clipH_st": 0.07, "clipH378_st": 0.1},
        "stack_gate": {"stack_base": 0.55, "main970": 0.18, "ft384": 0.1, "clipBigG_ft": 0.07, "clipH_st": 0.05, "meta_soft": 0.05},
    }
    candidates = {}
    for name, parts in recipes.items():
        parts = {k: v for k, v in parts.items() if k == "stack_base" or k in csv_test}
        if not parts:
            continue
        candidates[name] = blend(parts)
        print(f"recipe {name} mean={candidates[name].mean():.4f}")
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", default=True)
    ap.add_argument("--override-csv", default="artifacts/label_override_sem.csv")
    args = ap.parse_args()

    lab = pd.read_csv(sample_csv())
    y = lab["label"].astype(int).to_numpy()
    test_ids = pd.read_csv(test_csv())["id"].tolist()

    members = build_members(y, test_ids, lab_df=lab)
    if not members:
        print("no features found; run extract_features.py first")
        return

    names = [m[0] for m in members]
    M_oof = np.stack([m[1] for m in members], axis=1)
    M_te = np.stack([m[2] for m in members], axis=1)
    print(f"L1 members ({len(names)}): {names}")

    # L1 rank weight optimization
    prior = np.ones(len(names))
    for i, n in enumerate(names):
        if "cf384" in n:
            prior[i] = 3.0
        elif n.startswith("cf_"):
            prior[i] = 2.0
        elif n.startswith("clip"):
            prior[i] = 1.8
        elif n == "meta":
            prior[i] = 1.4
        elif n.startswith("drct"):
            prior[i] = 0.7
    best_auc, best_w = optimize_rank_weights(M_oof, y, prior=prior)
    print(f"[L1-rank] OOF AUC={best_auc:.6f}")

    # L2 nested stacking
    stack_oof = nested_stack_oof(M_oof, y)
    for k, v in stack_oof.items():
        print(f"[L2-{k}] OOF AUC={roc_auc_score(y, v):.6f}")

    best2, w2 = blend_l2(stack_oof, M_oof @ best_w, y)
    print(f"[L2-blend] OOF AUC={best2:.6f} w={w2}")

    # Test prediction
    stack_te = fit_stack_predict(M_oof, y, M_te)
    T_rank = np.stack([rank01(M_te[:, j]) for j in range(M_te.shape[1])], axis=1)
    base_rank = T_rank @ best_w
    stack_parts = [rank01(stack_te["lr2"]), rank01(stack_te["et"]), rank01(stack_te["gb"]), rank01(base_rank)]
    stack_base = sum(wi * pi for wi, pi in zip(w2, stack_parts))

    # 同时保存 L1-rank 简单加权结果（无 L2 元学习器，可能对抗集泛化更好）
    pd.DataFrame({"id": test_ids, "score": base_rank}).to_csv(PRED / "ensemble_stack_l1rank.csv", index=False)
    print(f"[L1-rank] saved {PRED / 'ensemble_stack_l1rank.csv'} (simpler, no L2 meta-learner)")

    # External predictions and recipe blending
    csv_test = load_external_predictions(test_ids)
    candidates = blend_recipes(stack_base, csv_test)

    # Save predictions
    for name, score in candidates.items():
        out = PRED / f"ensemble_{name}.csv"
        pd.DataFrame({"id": test_ids, "score": score}).to_csv(out, index=False)
        print(f"saved {out}")

    # Primary
    primary = "stack_ft_st"
    for pref in ("stack_clean984", "stack_push", "stack_bigg", "stack_clipH", "stack_ft_st", "stack_prior378"):
        if pref in candidates:
            primary = pref
            break
    out = PRED / "ensemble_stack_primary.csv"
    pd.DataFrame({"id": test_ids, "score": candidates[primary]}).to_csv(out, index=False)
    print(f"PRIMARY {primary} -> {out}")

    # Submit
    if args.submit:
        ov = args.override_csv
        for name in ("stack_prior378", "stack_clean984", "stack_push", "stack_bigg", "stack_clipH", "stack_ft_st"):
            if name not in candidates:
                continue
            cmd = [
                sys.executable,
                "submit.py",
                "--predictions",
                str(PRED / f"ensemble_{name}.csv"),
                "--out",
                f"outputs/submissions/alt_{name}_submission.csv",
            ]
            if ov and (CR / ov).exists():
                cmd += ["--override-csv", ov]
            subprocess.run(cmd, cwd=str(CR), check=True)


if __name__ == "__main__":
    main()
