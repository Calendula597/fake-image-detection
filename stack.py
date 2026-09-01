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
            te = fit_predict_mlp(Xs, lab_y, Xt)
            print(f"[L1] CF {tag}_{crop} AUC={roc_auc_score(lab_y, oof):.4f} (MLP)")
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
                te = fit_predict_mlp(Xs, lab_y, Xt)
                print(f"[L1] {tag}_{size}_{crop} AUC={roc_auc_score(lab_y, oof):.4f} (MLP)")
                members.append((f"{tag}_{size}_{crop}", oof, te))

    # DINOv3 (dinoL / dinoB)
    for tag in ("dinoL", "dinoB"):
        for crop in ("crop", "resize"):
            pair = (
                load_features(FEAT, tag, 224, crop, "sample"),
                load_features(FEAT, tag, 224, crop, "test"),
            )
            if pair[0] is None or pair[1] is None:
                continue
            Xs, Xt = pair
            oof = oof_mlp(Xs, lab_y)
            te = fit_predict_mlp(Xs, lab_y, Xt)
            print(f"[L1] {tag}_224_{crop} AUC={roc_auc_score(lab_y, oof):.4f} (MLP)")
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
