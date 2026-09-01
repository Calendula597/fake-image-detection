"""CO-SPY (CVPR2025) 特征提取：semantic (SigLIP-SO400M) + artifact (SD VAE recon) + fusion。

输出：
- outputs/features/cospySem_384_crop_{sample,test}.npy   (N, 1152) encode_image 特征
- outputs/features/cospyArt_224_crop_{sample,test}.npy   (N, 512)  artifact_encoder 特征
- outputs/features/cospy_{sem,art,fusion}_prob_{sample,test}.npy  (N,) 各分支 sigmoid 概率
- outputs/predictions/cospy_fusion.csv                    (id, score) 测试集 fusion 概率
并打印训练集上各特征/概率的 OOF / 直接 AUC。

权重：weights/cospy/sd-v1_4/sd-v1_4/{semantic,artifact,fusion}_weights.pth
注意：artifact_weights.pth 内含完整 SD v1-4 VAE 权重（diffusers 命名），无需再下载 VAE。
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # huggingface.co 被墙，走镜像

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from fake_image_detection.feature_extractor import ImageDataset
from fake_image_detection.paths import resolve_code_root, sample_csv, test_csv
from fake_image_detection.stacking import oof_lr

CR = resolve_code_root()
FEAT_DIR = CR / "outputs" / "features"
PRED_DIR = CR / "outputs" / "predictions"
WEIGHT_DIR = CR / "weights" / "cospy" / "sd-v1_4" / "sd-v1_4"


# ---------------------------------------------------------------------------
# Artifact 分支：VAEReconEncoder（拷贝自 external/co-spy/detectors/sd-v1_4/artifact.py，
# 保持键名一致以便 strict 加载 artifact_weights.pth）
# ---------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class VAEReconEncoder(nn.Module):
    def __init__(self, vae, block=Bottleneck):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.vae = vae

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.mean
            x_recon = self.vae.decode(latent).sample
        x = (x - x_recon) / 7.0 * 100.0
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        return self.avgpool(x).flatten(1)


def build_sd_vae():
    """实例化 SD v1-4 的 VAE 结构（权重来自 artifact_weights.pth，不联网）。"""
    from diffusers import AutoencoderKL

    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        latent_channels=4,
        norm_num_groups=32,
        sample_size=224,
        scaling_factor=0.18215,
    )


class ArtifactBranch(nn.Module):
    """与 co-spy ArtifactDetector 键名一致：artifact_encoder.* + fc.*"""

    def __init__(self):
        super().__init__()
        self.artifact_encoder = VAEReconEncoder(build_sd_vae())
        self.fc = nn.Linear(512, 1)


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------
def load_semantic(device):
    import open_clip

    clip, _, _ = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli")
    clip.requires_grad_(False)
    fc = nn.Linear(1152, 1)
    w = torch.load(WEIGHT_DIR / "semantic_weights.pth", map_location="cpu")
    fc.weight.data = w["fc.weight"]
    fc.bias.data = w["fc.bias"]
    return clip.to(device).eval(), fc.to(device).eval()


def load_artifact(device):
    model = ArtifactBranch()
    sd = torch.load(WEIGHT_DIR / "artifact_weights.pth", map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def load_fusion(device):
    fc = nn.Linear(2, 1)
    w = torch.load(WEIGHT_DIR / "fusion_weights.pth", map_location="cpu")
    fc.weight.data = w["fc.weight"]
    fc.bias.data = w["fc.bias"]
    return fc.to(device).eval()


def build_sem_transform():
    # co-spy SemanticDetector.test_transform: Resize(384) + CenterCrop(384), mean=std=0.5
    return transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_art_transform():
    # co-spy ArtifactDetector.test_transform: Resize(256) + CenterCrop(224), mean=0, std=1
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])


# ---------------------------------------------------------------------------
# 提取
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_sem(clip, fc, ids, rel_root, device, batch_size, num_workers):
    ds = ImageDataset(ids, rel_root, build_sem_transform())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats, logits = [], []
    for imgs, _ in tqdm(dl, desc=f"cospy-sem {rel_root}"):
        imgs = imgs.to(device, non_blocking=True)
        f = clip.encode_image(imgs)
        lg = fc(f).squeeze(-1)
        feats.append(f.float().cpu().numpy())
        logits.append(lg.float().cpu().numpy())
    return np.concatenate(feats, 0), np.concatenate(logits, 0)


@torch.no_grad()
def extract_art(model, ids, rel_root, device, batch_size, num_workers):
    ds = ImageDataset(ids, rel_root, build_art_transform())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats, logits = [], []
    for imgs, _ in tqdm(dl, desc=f"cospy-art {rel_root}"):
        imgs = imgs.to(device, non_blocking=True)
        f = model.artifact_encoder(imgs)
        lg = model.fc(f).squeeze(-1)
        feats.append(f.float().cpu().numpy())
        logits.append(lg.float().cpu().numpy())
    return np.concatenate(feats, 0), np.concatenate(logits, 0)


@torch.no_grad()
def fusion_prob(fusion_fc, sem_logit, art_logit):
    # fusion fc 的输入就是两个分支的原始 logit（见 co-spy fusion.py forward）
    x = torch.from_numpy(np.stack([sem_logit, art_logit], 1).astype(np.float32))
    x = x.to(next(fusion_fc.parameters()).device)
    return torch.sigmoid(fusion_fc(x)).squeeze(-1).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size-sem", type=int, default=32)
    ap.add_argument("--batch-size-art", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    splits = [("sample", pd.read_csv(sample_csv()), "data/image_sample_data")]
    if not args.skip_test:
        splits.append(("test", pd.read_csv(test_csv()), "data/image_test"))

    clip, sem_fc = load_semantic(device)
    art_model = load_artifact(device)
    fusion_fc = load_fusion(device)

    sample_probs = {}
    for split, df, rel in splits:
        ids = df["id"].tolist()

        sem_feat, sem_logit = extract_sem(clip, sem_fc, ids, rel, device, args.batch_size_sem, args.num_workers)
        np.save(FEAT_DIR / f"cospySem_384_crop_{split}.npy", sem_feat)
        np.save(FEAT_DIR / f"cospy_sem_prob_{split}.npy", 1 / (1 + np.exp(-sem_logit)))
        print(f"saved cospySem_384_crop_{split} {sem_feat.shape}")

        art_feat, art_logit = extract_art(art_model, ids, rel, device, args.batch_size_art, args.num_workers)
        np.save(FEAT_DIR / f"cospyArt_224_crop_{split}.npy", art_feat)
        np.save(FEAT_DIR / f"cospy_art_prob_{split}.npy", 1 / (1 + np.exp(-art_logit)))
        print(f"saved cospyArt_224_crop_{split} {art_feat.shape}")

        fus_prob = fusion_prob(fusion_fc, sem_logit, art_logit)
        np.save(FEAT_DIR / f"cospy_fusion_prob_{split}.npy", fus_prob)

        if split == "test":
            pd.DataFrame({"id": ids, "score": fus_prob}).to_csv(PRED_DIR / "cospy_fusion.csv", index=False)
            print(f"saved {PRED_DIR / 'cospy_fusion.csv'}")
        else:
            sample_probs = {
                "sem": 1 / (1 + np.exp(-sem_logit)),
                "art": 1 / (1 + np.exp(-art_logit)),
                "fusion": fus_prob,
            }

    # ---------------- OOF 评估 ----------------
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(sample_csv())
    y = df["label"].to_numpy()
    sem_feat = np.load(FEAT_DIR / "cospySem_384_crop_sample.npy").astype(np.float32)
    art_feat = np.load(FEAT_DIR / "cospyArt_224_crop_sample.npy").astype(np.float32)

    print("\n===== CO-SPY sample AUC =====")
    for name, p in sample_probs.items():
        print(f"{name}_prob direct AUC: {roc_auc_score(y, p):.4f}")
    for name, X in [("cospySem_384", sem_feat), ("cospyArt_224", art_feat),
                    ("cospySem+Art", np.concatenate([sem_feat, art_feat], 1))]:
        oof = oof_lr(X, y)
        print(f"{name} oof_lr AUC: {roc_auc_score(y, oof):.4f}")


if __name__ == "__main__":
    main()
