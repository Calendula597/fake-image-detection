# Fake Image Detection — 最小可运行基线 + prior378 堆叠方案

从 `/root/autodl-tmp/Detector/image/` 抽取的精简版项目，包含两部分：

1. **ConvNeXt-Large 单模型基线**（训练 → TTA 推理 → 提交）
2. **prior378 最优方案**（多模型特征提取 → 两层 OOF 堆叠 → 配方融合 → 标签覆盖）

## 目录结构

```
fake-image-detection/
├── configs/
│   ├── baseline.yaml          # ConvNeXt 5-fold 训练配置
│   └── full_train.yaml        # ConvNeXt 全量训练配置
├── fake_image_detection/      # 核心包
│   ├── dataset.py             # Dataset / collate
│   ├── transforms.py          # 数据增强
│   ├── model.py               # ConvNeXt 模型
│   ├── cf_model.py            # Community Forensics 模型包装
│   ├── clip_model.py          # CLIP/OpenCLIP 模型包装
│   ├── feature_extractor.py   # CF/CLIP 特征提取
│   ├── stacking.py            # OOF 堆叠 / rank 优化 / L2 融合
│   ├── train_engine.py        # EMA / train_one_epoch / evaluate
│   ├── losses.py              # BCE / Focal
│   ├── metrics.py             # ROC-AUC 等
│   ├── optimizer.py           # AdamW + layer decay
│   ├── scheduler.py           # cosine + warmup
│   ├── paths.py               # 数据/输出路径解析
│   ├── config.py              # YAML 配置加载
│   ├── logger.py              # 日志
│   ├── io_utils.py            # 图片读取
│   ├── checkpoints.py         # 保存/读取 checkpoint
│   └── seed.py                # 随机种子
├── train.py                   # ConvNeXt 训练入口
├── predict.py                 # ConvNeXt 推理入口
├── submit.py                  # 生成提交 CSV
├── validate_submission.py     # 提交格式校验
├── extract_features.py        # CF/CLIP 特征提取
├── self_train.py              # 特征 + MLP 自训练
├── stack.py                   # OOF 堆叠 + prior378 配方融合
├── label_override.py          # 近重复标签覆盖
├── run_train_predict.sh       # ConvNeXt 一键脚本
└── tests/                     # 最小测试
```

## 环境

```bash
pip install -r requirements.txt
```

## 数据路径

默认自动寻找以下数据根目录（按优先级）：

1. `DATA_ROOT` 环境变量
2. `../dateset/sem_image`
3. `../dateset/image_data`
4. `../dataset/dataset/image_data`

数据根目录下需要包含：

- `image_sample_data.csv`（`id,label`）
- `image_test.csv`（`id`）
- `data/image_sample_data/*.jpg`
- `data/image_test/*.jpg`
- `image_submission_example.csv`（`id,score`）

## 用法

### 一、ConvNeXt 单模型基线

```bash
# 1. 单折训练
python train.py --config configs/baseline.yaml --fold 0 --output-tag baseline_fold0

# 2. 全量训练
python train.py --config configs/full_train.yaml --fold -1 --output-tag full_convnext_large

# 3. TTA 推理
python predict.py \
  --config configs/full_train.yaml \
  --checkpoint outputs/checkpoints/full_convnext_large/best.pt \
  --output outputs/predictions/full_convnext_large.csv \
  --tta

# 4. 生成提交
python submit.py --predictions outputs/predictions/full_convnext_large.csv

# 5. 校验
python validate_submission.py --submission outputs/submissions/submission.csv
```

### 二、prior378 最优方案

#### 1. 提取特征

```bash
# Community Forensics 特征（cf384 / cf224）
python extract_features.py commfor --repo weights/commfor/commfor-model-384
python extract_features.py commfor --repo weights/commfor/commfor-model-224

# CLIP 特征（clipH / clipBigG / clipH378）
python extract_features.py clip --arch ViT-H-14 --arch-dir clip-vit-h-14 --tag clipH --size 224
python extract_features.py clip --arch ViT-bigG-14 --arch-dir clip-vit-bigg-14 --tag clipBigG --size 224
python extract_features.py clip --arch ViT-H-14-378-quickgelu --arch-dir clip-vit-h14-378 --tag clipH378 --size 378
```

#### 2. 自训练（可选，提升各特征）

```bash
python self_train.py --feature commfor_cf384 --size 384 --mode crop --rounds 4
python self_train.py --feature clipH --size 224 --mode crop --rounds 4
python self_train.py --feature clipBigG --size 224 --mode crop --rounds 4
python self_train.py --feature clipH378 --size 378 --mode crop --rounds 4
```

#### 3. OOF 堆叠 + 配方融合

```bash
python stack.py
```

输出：
- `outputs/predictions/ensemble_stack_*.csv`：各配方预测
- `outputs/predictions/ensemble_stack_primary.csv`：主预测
- `outputs/submissions/alt_stack_*_submission.csv`：各配方提交文件

#### 4. 近重复标签覆盖（可选）

```bash
# 需要先运行原项目的 audit_dataset.py 生成 artifacts/data_manifest.csv
python label_override.py --manifest artifacts/data_manifest.csv
```

#### 5. 提交

```bash
python submit.py --predictions outputs/predictions/ensemble_stack_prior378.csv
python validate_submission.py --submission outputs/submissions/submission.csv
```

## prior378 配方

`stack.py` 内置了原项目最优配方：

| 成分 | 权重 | 说明 |
|---|---|---|
| `stack_base` | 0.45 | 14 组特征两层堆叠 |
| `main970` | 0.20 | CF 主模型微调 |
| `ft384` | 0.10 | CF384 端到端微调 + TTA |
| `clipBigG_ft` | 0.08 | CLIP-bigG 微调 |
| `clipH_st` | 0.07 | CLIP-H 自训练 |
| `clipH378_st` | 0.10 | CLIP-H-378 自训练 |

历史成绩：LB AUC **0.986774**。

## 输出

- checkpoint: `outputs/checkpoints/<tag>/best.pt`
- OOF 预测: `outputs/oof/<tag>.csv`
- 特征: `outputs/features/*.npy`
- 测试预测: `outputs/predictions/<tag>.csv`
- 提交文件: `outputs/submissions/submission.csv`

## 说明

- 正类 `label=1` 表示 **AI 生成图片**。
- 提交文件 `score` 为 AI 生成的置信度，范围 `[0, 1]`。
- 原项目中的 Organika、UnivFD、DRCT 等实验性模型未完全移植，可按需添加。

## 复赛（sem_image）新增管线（2026-09）

> 详细实验记录见 `HANDOFF.md`。以下是复赛改进新增的可复现管线。

### 多生成器扩增头（当前主力，stack_aug4）

核心发现：冻结骨干 + 比赛 1000 张训练的头对**未见生成器**（ADM/Midjourney/VQDM）存在盲区。
修复：用公开生成器假图 + COCO 真图扩增头部训练数据。

```bash
# 1. FLUX 假图生成（流匹配，GGUF Q8，~16s/张，断点续跑）
python generate_flux.py --n 600 --out data_real/coco_val_ai_flux

# 2. 提取扩增池特征（COCO 真 + 各生成器假 + 退化变体）
python extract_aug.py --backbones cf384 clipH378 clipBigG dinoL --all-genimage
python extract_aug.py --backbones cf384 clipH378 --deg        # data_real_deg/ 退化变体

# 3. 堆叠（自动加载 aug_*_*.npy 训练 fluxaug_* 成员）
python stack.py
python submit.py --predictions outputs/predictions/ensemble_stack_only.csv \
    --override-csv artifacts/label_override_sem.csv --out outputs/submissions/stack_aug4_sem.csv
```

扩增池（`data_real/`，label=1 除 coco 外）：GenImage（midjourney/wukong/glide/biggan/adm/vqdm 各 500，
来自 `bitmind/GenImage_*`）、本地 FLUX.1-schnell 生成、SD1.4 img2img；`data_real_deg/` 为退化变体
（JPEG q30-75 + 缩放回采 + 模糊）。`holdout_*` 目录是留出验证集，**绝不进训练**。

### 其他新成员

- `sizeprior`：尺寸查表先验（生成器原生尺寸偏 AI），`stack.py` 内置，OOF 0.81
- `srm`：SRM/Bayar 高通残差取证特征，`python extract_srm.py`
- 自建测试集评估：`python eval_selftest.py --backbone cf384`（COCO 真 vs 各生成器假）

注意：`ImageDataset` 对打不开的图片会**静默回退黑图**——评估脚本务必传绝对路径。
