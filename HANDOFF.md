# 复赛改进 — 交接文档（2026-08-31）

> 给明天的自己 / 新会话：本文档包含恢复进度所需的全部状态。
> 恢复对话：`cd /root/autodl-tmp && kimi --continue`，然后 `/goal resume`。

## 任务

琶洲算法大赛 · AI 生成图像检测（**复赛**，二分类，指标 AUC）。
目标：在复赛测试集上做出比当前最好更高的 LB 分数。每次改动都 git commit。

## 数据（关键！）

- 训练集：`/root/autodl-tmp/Detector/dateset/sem_image/data/image_sample_data`（1000 张，500真/500AI）
- 测试集：`/root/autodl-tmp/Detector/dateset/sem_image/data/image_test`（20000 张，**复赛对抗性测试集**）
- 提交格式：`id,score`，20000 行，score∈[0,1]，1=AI生成
- **每天只有 3 次提交机会**，手动上传到比赛平台

### 数据对齐 bug（已修复，重要教训）

- 原项目特征/预测基于 `dateset/image_data`（**初赛**测试集）
- 比赛实际用 `dateset/sem_image`（**复赛**测试集）
- 两者文件名相同（00000.jpg~19999.jpg）但图片完全不同 → 第一次提交只得 0.501237（随机）
- 修复：在 sem_image 上重新提取全部特征（`reextract_test_features.sh`）
- 训练特征有效（训练集未变），测试特征必须重提取
- 错误的旧特征/预测已隔离到 `outputs/features/_wrong_testset/` 和 `outputs/predictions/_wrong_testset/`

## 当前成绩（复赛 LB）

| 提交 | 构成 | LB |
|---|---|---|
| prior378（错测试集） | — | 0.501237 |
| stack_base（11 成员） | 纯堆叠 | **0.891552** ← 当前最好 |
| blend_v2（15成员base + ft384 + 自训练） | 融合 | 0.889155 |

- **stack_base 是目前最好的**，融合 ft384+自训练反而略降
- OOF AUC 0.989 → LB 0.891，差距大 = 复赛是对抗性/分布偏移测试集
- 自训练在对抗集上无效甚至有害（伪标签"自信地错"）

## 当前 pipeline（已验证正确）

代码：`/root/autodl-tmp/fake-image-detection/`，GitHub: `git@github.com:Calendula597/fake-image-detection.git`

1. **特征提取**（`extract_features.py`，GPU）：CF384/CF224 + clipH/clipBigG/clipH378，crop+resize
   - 权重软链接 `weights -> /root/autodl-tmp/Detector/image/weights`
2. **堆叠**（`stack.py`，CPU）：15 成员（CF×4 + clipH×2 + clipBigG×2 + clipH378×2 + DRCT×4 + meta）
   - CF/CLIP 用 MLP 头（比 LR 强，clipH378 提升最大），DRCT/meta 用 LR/ExtraTrees
   - L1 单特征 → L2 元学习器（LR/ET/GB）→ rank 加权
   - metadata 指纹：`recompute_metadata.py` → `artifacts/metadata_fingerprints_sem.csv`
3. **标签覆盖**：`recompute_override.py` → `artifacts/label_override_sem.csv`（9 个近重复，全 label=1）
4. **提交**：`submit.py --predictions X --override-csv artifacts/label_override_sem.csv`，`validate_submission.py` 校验

环境：`source .venv/bin/activate`（venv 在项目内，GPU=RTX4090）

## 当前最好提交文件

`outputs/submissions/stack_base_sem.csv`（LB 0.891552）—— 但注意这是 11 成员版本。
最新 15 成员 stack_base（OOF 0.990422）尚未单独提交过，只以 blend_v2 形式提交过（0.889155）。

## 明天的改进方向（针对对抗性）

核心：让 stack_base 成员**对对抗扰动更鲁棒**，而非加更多会过拟合的预测。

1. **JPEG 重压缩防御（首选，原项目没用）**：对抗样本依赖微妙像素扰动，JPEG 重压缩/缩放可摧毁。
   - 做法：对测试图 JPEG 重压缩后再提取特征，作为新 stack 成员
   - 也可对训练图做同样处理保持一致
2. **多尺度/更高分辨率特征**（512）
3. **L2 元学习器正则化**（防过拟合训练分布）
4. 单独提交 15 成员 stack_base（OOF 0.990422），看是否超过 0.891552

## 实验记录

见 `results.tsv`。git log 有完整改动历史。

## 远程机器

原远程机器（region-42.seetacloud.com:31041）已关闭，权重和 commfor_ft384 checkpoint 已拉到本地。
