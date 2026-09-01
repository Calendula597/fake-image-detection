# 复赛改进 — 交接文档（更新至 2026-09-01）

> 给明天的自己 / 新会话：本文档包含恢复进度所需的全部状态。
> 恢复对话：`cd /root/autodl-tmp && kimi --continue`，然后 `/goal resume`。

## 任务

琶洲算法大赛 · AI 生成图像检测（**复赛**，二分类，指标 AUC，复赛测试集含对抗样本）。
目标：复赛 LB 越高越好（用户期望 0.99，但实测该对抗集极难）。每次改动都 git commit。
评判（claim.png）：复赛 `Score = e^(4·AUC_text) + e^(4·AUC_image)`，指数加权，我只做 image。

## 数据（关键！）

- 训练集：`/root/autodl-tmp/Detector/dateset/sem_image/data/image_sample_data`（1000 张，500真/500AI）
- 测试集：`/root/autodl-tmp/Detector/dateset/sem_image/data/image_test`（20000 张，复赛对抗性测试集）
- 提交格式：`id,score`，20000 行，score∈[0,1]，1=AI生成。**每天 3 次提交，手动上传**
- 网络：huggingface.co 被墙 → 用 `export HF_ENDPOINT=https://hf-mirror.com`；github 超时 → `https://gh-proxy.com/https://github.com`
- **数据对齐 bug（已修复）**：原项目特征基于初赛 image_data，比赛用复赛 sem_image，文件名同但图不同。所有特征必须在 sem_image 上重新提取。

## LB 成绩（复赛，按时间）

| 日期 | 提交 | 构成 | LB |
|---|---|---|---|
| 08-31 | prior378(错测试集) | — | 0.501237 |
| 08-31 | stack_base(11成员) | 冻结堆叠 | 0.891552 |
| 08-31 | blend_v2(base+ft+自训练) | 融合 | 0.889155 |
| 09-01 | stack_base19(+dinov3) | 冻结堆叠 | 0.902021 |
| 09-01 | stack_base21(+CO-SPY) | 冻结堆叠 | **0.902824** ← 当前最好 |
| 09-01 | ft_ens2(微调集成) | 端到端微调 | 0.846965 |

## 当前最好提交文件（明天用）

`outputs/submissions/stack_base26_sem.csv` — 26 成员堆叠（OOF 0.991），加了 FSD+AIDE。未提交过。
备选：`outputs/submissions/stack_base21_sem.csv`（LB 实测 0.902824）。

## 已证伪/确认的方向（重要教训，别再重复）

- ❌ **端到端微调（finetune）**：ft_ens2 LB 0.846 < 冻结 0.902，OOF→LB 差距更大（-0.131 vs -0.088）。微调让模型过拟合训练分布，对抗泛化更差。**已证伪，别再试**。微调 checkpoint 已删。
- ❌ **dinov3-H+（最大版）**：OOF 0.9393，仍弱于 clipH378（0.9688），边际递减。
- ❌ **RIGID/WePe/WaRPAD（扰动一致性信号）**：都 ~0.58-0.61，此数据集无效。
- ❌ **近重复标签覆盖**：只有 9 个可靠，且高相似样本 stack 本来就分对，覆盖价值小。
- ✅ **CLIP(clipH378) 是最强冻结骨干**（0.9688），比所有 dinov3 强。
- ✅ **"加新算法族"有效，"加同类特征"封顶**：11→19(加dinov3新族)+0.0105，19→21(加CO-SPY)只+0.0008。
- ⚠️ **SOTA 论文方法在此对抗集上都变弱**：FSD(0.777)、AIDE(0.652)、NPR(0.68)、cospyArt(0.69)，都不如 CF/CLIP。论文在其它 benchmark 的 0.96 不迁移。

## 当前 pipeline（已验证正确）

代码 `/root/autodl-tmp/fake-image-detection/`，GitHub `git@github.com:Calendula597/fake-image-detection.git`。

- **特征**（26 个 .npy，在 sem_image 上提取）：CF(cf384/cf224)、CLIP(clipH/bigG/H378/L)、DINOv3(B/L/H)、DRCT、CO-SPY(SigLIP+VAE)、NPR、FSD、AIDE、meta指纹
- **堆叠** `stack.py`：CF/CLIP/DINO/CO-SPY/FSD/AIDE 用 MLP 头，DRCT/meta/NPR 用 LR/ExtraTrees；L1 单特征→L2 元学习器→rank 加权
- **标签覆盖**：`artifacts/label_override_sem.csv`（9 个）
- **提交**：`python submit.py --predictions X --override-csv artifacts/label_override_sem.csv`，`validate_submission.py` 校验
- 环境：`source .venv/bin/activate`，GPU=RTX4090，权重软链接 `weights -> Detector/image/weights`

## 各特征提取脚本

`extract_features.py`(CF/CLIP)、`extract_dino.py`(dinov3)、`extract_drct.py`、`extract_cospy.py`、`extract_npr.py`、`extract_fsd.py`、`extract_aide.py`、`recompute_metadata.py`(meta指纹)、`recompute_override.py`(近重复)。`external/` 下有 co-spy/npr/fsd/aide/rine 仓库。

## 磁盘

曾到 92%，清理废弃微调 checkpoint 后到 80%（~10G 空闲）。dinov3-7B(28G) 不可行。
**别再下载超大模型**。要删可删：DRCT 权重(2.3G,特征弱)。

## 明天方向（诚实评估）

纯堆特征已接近上限（~0.90-0.91）。0.99 在此对抗集用现有公开方法大概率不可达。可试：
1. 提交 stack_base26（FSD/AIDE 是不同机制，可能小幅提升）
2. 若还要涨，只能找机制更独特的信号（频域/谱域/VLM推理），但收益预期递减
3. 考虑接受 ~0.91 作为实际可达目标
