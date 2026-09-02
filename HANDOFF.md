# 复赛改进 — 交接文档（更新至 2026-09-02）

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
| 09-01 | stack_base21(+CO-SPY) | 冻结堆叠 | 0.902824 |
| 09-01 | ft_ens2(微调集成) | 端到端微调 | 0.846965 |
| 09-01 | stack_base27(+Qwen/FSD/AIDE) | 冻结堆叠 | **0.910985** ← 当前最好 |
| 09-01 | stack_deg(退化匹配) | 退化匹配训练 | 0.910284 |
| 09-01 | stack_base29_rank(+bfree) | 冻结堆叠 | 0.905813 |

## 09-02 新增候选（未上 LB）

- **`stack_base31_sem.csv`（首选）** — 30 成员 + `sizeprior`（尺寸先验分层回退版：精确尺寸→q8→宽高比×面积档×mult8桶；LOO OOF 0.809，测试集 81% 精确尺寸覆盖）
- `stack_base30_sem.csv` — sizeprior 初版（未见尺寸回退 0.5），已被 base31 取代
- `blend27_degrank_sem.csv` — 0.6×base27 + 0.4×deg_rank 的 rank 融合
- `stack_base28_sem.csv` — 只 +DIFT 的对照

## 09-02 尺寸先验发现（重要新信号）

- 训练集 尺寸→AI率 强相关：512²=0.96、1024²=0.85、256×144=1.00（AI）；1024×683=0.02、341×512=0.00（真实相机尺寸）
- LOO 尺寸查表单特征 OOF AUC=**0.8127**，强于整个 meta 成员（0.70），与 meta 相关性仅 0.45
- 测试集 81% 图片尺寸在训练表中；未见尺寸用分层回退
- 量化表指纹查表已证伪（AUC 0.41，赛方统一重编码，只有 14 种表）

## 09-02 赛事情报（subagent 调研确认）

- 比赛 = **第五届琶洲算法大赛"湾区杯"AI生成内容鉴别挑战赛**，task2 图片检测，出题方 = **腾讯朱雀实验室**。复赛进行中，无公开题解。
- 官方说明测试集覆盖"扩散模型、GAN、**流匹配**"等主流架构 + 压缩/裁剪/滤镜/社交平台转发对抗。
- 复赛假图大概率含商业生成器：MJ/SD/DALL·E/FLUX + 国产（**混元、即梦/Seedream、可灵/Kling、Qwen-Image/万相**，腾讯系可能偏多）。
- NTIRE 2026 报告（arXiv:2604.11487）是最近似参考：real 来自 CC12M/CommonPool/RedCaps，fake 用 VLM caption→LLM 改写 prompt 配对生成，**对齐分辨率/宽高比/JPEG 质量分布**，测试保留更新生成器（Qwen-Image/HiDream/Nano Banana 等）。
- 未试过的高价值方向（按优先级）：
  1. **商业 API 生成数据**（Seedream/Kling/混元/Nano Banana）— 需 API key，问用户
  2. **本地跑 FLUX.1-schnell（流匹配，官方点名）生成新假图**：扩增训练池 + 建更准的自建测试集 ← 当前进行中（GGUF Q8 下载）
  3. **DDA 数据对齐**（NeurIPS'25 腾讯优图，github.com/roy-ch/Dual-Data-Alignment）：VAE 重建真实图成内容匹配假图 + 重加 JPEG 压缩对齐频谱 + mixup（与 B-Free 不同在 JPEG 对齐）
  4. 成对干净/退化训练（TeleAI LPT）+ 赛事匹配的退化（**moiré、color cast、speckle noise**）
  5. SRM/Bayar 高通残差分支（RAPID）— 与 NPR 不同的取证线索，便宜可加
  6. logit 空间门控级联融合（INTSIG）— 推理期技巧
- ⚠️ 朱雀检测器本身公开可用，可作为特征，但它是出题方产品，涉嫌违反诚信参赛，**不要用**除非用户明确批准

## 当前最好提交文件（明天用）

- **首选 `outputs/submissions/stack_deg_sem.csv`** — 退化匹配训练版（28 成员，强骨干测试头用干净+退化训练）。**用于检验 NTIRE 退化匹配结论**。未提交过。
- **对照 `outputs/submissions/stack_deg_rank_sem.csv`** — 同上但 L1-rank 堆叠（无干净训练的 L2）。
- 备选：`stack_base28_sem.csv`（+DIFT 扩散侧）、`stack_base27_sem.csv`（LB 实测 0.910985）。

## 已证伪/确认的方向（重要教训，别再重复）

- ⭐ **退化匹配训练（最重要突破，待 LB 验证）**：NTIRE 2026 调研发现 OOF→LB 鸿沟的根因是"**干净特征训练 vs 退化测试**"。修复 = 对训练图施加与测试类似的退化（JPEG/缩放/模糊/噪声），重新提取特征，**让堆叠头在 [干净+退化] 特征上训练**。已实现：`extract_degraded.py` 提取退化特征（CF384/CF224/clipH/clipBigG/clipH378/dinoL/dinoB），`stack.py` 的 `_fit_mlp_deg` 让强成员测试头用干净+退化训练。**候选：`stack_deg_sem.csv`(L2) / `stack_deg_rank_sem.csv`(L1-rank)**。**我此前"微调证伪"结论可能错了——不是微调不行，是训练数据/增强没对齐对抗分布。**
- ❌ **端到端微调（finetune）**：ft_ens2 LB 0.846 < 冻结 0.902。但注意：NTIRE 前两名都是微调大 backbone 且成功，区别在退化增强对齐。微调 checkpoint 已删。
- ❌ **dinov3-H+（最大版）**：OOF 0.9393，仍弱于 clipH378（0.9688），边际递减。
- ❌ **RIGID/WePe/WaRPAD（扰动一致性信号）**：都 ~0.58-0.61，此数据集无效。
- ❌ **近重复标签覆盖**：只有 9 个可靠，且高相似样本 stack 本来就分对，覆盖价值小。
- ✅ **CLIP(clipH378) 是最强冻结骨干**（0.9688），比所有 dinov3 强。
- ✅ **"加新算法族"有效，"加同类特征"封顶**：11→19(加dinov3新族)+0.0105，19→21(加CO-SPY)只+0.0008，21→27(加Qwen/FSD/AIDE)+0.0082。
- ✅ **Qwen VLM 是最有效的非视觉新机制**（OOF 0.76，但 LB 证明有效）。**VLM 侧已探明：Qwen-0.8B 就是最优，专业取证 VLM（SIDA-7B 0.74、AntifakePrompt 拿不到、其它太大）都更差或不可行，别再试 VLM 了**。
- ⚠️ **SOTA 论文方法在此对抗集上都变弱**：FSD(0.777)、AIDE(0.652)、NPR(0.68)、cospyArt(0.69)，都不如 CF/CLIP。论文在其它 benchmark 的 0.96 不迁移。**OOF 不预测 LB，一切以 LB 实测为准**。
- ⚠️ **扩散侧"重建/噪声"路线弱**（SD 在 LAION 训练，真实图也在分布内）：单步噪声误差~0.50、cospyArt(VAE)0.69。**DIFT(SD UNet 特征)=0.7442 是扩散侧最好信号**（把扩散模型当 backbone）。已入 28 成员堆叠，待 LB 检验。

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
1. **先提交 stack_base27**（OOF 最高 0.992084，Qwen 是不同机制，可能小幅提升到 ~0.905-0.91）
2. Qwen VLM（语义推理）是唯一在本数据集上给堆叠带来 OOF 提升的新机制（+0.001），可考虑更强 VLM 或更好 prompt
3. 若还要涨，只能找机制更独特的信号（频域/谱域），但收益预期递减
4. 理性目标 ~0.91-0.92，0.99 大概率不可达

## 09-03 重要更正：FLUX 致盲是假象（评估 bug）

- 09-02 晚报告"CF384/clipH378 对 FLUX AUC≈0.5 致盲"是**错的**：我的临时评估脚本给 `ImageDataset` 传相对路径，它对打不开的图片**静默回退黑图**（feature_extractor.py:38-39），评估的全是空白图。
- 干净评估（eval_selftest.py，绝对路径）：CF384 对 FLUX=**0.9987**、SD14=1.0000；clipH378 对 FLUX=0.8275、SD14=0.7954。**骨干对开源新生代检测良好**。
- 策略修正：LB 0.91 的差距大概率来自**商业闭源生成器**（Midjourney/Seedream/混元/DALL·E），本地无法生成 → **商业 API 数据是唯一未证伪的大杠杆**（等用户答复 API 渠道）。
- 教训：任何"异常低"的评估结果先检查输入是否被静默替换；`ImageDataset` 的静默黑图回退是个坑。

## 09-03 生成器难度地图（比赛训练头，COCO 为真）

| 生成器 | CF384 | clipH378 |
|---|---|---|
| ADM (guided-diffusion) | **0.635** | 0.868 |
| Midjourney (GenImage v5) | **0.861** | 0.978 |
| glide | 0.933 | 0.958 |
| FLUX schnell | 0.962 | 0.952 |
| BigGAN | 0.968 | 0.995 |
| wukong | 0.995 | 0.994 |
| SD1.4 img2img | 0.995 | 0.801 |

- **ADM 和 Midjourney 是 CF384 的最弱切片**（mean_p=0.20/0.57，被判为真）；wukong/BigGAN 已基本解决。
- 候选 `stack_aug1_sem.csv`：33 成员，新增 fluxaug_cf384/clipH378（3899 张多生成器 aug 训练头，OOF 0.9425/0.9497）。
- 数据：`data_real/genimage_{midjourney,wukong,glide,biggan,adm}` 各 500（bitmind/GenImage_*），`coco_val_ai_flux`（本地 FLUX schnell 生成中）。

## 09-03 留出集验证（aug 机制实证有效）

- 留出集 = GenImage streaming skip 前 500 后的新图（MJ/ADM 各 300，未参与训练）
- cf384: MJ 0.9847→**0.9980**, ADM 0.8442→**0.9974**；clipH378: MJ 0.9537→**0.9999**, ADM 0.8475→**1.0000**
- 结论：多生成器 aug 头确实补上了已证实的盲区，stack_aug2 值得优先提交。剩余风险仅在于比赛测试集的生成器构成。
- 另：MLP 头（归一化）比 LR 头（原始特征）跨生成器泛化显著更好（ADM 无 aug：MLP 0.844 vs LR 0.635）。

## 09-03 退化扩增（aug3）与验证

- 退化变体：`data_real_deg/`（JPEG q30-75 + 0.4-0.75 下采上采 + 30% 高斯模糊），训练池每个 split 一份；`extract_aug.py --deg` 提特征；**holdout_* 目录已排除出训练**（--all-genimage 和 --deg 都排除）
- 退化留出集验证（图未参与训练）：cf384 退化MJ 0.9799→**0.9941**、退化ADM 0.9305→**0.9970**；clipH378 0.9880→**1.0000**、0.9758→**0.9998**
- 候选 `stack_aug3_sem.csv`：35 成员，aug 头训练样本 7867+（含退化）。**优先于 aug2**
- 事故记录：两次特征提取抢显存把 FLUX 生成挤到 OOM；FLUX 生成需要独占 ~13GB，并发 GPU 任务务必 bs 小或串行
