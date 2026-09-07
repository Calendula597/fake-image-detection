# 复赛改进 — 交接文档（更新至 2026-09-07，全量重写版）

> 给明天的自己 / 新会话：本文档包含恢复进度所需的全部状态。
> 恢复对话：`cd /root/autodl-tmp && kimi --continue`，然后 `/goal resume`。

## 一、任务与赛事

- **赛事**：第五届琶洲算法大赛"湾区杯"AI生成内容鉴别挑战赛（2026），task2 图片检测，出题方 = **腾讯朱雀实验室**（subagent 调研确认，官方页 aicompetition-pz.com/topic_detail/41）。复赛进行中，无公开题解。
- **指标**：ROC-AUC（复赛总分 = e^(4·AUC_text) + e^(4·AUC_image)，我只做 image）。
- **测试集特点**（官方说明 + 实测推断）：覆盖扩散模型/GAN/流匹配；含压缩、裁剪、滤镜、**社交平台多层转发**对抗；假图大概率含商业生成器（混元/即梦 Seedream/可灵/MJ/DALL·E/FLUX 系）。
- **提交**：`id,score` 20000 行，score∈[0,1]，1=AI。**每天仅 3 次机会，只能用户手动上传**。

## 二、环境与数据路径

- 代码根：`/root/autodl-tmp/fake-image-detection/`（venv：先 `source .venv/bin/activate`）
- 训练集：`/root/autodl-tmp/Detector/dateset/sem_image/data/image_sample_data`（1000 张 500真/500AI）
- 测试集：`/root/autodl-tmp/Detector/dateset/sem_image/data/image_test`（20000 张无标签）
- **网络**：huggingface.co 被墙 → 必须 `export HF_ENDPOINT=https://hf-mirror.com`；github 走 `https://gh-proxy.com/https://github.com`；**Google Drive 不可达**（SPAI/Effort/RINE 等权重死在此）；ModelScope 无检测器权重。大文件用 `aria2c -x 8`（hf-mirror 单流会被限到 60KB/s，多连接恢复 5-8MB/s）。xet 后端偶发 401 → `export HF_HUB_DISABLE_XET=1`。
- **磁盘**（易踩坑）：系统盘 `/` 仅 30G，数据盘 `/dev/md0` **实为 50G**（不是 70G）。HF 缓存已迁到数据盘 `/root/autodl-tmp/hf_home`，`~/.cache/huggingface` 逐条目软链（脚本 `scripts/migrate_hf_cache.sh`、`link_hf_cache.sh`）。下载临时文件放系统盘 /tmp。两个大 dinov3 模型（huge/large）因历史原因以实体目录留在系统盘，路径照常可用。
- git 仓库：`git@github.com:Calendula597/fake-image-detection.git`，**每次改动必须 commit+push**；大目录（outputs/weights/artifacts/data_real*/materials/）已在 .gitignore；add 时用具体文件名，别 `git add -A`。

## 三、LB 成绩史（复赛，按时间）

| 日期 | 提交 | LB | 备注 |
|---|---|---|---|
| 08-31 | prior378（错测试集） | 0.501237 | 原项目特征基于初赛集，教训：全部特征须在 sem_image 重提取 |
| 08-31 | stack_base(11成员) | 0.891552 | |
| 08-31 | blend_v2（自训练） | 0.889155 | 伪标签在对抗集"自信地错" |
| 09-01 | stack_base19(+dinov3) | 0.902021 | 加新算法族有效 |
| 09-01 | stack_base21(+CO-SPY) | 0.902824 | 加同类特征封顶 |
| 09-01 | ft_ens2（端到端微调） | 0.846965 | 微调过拟合，证伪 |
| 09-01 | stack_base27(+Qwen/FSD/AIDE) | 0.910985 | |
| 09-02 | stack_deg（退化匹配） | 0.910284 | NTIRE 结论不迁移 |
| 09-02 | stack_base29_rank(+bfree) | 0.905813 | B-Free 被 SD v1-4 单一来源封顶 |
| 09-03 | stack_base32(+sizeprior+SRM) | 0.912225 | 尺寸先验+残差取证有效 |
| 09-03 | stack_aug9（多生成器扩增） | 0.914094 | 扩增路线成立 |
| 09-04 | stack_r1ps（初赛软标签20k） | 0.908262 | 软标签方向证伪（两剂量均负） |
| 09-04 | blend_aug9_base32 | 0.91385 | 融合稀释 |
| 09-04 | **stack_aug13h1（aug9+混元）** | **0.916647** | **当前最好。混元（腾讯系）扩增是获胜单机制** |
| 09-04 | stack_aug10t1（混元+TTA） | 0.916478 | TTA 无增益 |
| 09-05 | stack_aug14h（混元×3） | 0.916228 | 混元轴见顶于 500 张 |
| 09-05 | stack_aug12t（稳态化全包） | 0.913184 | 种子平均+TTA 无增益 |
| 09-05 | stack_aug15k(+Kolors) | 0.91429 | 代理覆盖饱和 |
| 09-05 | blend_13h1_10t1 | 0.916623 | 持平 |
| 09-07 | stack_aug17d（多轮退化全包） | 0.912255 | **多轮假设部分证伪：重度退化训练有害** |
| 09-07 | stack_aug18x（鲁棒成员版） | 0.915349 | 仍低于 aug13h1，但 > aug17d |
| 待测 | **stack_aug19c（CoReBench 商业模型入池）** | — | **下一个首选** |

## 四、当前最好与候选队列

- **最好**：`outputs/submissions/stack_aug13h1_sem.csv` = **0.916647**
- **下一批建议顺序**：① `stack_aug19c_sem.csv`（aug13h1 配方 + 12 个 CoReBench 商业模型入池）② `stack_aug13h1_sem.csv`（保底复测）③ 视 ① 结果定（CoDE 评估 / 新机制）
- 提交前问我，我按当日候选期望值重新排序，避免重复提交浪费名额。

## 五、核心管线（可复现命令）

```bash
source .venv/bin/activate && export HF_ENDPOINT=https://hf-mirror.com

# 主堆叠（35 成员：CF/CLIP/DINOv3/CO-SPY/FSD/AIDE/DIFT/Qwen/bfree/DRCT/meta/sizeprior/npr/srm + 4个fluxaug扩增头）
python stack.py                    # 环境变量：FLUXAUG_SEEDS=n、MLP_SEEDS=n、NO_TTA=1、EXCLUDE_PLAIN_STRONG=1

# 生成提交（9 个近重复标签覆盖）
python submit.py --predictions outputs/predictions/ensemble_stack_only.csv \
  --override-csv artifacts/label_override_sem.csv --out outputs/submissions/XXX.csv
python validate_submission.py --submission outputs/submissions/XXX.csv
```

扩增数据管线：
- `generate_flux.py`（FLUX.1-schnell GGUF Q8，COCO caption，已产 600 张入池；**权重已删**，需要时按脚本头部说明重下 ~16GB）
- `generate_hydit.py`（HunyuanDiT-v1.2，已产 1500 张，40% 中文 prompt；权重已删）
- `generate_kolors.py`（Kolors，已产 500 张；权重已删）
- `generate_deg2.py`（多轮退化变体，aug17d 用，现不推荐使用）
- `extract_aug.py`（扩增池特征，--all-genimage / --deg / --deg2；**holdout_* 目录永不入池**）
- `extract_flip.py`（TTA 翻转测试特征）、`extract_srm.py`（SRM 残差）、`extract_dinomac.py`（DINO-MAC 特征，边际）
- `scripts/fetch_corebench.sh`（CoReBench tar.gz 队列下载+抽 400 张；corebench 目录已扁平化、嵌套 PNG 已删）
- `eval_selftest.py`（COCO 真实 vs 各生成器假的骨干评估；**评估脚本必须传绝对路径**——`ImageDataset` 对打不开的图静默回退黑图，曾造成"FLUX 致盲"误判）

## 六、关键认知（重要，别再重复踩坑）

### 已验证有效（按 LB 贡献排序）
1. **多生成器扩增池**（11→19→27→35 成员演进中的最大单机制）：fluxaug_* 头 = [1000 训练图 + COCO/ImageNet 真 + 13+12 个生成器假 + 温和退化变体] 训练 Muon-MLP。**混元（腾讯系）是最有效的单个生成器**（+0.0026）
2. **尺寸先验 sizeprior**（生成器原生尺寸 512²/1024² 偏 AI，LOO 查表 OOF 0.81，测试集 81% 覆盖）
3. **SRM/Bayar 残差**（弱 0.66 但零相关，多样性好）
4. **L2 堆叠 > L1-rank 融合 > 手工 rank 融合**（融合两个强提交会稀释）
5. **冻结骨干：CF384、clipH378 最强**（OOF 0.96/0.9688）；MLP 头 > LR 头（跨生成器）

### 已证伪/封顶（别再试）
- 端到端微调（0.846）、自训练/伪标签（对抗集+初赛软标签均负）、B-Free（SD 单一来源封顶）、退化匹配训练、**重度多轮退化训练（deg2/deg3/deg4，LB -0.004~-0.001，比赛退化没有模拟的那么重）**
- 种子平均（-0.0007）、TTA 翻转（+0）、L1-rank、SAFE checkpoint（自测 0.26-0.59）、RPTC/PatchCraft（0.63-0.98 太弱）、DINO-MAC（0.9129 < 原 dinoL 0.9204）、取证 VLM（SIDA-7B < Qwen）、RIGID/WePe/一致性信号（~0.58）、纯 FFT（~0.60）、dinov3-H+（< clipH378）
- 混元 1500 张 ≈ 500 张（见顶）；Kolors 无增益（代理饱和）

### 侦查结论（09-07 定型）
- **30 个生成器家族全部本地 0.95+**（含 Seedream-3、HunyuanImage-3.0、GPT-Image、Nano Banana、imagen-4、FLUX.1/2-dev、SD3.5L、Qwen-Image、MJ v5/v5.1、wukong、ADM、glide、BigGAN、VQDM 等）——**盲区不在生成器身份**
- 比赛训练集真假内容相似度 cos>0.8 仅 1 对 → 非内容匹配构造
- 已证实的小损失机制：真实侧误判（ImageNet 真实图 13-15% FP）、内容依赖型头部（clipH378 在 img2img 上 0.80，cf384 型不受影响）、多轮退化（仅对 1000 张训练的主成员有崩坏，fluxaug 头天然鲁棒 0.9989）
- 唯一无法验证的：MJ v7 / Firefly v4 / Seedream-4（21.5GB 超磁盘）本地不可得；比赛真实图的具体来源分布

## 七、扩增池清单（data_real/，label=1 除标注外）

- 真实侧（label=0）：`coco_val`(600)、`imagenet_real`(400)
- 开源生成器：`coco_val_ai_sd14`(600)、`coco_val_ai_flux`(600)、`sdxl_10k`(500)、`hydit`(1500)、`kolors`(500)、`genimage_{midjourney,wukong,glide,biggan,adm,vqdm}`(各500)、`journeydb`(300)
- CoReBench 商业/新模型（各 400，扁平 jpg）：`corebench_{Seedream_3,Nano_Banana,imagen_4,HunyuanImage_3_0,GPT_Image_1_5,FLUX_1_dev,FLUX_2_dev,FLUX_1_Krea_dev,SD_3_5_Large,HiDream_I1,Z_Image,Qwen_Image(69),LongCat_Image}`
- 退化变体：`data_real_deg/`（单轮温和，aug13h1 配方）、`data_real_deg2/`（多轮，已证伪不推荐）
- 验证集（永不入池）：`data_val/`（cm_multideg 多轮退化测试集等）、`data_real/holdout_{midjourney,adm}`、`low_conf_1000/`（模型最拿不准的 1000 张测试图+清单）

## 八、未完成的候选方向（按优先级）

1. **提交 stack_aug19c**（已生成、校验通过）
2. **CoDE 评估**（aimagelab/CoDE，HF 非 gated，DINOv2+对比对齐，sklearn 头在 9.2M 图 D³ 上训练，<2h）
3. sidbench 剩余检测器快测（NPR/GramNet/FreqDetect，`/tmp/sidbench` + dkarageo/sidbench；RPTC 已证伪）
4. A*STAR 2026.01 思路自实现：真实图过生成器末端组件（VAE 解码器等）造假图微调 DINOv3（约 1 天）
5. 真实侧继续扩（Unsplash/LAION 下载曾失败，可用 bhargavsdesai/laion_improved_aesthetics 重试）
6. TeleGuard 式成对干净/退化特征对齐损失（NTIRE 季军，~3h）

## 九、恢复检查单

1. `cd /root/autodl-tmp && kimi --continue` → `/goal resume`
2. 确认磁盘：`df -h / /root/autodl-tmp`（系统盘 >3G、数据盘 >5G 空闲才安全）
3. 跑 stack 前：`source .venv/bin/activate && export HF_ENDPOINT=https://hf-mirror.com`
4. 当日提交前问我排序；提交后把分数告诉我（格式：`文件名:分数`）
