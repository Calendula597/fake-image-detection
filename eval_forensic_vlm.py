# -*- coding: utf-8 -*-
"""评测专业取证 VLM（SIDA-7B, CVPR2025）在样本集上的 OOF AUC。

模型: saberzl/SIDA-7B (https://huggingface.co/saberzl/SIDA-7B)
  - 基于 LISA-7B-v1 (LLaVA-1.5 Vicuna-7B + SAM) 微调，SID_Set 训练
    (real / full synthetic / tampered 三分类)。
  - 分类方式: 在回复开头生成 [CLS] token，取其最后一层 hidden state
    过 cls_head (Linear 4096->2048->3) 得三分类 logits。
  - 本脚本 teacher-force [CLS] token，单次前向得到 logits，
    AI 分数 = P(full synthetic) + P(tampered) = 1 - P(real)。

首选方案 AntifakePrompt 的软提示 checkpoint 仅存于 Google Drive，
本机网络无法访问（drive.google.com 连接超时），HF/ModelScope 无镜像，
故改用 SIDA-7B（权重在 HF，可经 hf-mirror 下载，约 16.2GB）。

运行环境: .venv-sida（transformers==4.37.2 + 系统 torch 2.12），
依赖 external/SIDA 官方代码（仅导入，不修改）。
用法:
  python eval_forensic_vlm.py [--limit N] [--out outputs/features/fvlm_sample.npy]
"""
from __future__ import annotations

import argparse
import importlib.machinery
import os
import sys
import time
import types
from pathlib import Path

CR = Path(__file__).resolve().parent
SIDA_REPO = CR / "external" / "SIDA"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(CR / "materials" / "hf_cache"))


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


# SIDA 官方代码顶层 import 了训练期依赖，推理用不到，打桩跳过
_stub("deepspeed")
_stub("torchviz", make_dot=lambda *a, **k: None)


class _Dummy:
    pass


# llava_mpt 依赖 transformers 4.31 的 bloom 内部函数，4.37 已移除；推理用不到 MPT
_stub("model.llava.model.language_model.llava_mpt",
      LlavaMPTConfig=_Dummy, LlavaMPTForCausalLM=_Dummy)

import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import AutoConfig, AutoModelForCausalLM

# transformers 4.37 已内置 llava，与 LLaVA 1.0 旧代码的注册冲突，改为静默跳过；
# config 由我们显式用其 LlavaConfig 加载
_ac_register = AutoConfig.register
AutoConfig.register = classmethod(
    lambda cls, model_type, config, exist_ok=True: _ac_register(model_type, config, exist_ok=True))
_am_register = AutoModelForCausalLM.register
AutoModelForCausalLM.register = classmethod(
    lambda cls, config, model, exist_ok=True: _am_register(config, model, exist_ok=True))

sys.path.insert(0, str(SIDA_REPO))

from model.SIDA import SIDAForCausalLM  # noqa: E402
from model.llava import conversation as conversation_lib  # noqa: E402
from model.llava.mm_utils import tokenizer_image_token  # noqa: E402
from model.llava.model.language_model.llava_llama import (  # noqa: E402
    LlavaConfig, LlavaLlamaForCausalLM)
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,  # noqa: E402
                         DEFAULT_IMAGE_TOKEN)

MODEL_ID = "saberzl/SIDA-7B"
# 训练时使用的分类提问（external/SIDA/utils/SID_Set.py）
QUESTION = ("Can you identify if this image is real, full synthetic, or tampered "
            "image? Please mask the tampered regions if it is tampered.")


def build_prompt() -> str:
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + QUESTION
    replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（调试用）")
    ap.add_argument("--out", default=str(CR / "outputs" / "features" / "fvlm_sample.npy"))
    ap.add_argument("--csv", default="/root/autodl-tmp/Detector/dateset/sem_image/image_sample_data.csv")
    ap.add_argument("--img-dir", default="/root/autodl-tmp/Detector/dateset/sem_image/data/image_sample_data")
    ap.add_argument("--sanity-gen", type=int, default=3, help="前 N 张额外做生成 sanity check")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, CLIPImageProcessor

    device = "cuda:0"
    dtype = torch.bfloat16

    model_path = snapshot_download(MODEL_ID)
    print(f"model path: {model_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=512, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    print(f"[CLS] id={cls_token_idx}, [SEG] id={seg_token_idx}", flush=True)

    cfg = LlavaConfig.from_pretrained(model_path)
    t0 = time.time()
    model = SIDAForCausalLM.from_pretrained(
        model_path, config=cfg, torch_dtype=dtype, low_cpu_mem_usage=True,
        vision_tower="openai/clip-vit-large-patch14",
        seg_token_idx=seg_token_idx, cls_token_idx=cls_token_idx,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=device, dtype=dtype)
    model.eval()
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

    clip_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    prompt_text = build_prompt()
    print(f"prompt: {prompt_text!r}", flush=True)

    df = pd.read_csv(args.csv)
    ids = df["id"].tolist()
    labels = df["label"].to_numpy()
    if args.limit > 0:
        ids = ids[: args.limit]
        labels = labels[: args.limit]
    img_dir = Path(args.img_dir)

    base_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt")
    base_ids = torch.cat([base_ids, torch.tensor([cls_token_idx])]).unsqueeze(0).to(device)
    attn = torch.ones_like(base_ids)

    scores = np.zeros(len(ids), dtype=np.float64)
    logits_all = np.zeros((len(ids), 3), dtype=np.float32)
    t_start = time.time()
    with torch.no_grad():
        for i, sid in enumerate(ids):
            t_img = time.time()
            try:
                img = Image.open(img_dir / sid).convert("RGB")
                px = clip_processor.preprocess(np.asarray(img), return_tensors="pt")["pixel_values"]
            except Exception as e:
                print(f"[warn] {sid}: {e}", flush=True)
                px = torch.zeros(1, 3, 224, 224)
            px = px.to(device=device, dtype=dtype)
            # 直接调用基类 forward，绕过 SIDA 的训练向 model_forward 路由
            out = LlavaLlamaForCausalLM.forward(
                model, input_ids=base_ids, attention_mask=attn, images=px,
                output_hidden_states=True, return_dict=True)
            hs = out.hidden_states[-1]
            cls_hidden = hs.reshape(-1, hs.shape[-1])[-1]  # [CLS] 位置的最后一层 hidden
            head = model.model.cls_head[0]
            logits3 = head(cls_hidden.to(head[0].weight.dtype)).float()  # [real, full_syn, tampered]
            logp = torch.log_softmax(logits3, dim=-1)
            ai_score = 1.0 - logp[0].exp().item()
            scores[i] = ai_score
            logits_all[i] = logits3.cpu().numpy()

            if i < args.sanity_gen:
                # 用 [CLS] 前一位置的 LM logits 验证模型确实会以 [CLS] 开头回复
                next_logits = out.logits[0, -2]
                top = next_logits.argmax().item()
                top_tok = tokenizer.decode([top])
                print(f"[sanity] {sid} label={labels[i]} score={ai_score:.4f} "
                      f"logits3={[round(x,3) for x in logits3.tolist()]} "
                      f"next_token={top_tok!r}(id={top}, expect [CLS] id={cls_token_idx})", flush=True)
            if (i + 1) % 50 == 0:
                el = time.time() - t_start
                print(f"{i+1}/{len(ids)} {el/(i+1):.2f}s/img", flush=True)

    total = time.time() - t_start
    per_img = total / len(ids)
    print(f"done: {len(ids)} imgs, {total:.0f}s total, {per_img:.2f}s/img", flush=True)
    print(f"est. 20000 test imgs: {per_img*20000/3600:.2f} h", flush=True)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, scores)
    print(f"OOF AUC (sample {len(ids)} imgs): {auc:.4f}", flush=True)
    print(f"Qwen3.5-0.8B zero-shot baseline: 0.7632", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, scores.astype(np.float32))
    np.save(out_path.with_name(out_path.stem + "_logits3.npy"), logits_all)
    print(f"scores saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
