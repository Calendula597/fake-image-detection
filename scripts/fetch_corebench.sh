#!/bin/bash
# 串行下载 T2I-CoReBench 各模型 tar.gz → 每个抽前 400 张 → 删压缩包
# 临时文件放系统盘 /tmp（24G 空闲），图片落数据盘
set -u
BASE="https://hf-mirror.com/datasets/lioooox/T2I-CoReBench-Images/resolve/main"
DEST_ROOT=/root/autodl-tmp/fake-image-detection/data_real
MODELS="Seedream-3 Nano-Banana imagen-4 HunyuanImage-3.0 GPT-Image-1.5 Z-Image LongCat-Image HiDream-I1 SD-3.5-Large Qwen-Image"
for m in $MODELS; do
  out="$DEST_ROOT/corebench_$(echo "$m" | tr '.-' '__')"
  n=$(find "$out" -type f 2>/dev/null | wc -l)
  if [ "$n" -ge 380 ]; then echo "SKIP $m ($n imgs)"; continue; fi
  echo "=== $m ==="
  rm -f "/tmp/cb_$m.tar.gz"
  aria2c -x 8 -s 8 --connect-timeout=15 --timeout=60 --max-tries=5 --console-log-level=error \
    -d /tmp -o "cb_$m.tar.gz" "$BASE/$m.tar.gz" || { echo "DL-FAIL $m"; continue; }
  mkdir -p "$out"
  # 流式抽前 400 个图片文件
  tar -tzf "/tmp/cb_$m.tar.gz" 2>/dev/null | grep -iE '\.(jpg|jpeg|png)$' | head -400 > "/tmp/cb_list_$m.txt"
  tar -xzf "/tmp/cb_$m.tar.gz" -C "$out" -T "/tmp/cb_list_$m.txt" 2>/dev/null
  rm -f "/tmp/cb_$m.tar.gz" "/tmp/cb_list_$m.txt"
  n=$(find "$out" -type f | wc -l)
  echo "DONE $m -> $n imgs"
  df -h /tmp /root/autodl-tmp | tail -2
done
echo ALL_DONE
