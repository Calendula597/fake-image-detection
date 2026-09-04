#!/bin/bash
# 统一 HF 缓存布局：已迁移项通过符号链接挂回 ~/.cache/huggingface/hub
set -u
SRC=/root/autodl-tmp/hf_home/hub
DST=/root/.cache/huggingface/hub
mkdir -p "$DST"
for d in "$SRC"/*/; do
  name=$(basename "$d")
  target="$DST/$name"
  if [ -L "$target" ]; then
    echo "SKIP(link) $name"
  elif [ -e "$target" ]; then
    # 系统盘已有完整副本（两个大 dinov3 模型），或重复项：字节级校验后删系统盘副本换链接
    if diff <(cd "$target" && find . -type f -exec du -b {} \; | sort) <(cd "$d" && find . -type f -exec du -b {} \; | sort) >/dev/null 2>&1; then
      rm -r "$target"
      ln -s "$d" "$target"
      echo "REPLACED $name (identical, system copy removed)"
    else
      echo "KEEP-REAL $name (differs, real dir stays on system disk)"
    fi
  else
    ln -s "$d" "$target"
    echo "LINKED $name"
  fi
done
# 顶层文件
for f in "$SRC"/*; do
  [ -f "$f" ] || continue
  name=$(basename "$f")
  [ -e "/root/.cache/huggingface/$name" ] || ln -s "$f" "/root/.cache/huggingface/$name"
done
echo "---"
df -h / /root/autodl-tmp | tail -2
ls /root/.cache/huggingface/hub | wc -l
