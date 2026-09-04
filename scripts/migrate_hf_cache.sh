#!/bin/bash
# 逐目录合并迁移 HF 缓存到数据盘（校验大小一致后删源）
set -u
rm -f /root/.cache/huggingface/hf_home
for d in /root/.cache/huggingface/hub/*/; do
  name=$(basename "$d")
  cp -an "$d" /root/autodl-tmp/hf_home/hub/ 2>/dev/null
  s=$(du -sb "$d" | cut -f1)
  t=$(du -sb "/root/autodl-tmp/hf_home/hub/$name" 2>/dev/null | cut -f1 || echo 0)
  if [ "$s" = "$t" ]; then
    rm -r "$d"
    echo "OK $name"
  else
    echo "MISMATCH $name src=$s dst=$t"
  fi
done
# 顶层零散文件
cp -an /root/.cache/huggingface/* /root/autodl-tmp/hf_home/ 2>/dev/null || true
rmdir /root/.cache/huggingface/hub /root/.cache/huggingface/datasets /root/.cache/huggingface 2>/dev/null
if [ ! -e /root/.cache/huggingface ]; then
  ln -s /root/autodl-tmp/hf_home /root/.cache/huggingface
  echo "SYMLINKED"
else
  echo "LEFTOVER:"; ls -la /root/.cache/huggingface/
fi
df -h / /root/autodl-tmp | tail -2
