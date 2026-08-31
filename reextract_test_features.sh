#!/usr/bin/env bash
set -uo pipefail
cd /root/autodl-tmp/fake-image-detection
source .venv/bin/activate

echo "=== [1/5] CF384 crop ==="
python extract_features.py commfor --repo weights/commfor/commfor-model-384 --test-only --batch-size 64
echo "=== [2/5] CF384 resize ==="
python extract_features.py commfor --repo weights/commfor/commfor-model-384 --no-crop --test-only --batch-size 64
echo "=== [3/5] CF224 crop ==="
python extract_features.py commfor --repo weights/commfor/commfor-model-224 --test-only --batch-size 64
echo "=== [4/5] CF224 resize ==="
python extract_features.py commfor --repo weights/commfor/commfor-model-224 --no-crop --test-only --batch-size 64
echo "=== [5/5] clipH 224 ==="
python extract_features.py clip --arch ViT-H-14 --arch-dir clip-vit-h-14 --tag clipH --size 224 --test-only --batch-size 48
echo "=== clipBigG 224 ==="
python extract_features.py clip --arch ViT-bigG-14 --arch-dir clip-vit-bigg-14 --tag clipBigG --size 224 --test-only --batch-size 32
echo "=== clipH378 378 ==="
python extract_features.py clip --arch ViT-H-14-378-quickgelu --arch-dir clip-vit-h14-378 --tag clipH378 --size 378 --test-only --batch-size 16
echo "=== ALL DONE ==="
