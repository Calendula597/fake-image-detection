#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$SCRIPT_DIR"
cd "$CODE_ROOT"

CONFIG="${CONFIG:-configs/full_train.yaml}"
TAG="${TAG:-full_convnext_large}"
EPOCHS="${EPOCHS:-15}"

echo "================ STEP 1: full-data training (all labeled, fold -1) ================"
python train.py \
  --config "$CONFIG" \
  --fold -1 \
  --epochs "$EPOCHS" \
  --output-tag "$TAG"

echo "================ STEP 2: TTA inference on test images ================"
rm -f "outputs/predictions/${TAG}.csv"
python predict.py \
  --config "$CONFIG" \
  --checkpoint "outputs/checkpoints/${TAG}/best.pt" \
  --output "outputs/predictions/${TAG}.csv" \
  --tta

echo "================ STEP 3: build submission ================"
python submit.py --predictions "outputs/predictions/${TAG}.csv" --out "outputs/submissions/submission.csv"

echo "================ STEP 4: validate ================"
python validate_submission.py --submission "outputs/submissions/submission.csv"

echo ""
echo "========================================================"
echo "DONE. Submission saved at:"
echo "  ${CODE_ROOT}/outputs/submissions/submission.csv"
echo "========================================================"
