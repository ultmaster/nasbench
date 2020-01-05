set -x
mkdir -p outputs
CUDA_VISIBLE_DEVICES=1 python test_arch38.py 2>&1 | tee outputs_arch38.log
