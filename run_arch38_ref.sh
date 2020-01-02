set -x
mkdir -p outputs
CUDA_VISIBLE_DEVICES=0 python test_arch38.py | tee outputs/arch38.log
