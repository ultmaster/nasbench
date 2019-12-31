set -x
CUDA_VISIBLE_DEVICES=0 python test_arch38.py --intermediate_evaluations 0.0093 0.0185 0.25 0.5 0.75 0.9 | tee outputs/arch38.log
