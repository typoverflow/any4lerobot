export CUDA_VISIBLE_DEVICES=""

python openx_rlds.py \
    --raw-dir /localscratch/cgao304/dev/datasets/rlds/bridge_orig/0.0.1 \
    --local-dir /localscratch/cgao304/dev/datasets/lerobot_v3/bridge_orig/ \
    --use-videos \
    --num-proc 8 \
    --image-writer-process 1 \
    --repo-id typoverflow/bridge_orig \
    --push-to-hub