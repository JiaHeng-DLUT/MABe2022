export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0 \
    python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=4321 \
    mabe/train.py \
    -opt options/keypoint/seed_0.yml
