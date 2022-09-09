export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=4321 \
    mabe/train.py \
    -opt options/video/seed_0.yml
