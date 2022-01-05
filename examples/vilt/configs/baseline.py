from colossalai.amp import AMP_TYPE


# ViT Base
BATCH_SIZE = 256
DROP_RATE = 0.1
NUM_EPOCHS = 10

fp16 = dict(
    mode=AMP_TYPE.TORCH,
)

gradient_accumulation = 16
gradient_clipping = 1.0

parallel = dict(
    data=dict(size=1),
    # tensor=dict(size=2, mode='1d'),
)
num_epochs = 10

# config logging path
logging = dict(
    root_path='./logs'
)
# dali = dict(
#     # root='./dataset/ILSVRC2012_1k',
#     root='/project/scratch/p200012/dataset/ILSVRC2012_1k',
#     gpu_aug=True,
#     mixup_alpha=0.2
# )
