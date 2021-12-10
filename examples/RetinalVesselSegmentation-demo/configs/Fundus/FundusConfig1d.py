# compulsory
# build optimizer from this dictionary
optimizer = dict(
    # Avaluable types: 'ZeroRedundancyOptimizer_Level_1', 'ZeroRedundancyOptimizer_Level_2', 'ZeroRedundancyOptimizer_Level_3'
    # 'Adam', 'Lamb', 'SGD', 'FusedLAMB', 'FusedAdam', 'FusedSGD', 'FP16Optimizer'
    type='Adam',
    lr=0.001,
    weight_decay=0
)

# compulsory
# build loss function from this dictionary
# loss = dict(
#     # Avaluable types:
#     # 'CrossEntropyLoss2D', 'CrossEntropyLoss2p5D', 'CrossEntropyLoss3D'
#     type='CrossEntropyLoss2D',
# )

# compulsory
# build model from this dictionary
# model = dict(
#     # types avaluable: 'PretrainBERT', 'VanillaResNet', 'VisionTransformerFromConfig'
#     type='VisionTransformerFromConfig',
#     # each key-value pair above refers to a layer
#     # input data pass through these layers recursively
#     tensor_splitting_cfg=dict(
#         type='ViTInputSplitter2D',
#     ),
#     embedding_cfg=dict(
#         type='ViTPatchEmbedding2D',
#         img_size=IMG_SIZE,
#         patch_size=PATCH_SIZE,
#         embed_dim=DIM,
#     ),
#     token_fusion_cfg=dict(
#         type='ViTTokenFuser2D',
#         img_size=IMG_SIZE,
#         patch_size=PATCH_SIZE,
#         embed_dim=DIM,
#         drop_rate=0.1
#     ),
#     norm_cfg=dict(
#         type='LayerNorm2D',
#         normalized_shape=DIM,
#         eps=1e-6,
#     ),
#     block_cfg=dict(
#         # ViTBlock is a submodule
#         type='ViTBlock',
#         attention_cfg=dict(
#             type='ViTSelfAttention2D',
#             hidden_size=DIM,
#             num_attention_heads=NUM_ATTENTION_HEADS,
#             attention_dropout_prob=0.,
#             hidden_dropout_prob=0.1,
#             checkpoint=True
#         ),
#         droppath_cfg=dict(
#             type='VanillaViTDropPath',
#         ),
#         mlp_cfg=dict(
#             type='ViTMLP2D',
#             in_features=DIM,
#             dropout_prob=0.1,
#             mlp_ratio=4,
#             checkpoint=True
#         ),
#         norm_cfg=dict(
#             type='LayerNorm2D',
#             normalized_shape=DIM,
#             eps=1e-6,
#         ),
#     ),
#     head_cfg=dict(
#         type='ViTHead2D',
#         hidden_size=DIM,
#         num_classes=NUM_CLASSES,
#     ),
#     embed_dim=DIM,
#     depth=DEPTH,
#     drop_path_rate=0.,
# )

# hooks are built when initializing trainer
# possible hooks: 'BaseHook', 'MetricHook','LoadCheckpointHook'
# 'SaveCheckpointHook','LossHook', 'AccuracyHook', 'Accuracy2DHook'
# 'LogMetricByEpochHook', 'TensorboardHook','LogTimingByEpochHook', 'LogMemoryByEpochHook' 
hooks = [
    dict(type='LogMetricByEpochHook'),
    dict(type='LogTimingByEpochHook'),
    dict(type='LogMemoryByEpochHook'),
    dict(type='Accuracy2DHook'),
    dict(type='LossHook'),
    # dict(type='TensorboardHook', log_dir='./tfb_logs'),
    # dict(type='SaveCheckpointHook', interval=5, checkpoint_dir='./ckpt'),
    # dict(type='LoadCheckpointHook', epoch=20, checkpoint_dir='./ckpt')
]

# three keys: pipeline, tensor, data
# if data=dict(size=1), which means no data parallelization, then there is no need to define it
parallel = dict(
    pipeline=dict(size=1),
    data=dict(size=1),
    tensor=dict(size=4, mode='2d'),
)

# not compulsory
# pipeline or no pipeline schedule
# fp16 = dict(
#     mode=AMP_TYPE.PARALLEL,
#     initial_scale=2 ** 8
# )

# not compulsory
# build learning rate scheduler
# lr_scheduler = dict(
#     type='LinearWarmupLR',
#     warmup_epochs=5
# )

schedule = dict(
    num_microbatches=8
)

# training stopping criterion
# you can give num_steps or num_epochs
num_epochs = 60

# config logging path
logging = dict(
    root_path='./logs'
)
