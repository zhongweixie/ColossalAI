try:
    from colossalai.zero.legacy.shard_utils import TensorShardStrategy
except ImportError:
    # colossalai > 0.2.8
    from colossalai.zero.legacy import TensorShardStrategy

zero = dict(model_config=dict(shard_strategy=TensorShardStrategy(),
                              tensor_placement_policy="auto",
                              reuse_fp16_shard=True),
            optimizer_config=dict(gpu_margin_mem_ratio=0.8, initial_scale=16384))
