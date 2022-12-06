import torch
from torch import nn

from colossalai.pipeline.rpc._pipeline_schedule import OneFOneBPipelineEngine
from colossalai.fx.passes.adding_split_node_pass import split_with_split_nodes_pass, balanced_split_pass
from colossalai.fx import ColoTracer
from colossalai.pipeline.middleware.adaptor import get_fx_topology
from rpc_test_utils import rpc_run, parse_args, MLP, DAG_MLP
from functools import partial

# global variable for model created
batch_size = 16
dim = 10

def create_partition_module(pp_rank: int, stage_num: int, model, data_kwargs):
    model.eval()
    tracer = ColoTracer()
    meta_args = {k: v.to('meta') for k, v in data_kwargs.items()}
    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
    annotated_model = balanced_split_pass(gm, stage_num)
    top_module, split_submodules = split_with_split_nodes_pass(annotated_model, merge_output=True)
    topo = get_fx_topology(top_module)
    for submodule in split_submodules:
        if isinstance(submodule, torch.fx.GraphModule):
            setattr(submodule, '_topo', topo)
    return split_submodules[pp_rank+1]

def partition(model, data_kwargs: dict, pp_rank: int, chunk: int, stage_num: int):
    torch.manual_seed(1024)
    partition = create_partition_module(pp_rank, stage_num, model, data_kwargs)
    return partition

def run_master(args):
    torch.manual_seed(100)

    epoch = args.epoch
    device = args.device
    stage_num = args.world_size
    chunk = args.chunk
    num_microbatches = args.num_microbatches
    use_checkpoint = args.use_checkpoint

    def data_gen(model: str):
        if model == 'MLP':
            x = torch.zeros((batch_size, dim))
            kwargs = dict(x=x)
            return kwargs
        elif model == 'DAG_MLP':
            x = torch.zeros((batch_size, dim))
            y = torch.zeros((batch_size, dim))
            kwargs = dict(x=x, y=y)
            return kwargs
        else:
            pass
    
    data_kwargs = [
        data_gen('MLP'),
        data_gen('DAG_MLP'),
    ]
    model = [
        MLP(dim, stage_num * 3),
        DAG_MLP(dim, stage_num * 3),
    ]

    i = 1
    engine = OneFOneBPipelineEngine(partition_fn=partial(partition, model[i], data_kwargs[i]),
                                    stage_num=stage_num,
                                    num_microbatches=num_microbatches,
                                    device=device,
                                    chunk=chunk,
                                    checkpoint=use_checkpoint)

    for _ in range(epoch):
        input_x = torch.randn((batch_size, dim), device=device)
        input_y = torch.randn((batch_size, dim), device=device)
        logits = engine.forward_backward({'x': input_x, 'y': input_y}, forward_only=True)

if __name__ == "__main__":
    args = parse_args()
    args.epoch = 10
    args.world_size = 4
    args.num_microbatches = 8
    rpc_run(args, run_master)