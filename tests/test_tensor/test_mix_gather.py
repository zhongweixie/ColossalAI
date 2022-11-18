from functools import partial

import torch
import torch.multiprocessing as mp

from colossalai.core import global_context as gpc
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.shape_consistency import CollectiveCommPattern, CommSpec
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.tensor.utils import mix_gather_simulator
from colossalai.utils import free_port


def check_mix_gather_S0S1(device_mesh, rank):
    tensor_to_check = torch.arange(64).reshape((8, 8)).cuda()
    (f, b) = (0, 1)
    f_target_pair = (f, [0])
    b_target_pair = (b, [1])
    gather_dim, logical_process_axes = mix_gather_simulator(f_target_pair, b_target_pair)
    tensor_slice = [4, 2]    # (4, 2)
    rank_slice = 4
    f_start = (rank // rank_slice) * tensor_slice[0]
    b_start = (rank % rank_slice) * tensor_slice[1]
    tensor_to_comm = tensor_to_check[f_start:f_start + tensor_slice[0],
                                     b_start:b_start + tensor_slice[1]].contiguous().cuda()

    dim_partition_dict = {0: [0], 1: [1]}

    # DistSpec:
    #     shard_sequence: S0,S1
    #     device_mesh_shape: (2, 4)
    source_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    comm_spec = CommSpec(CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD,
                         sharding_spec=source_spec,
                         gather_dim=gather_dim,
                         logical_process_axis=logical_process_axes,
                         forward_only=True,
                         mix_gather=True)
    tensor_to_comm = tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_mix_gather_S1S0(device_mesh, rank):
    tensor_to_check = torch.arange(64).reshape((8, 8)).cuda()
    (f, b) = (0, 1)
    f_target_pair = (f, [1])
    b_target_pair = (b, [0])
    gather_dim, logical_process_axes = mix_gather_simulator(f_target_pair, b_target_pair)
    tensor_slice = [2, 4]
    rank_slice = 4
    f_start = (rank % rank_slice) * tensor_slice[0]
    b_start = (rank // rank_slice) * tensor_slice[1]
    tensor_to_comm = tensor_to_check[f_start:f_start + tensor_slice[0],
                                     b_start:b_start + tensor_slice[1]].contiguous().cuda()

    dim_partition_dict = {0: [1], 1: [0]}

    # DistSpec:
    #     shard_sequence: S1,S0
    #     device_mesh_shape: (2, 4)
    source_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    comm_spec = CommSpec(CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD,
                         sharding_spec=source_spec,
                         gather_dim=gather_dim,
                         logical_process_axis=logical_process_axes,
                         forward_only=True,
                         mix_gather=True)
    tensor_to_comm = tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    if rank == 0:
        print(tensor_to_comm)


def check_mix_gather_S01R(device_mesh, rank):
    pass


def check_mix_gather_RS01(device_mesh, rank):
    pass


def check_comm(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    physical_mesh_id = torch.arange(0, 8)
    assert rank == gpc.get_global_rank()

    mesh_shape = (2, 4)
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True, need_flatten=True)

    check_mix_gather_S0S1(device_mesh, rank)

    #check_mix_gather_S1S0(device_mesh, rank)

    #check_mix_gather_S01R(device_mesh, rank)

    #check_mix_gather_RS01(device_mesh, rank)


def test_comm_spec():
    world_size = 8
    run_func = partial(check_comm, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_comm_spec()
