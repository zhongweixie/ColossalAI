import torch
import random
from typing import List, Any
from .base import ReplayBuffer
# from torch.multiprocessing import Queue
from ray.util.queue import Queue
import ray
from chatgpt.experience_maker.base import Experience
from .utils import BufferItem, make_experience_batch, split_experience_batch
from threading import Lock
import copy

class DetachedReplayBuffer:
    '''
        Detached replay buffer. Share Experience across workers on the same node. 
        Therefore a trainer node is expected to have only one instance. 
        It is ExperienceMakerHolder's duty to call append(exp) method, remotely.
    
    Args:
        sample_batch_size: Batch size when sampling. Exp won't enqueue until they formed a batch.
        tp_world_size: Number of workers in the same tp group
        limit: Limit of number of experience sample BATCHs. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload: Whether to offload experience to cpu when sampling. Defaults to True.
    '''

    def __init__(self, sample_batch_size: int, tp_world_size: int = 1, limit : int = 0, cpu_offload: bool = True) -> None:
        self.cpu_offload = cpu_offload
        self.sample_batch_size = sample_batch_size
        self.limit = limit
        self.items = Queue(self.limit, actor_options={"num_cpus":1})
        self.batch_collector : List[BufferItem] = []

        '''
        Workers in the same tp group share this buffer and need same sample for one step.
            Therefore a held_sample should be returned tp_world_size times before it could be dropped.
            worker_state records wheter a worker got the held_sample
        '''
        self.tp_world_size = tp_world_size
        self.worker_state = [False] * self.tp_world_size
        self.held_sample = None
        self._worker_state_lock = Lock()

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        '''
        Expected to be called remotely.
        '''
        if self.cpu_offload:
            experience.to_device(torch.device('cpu'))
        items = split_experience_batch(experience)
        self.batch_collector.extend(items)
        while len(self.batch_collector) >= self.sample_batch_size:
            items = self.batch_collector[:self.sample_batch_size]
            experience = make_experience_batch(items)
            self.items.put(experience, block=True)
            print(" queue exp in")
            self.batch_collector = self.batch_collector[self.sample_batch_size:]

    def clear(self) -> None:
        # self.items.close()
        self.items.shutdown()
        self.items = Queue(self.limit)
        self.worker_state = [False] * self.tp_world_size
        self.batch_collector = []
     
    @torch.no_grad()
    def sample(self, worker_rank = 0, to_device = "cpu") -> Experience:
        self._worker_state_lock.acquire()
        if not any(self.worker_state):
            self.held_sample = self._sample_and_erase()
        self.worker_state[worker_rank] = True
        if all(self.worker_state):
            self.worker_state = [False] * self.tp_world_size
            ret = self.held_sample
        else:
            ret = copy.deepcopy(self.held_sample)
        self._worker_state_lock.release()
        ret.to_device(to_device)
        return ret

    @torch.no_grad()
    def _sample_and_erase(self) -> Experience:
        ret = self.items.get(block=True)
        print(" queue exp out")
        return ret

    def get_length(self) -> int:
        ret = self.items.qsize()
        print(" queue return length")
        return ret