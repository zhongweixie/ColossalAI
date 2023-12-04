from typing import Any


class CacheBlock:
    """A simplified version of logical cache block used for Paged Attention."""

    def __init__(self, block_id: int, block_size: int, elem_size: int, k_ptr: Any = None, v_ptr: Any = None):
        # Unique id of a cache block
        self.block_id = block_id

        # size/capacity of the block in terms of the number of tokens it can hold
        self.block_size = block_size

        # element size in bytes
        self.elem_size = elem_size

        # For common cases, we track the relationships between logical and physical caches in KV Cache Manager,
        # Additionally, k, v pointers can be optionally used for tracking the physical cache by CacheBlock itself.
        self.k_ptr = k_ptr
        self.v_ptr = v_ptr

        # Post-initialization setup
        self.ref_count = 0
        # the number of slots that have been allocated (i.e. the number of tokens occupying the block)
        self.allocated_size = 0
        self.token_ids = [None] * self.block_size

    def add_ref(self) -> None:
        self.ref_count += 1

    def remove_ref(self) -> None:
        assert self.ref_count > 0, f"Block#{self.block_id} has no reference to remove."
        self.ref_count -= 1

    def has_ref(self) -> bool:
        return self.ref_count > 0

    def has_space(self) -> bool:
        return self.allocated_size < self.block_size

    def available_space(self) -> int:
        return self.block_size - self.allocated_size

    def allocate(self, size: int) -> None:
        assert size <= self.available_space(), f"Block#{self.block_id} has no available space to allocate."
        self.allocated_size += size

    def is_empty(self):
        return self.allocated_size == 0

    def clear(self) -> None:
        assert self.ref_count > 0, f"Block#{self.block_id} has no reference to free."
        self.ref_count = 0
        self.allocated_size = 0
