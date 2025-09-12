import torch
from torch.utils.data import Sampler
import torch.distributed as dist
import numpy as np
from typing import List, Iterator
import torch
from torch.utils.data import Sampler
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import math

class BalancedBatchSampler(Sampler[List[int]]):
    """
    A custom sampler to create batches with a balanced number of samples from four different data sources.

    It ensures that each batch contains an equal number of samples from:
    1. wear_data1
    2. wear_data2
    3. no_wear_data1
    4. no_wear_data2

    Smaller datasets are oversampled to match the length of the largest dataset, ensuring
    all data from the largest source is seen once per epoch.
    It also supports DistributedDataParallel (DDP) by splitting batches across replicas.
    """
    def __init__(self, dataset, batch_size: int, world_size: int, rank: int, drop_last: bool = True):
        """
        Args:
            dataset: A dataset object that must have a `lengths` attribute, which is a list
                     of lengths of the four data sources.
            batch_size (int): The total batch size. Must be divisible by 4.
            world_size (int): The number of processes for distributed training.
            rank (int): The rank of the current process.
            drop_last (bool): If `True`, the sampler will drop the last incomplete batch.
        """
        if not hasattr(dataset, 'lengths') or len(dataset.lengths) != 4:
            raise ValueError("Dataset must have a 'lengths' attribute with 4 elements.")
        if batch_size % 4 != 0:
            raise ValueError("Batch size must be divisible by 4.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0

        self.num_sources = 4
        self.samples_per_source = self.batch_size // self.num_sources

        self.source_lengths = self.dataset.lengths
        self.cumulative_lengths = self.dataset.cumulative_lengths

        # Determine the number of batches based on the largest source
        self.max_source_len = max(self.source_lengths)
        self.num_batches_per_epoch = self.max_source_len // self.samples_per_source
        if not self.drop_last and self.max_source_len % self.samples_per_source != 0:
            self.num_batches_per_epoch += 1

        self.total_size = self.num_batches_per_epoch * self.batch_size

        if self.world_size > 1:
            self.num_samples_per_replica = self.total_size // self.world_size
        else:
            self.num_samples_per_replica = self.total_size


    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # Generate shuffled indices for each source with oversampling
        indices_per_source = []
        for i in range(self.num_sources):
            source_start_idx = self.cumulative_lengths[i] - self.source_lengths[i]
            source_indices = np.arange(source_start_idx, self.cumulative_lengths[i])

            # Oversample smaller datasets to match the largest one
            if self.source_lengths[i] > 0:
                num_repeats = (self.max_source_len + self.source_lengths[i] - 1) // self.source_lengths[i]
                repeated_indices = np.tile(source_indices, num_repeats)
                
                # Shuffle and truncate to the size of the largest dataset
                perm = torch.randperm(len(repeated_indices), generator=g).numpy()
                shuffled = repeated_indices[perm][:self.max_source_len]
                indices_per_source.append(shuffled)
            else:
                indices_per_source.append(np.array([], dtype=np.int64))


        # Combine indices into batches
        all_batch_indices = []
        for i in range(self.num_batches_per_epoch):
            batch = []
            for source_idx in range(self.num_sources):
                if len(indices_per_source[source_idx]) > 0:
                    start = i * self.samples_per_source
                    end = start + self.samples_per_source
                    batch.extend(indices_per_source[source_idx][start:end])
            
            # Shuffle within the batch to mix sources
            perm = torch.randperm(len(batch), generator=g).numpy()
            batch = np.array(batch)[perm].tolist()
            all_batch_indices.extend(batch)

        # Subsample for the current replica in DDP
        if self.world_size > 1:
            # Ensure all replicas have the same number of samples
            total_size_for_ddp = self.num_samples_per_replica * self.world_size
            all_batch_indices = all_batch_indices[:total_size_for_ddp]
            replica_indices = all_batch_indices[self.rank:total_size_for_ddp:self.world_size]
        else:
            replica_indices = all_batch_indices

        return iter(replica_indices)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int):
        self.epoch = epoch

class DomainBalancedSampler(Sampler):
    """
    A custom sampler for single-GPU training to ensure each batch contains a specific 
    fraction of samples from a minority domain (Domain B).
    """
    def __init__(self, dataset, batch_size, domain_B_fraction=0.3):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.domain_B_fraction = domain_B_fraction

        self.domain_A_indices = [i for i, label in enumerate(dataset.domain_labels) if label == 0]
        self.domain_B_indices = [i for i, label in enumerate(dataset.domain_labels) if label == 1]

        self.num_B_samples_per_batch = int(self.batch_size * self.domain_B_fraction)
        self.num_A_samples_per_batch = self.batch_size - self.num_B_samples_per_batch

        self.num_samples = len(self.dataset)
        
        if not self.domain_B_indices:
            raise ValueError("Domain B has no samples.")
        if not self.domain_A_indices:
            raise ValueError("Domain A has no samples.")

        if len(self.domain_B_indices) < self.num_B_samples_per_batch:
            print(f"Warning: Not enough samples in Domain B ({len(self.domain_B_indices)}) to meet the fraction requirement "
                  f"({self.num_B_samples_per_batch}). The number of Domain B samples per batch will be capped.")
            self.num_B_samples_per_batch = len(self.domain_B_indices)
            self.num_A_samples_per_batch = self.batch_size - self.num_B_samples_per_batch

    def __iter__(self):
        np.random.shuffle(self.domain_A_indices)
        np.random.shuffle(self.domain_B_indices)
        iter_A = iter(self.domain_A_indices)
        iter_B = iter(self.domain_B_indices)

        epoch_indices = []
        num_batches = self.num_samples // self.batch_size
        for _ in range(num_batches):
            batch = []
            for _ in range(self.num_B_samples_per_batch):
                try:
                    batch.append(next(iter_B))
                except StopIteration:
                    np.random.shuffle(self.domain_B_indices)
                    iter_B = iter(self.domain_B_indices)
                    batch.append(next(iter_B))
            
            for _ in range(self.num_A_samples_per_batch):
                try:
                    batch.append(next(iter_A))
                except StopIteration:
                    np.random.shuffle(self.domain_A_indices)
                    iter_A = iter(self.domain_A_indices)
                    batch.append(next(iter_A))
            
            epoch_indices.extend(batch)
        
        return iter(epoch_indices)

    def __len__(self):
        return (self.num_samples // self.batch_size) * self.batch_size

class DistributedDomainBalancedSampler(DistributedSampler):
    """
    A DistributedSampler that also ensures each batch contains a specific fraction
    of samples from a minority domain (Domain B).
    It first creates a balanced list of indices and then distributes
    a unique subset of this list to each GPU.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 seed=0, drop_last=False, batch_size=None, domain_B_fraction=0.3):
        
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)

        self.batch_size = batch_size
        self.domain_B_fraction = domain_B_fraction
        if self.batch_size is None:
            raise ValueError("batch_size must be provided to DistributedDomainBalancedSampler")

        self.domain_A_indices = [i for i, label in enumerate(dataset.domain_labels) if label == 0]
        self.domain_B_indices = [i for i, label in enumerate(dataset.domain_labels) if label == 1]

        self.num_B_samples_per_batch = int(self.batch_size * self.domain_B_fraction)
        self.num_A_samples_per_batch = self.batch_size - self.num_B_samples_per_batch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices_A = torch.tensor(self.domain_A_indices)[torch.randperm(len(self.domain_A_indices), generator=g)].tolist()
        indices_B = torch.tensor(self.domain_B_indices)[torch.randperm(len(self.domain_B_indices), generator=g)].tolist()

        iter_A = iter(indices_A)
        iter_B = iter(indices_B)

        total_batches = len(self.dataset) // self.batch_size
        
        full_epoch_indices = []
        for _ in range(total_batches):
            batch = []
            for _ in range(self.num_B_samples_per_batch):
                try:
                    batch.append(next(iter_B))
                except StopIteration:
                    indices_B = torch.tensor(self.domain_B_indices)[torch.randperm(len(self.domain_B_indices), generator=g)].tolist()
                    iter_B = iter(indices_B)
                    batch.append(next(iter_B))
            
            for _ in range(self.num_A_samples_per_batch):
                try:
                    batch.append(next(iter_A))
                except StopIteration:
                    indices_A = torch.tensor(self.domain_A_indices)[torch.randperm(len(self.domain_A_indices), generator=g)].tolist()
                    iter_A = iter(indices_A)
                    batch.append(next(iter_A))
            
            full_epoch_indices.extend(batch)

        if not self.drop_last:
            padding_size = self.total_size - len(full_epoch_indices)
            if padding_size <= len(full_epoch_indices):
                full_epoch_indices += full_epoch_indices[:padding_size]
            else:
                full_epoch_indices += (full_epoch_indices * math.ceil(padding_size / len(full_epoch_indices)))[:padding_size]
        else:
            full_epoch_indices = full_epoch_indices[:self.total_size]
        
        assert len(full_epoch_indices) == self.total_size

        indices_for_this_rank = full_epoch_indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices_for_this_rank) == self.num_samples

        return iter(indices_for_this_rank)
    """
    A DistributedSampler that also ensures each batch contains a specific fraction
    of samples from a minority domain (Domain B).
    It first creates a balanced list of indices and then distributes
    a unique subset of this list to each GPU.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 seed=0, drop_last=False, batch_size=None, domain_B_fraction=0.3):
        
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)

        self.batch_size = batch_size
        self.domain_B_fraction = domain_B_fraction
        if self.batch_size is None:
            raise ValueError("batch_size must be provided to DistributedDomainBalancedSampler")

        self.domain_A_indices = [i for i, label in enumerate(dataset.domain_labels) if label == 0]
        self.domain_B_indices = [i for i, label in enumerate(dataset.domain_labels) if label == 1]

        self.num_B_samples_per_batch = int(self.batch_size * self.domain_B_fraction)
        self.num_A_samples_per_batch = self.batch_size - self.num_B_samples_per_batch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices_A = torch.tensor(self.domain_A_indices)[torch.randperm(len(self.domain_A_indices), generator=g)].tolist()
        indices_B = torch.tensor(self.domain_B_indices)[torch.randperm(len(self.domain_B_indices), generator=g)].tolist()

        iter_A = iter(indices_A)
        iter_B = iter(indices_B)

        total_batches = len(self.dataset) // self.batch_size
        
        full_epoch_indices = []
        for _ in range(total_batches):
            batch = []
            for _ in range(self.num_B_samples_per_batch):
                try:
                    batch.append(next(iter_B))
                except StopIteration:
                    indices_B = torch.tensor(self.domain_B_indices)[torch.randperm(len(self.domain_B_indices), generator=g)].tolist()
                    iter_B = iter(indices_B)
                    batch.append(next(iter_B))
            
            for _ in range(self.num_A_samples_per_batch):
                try:
                    batch.append(next(iter_A))
                except StopIteration:
                    indices_A = torch.tensor(self.domain_A_indices)[torch.randperm(len(self.domain_A_indices), generator=g)].tolist()
                    iter_A = iter(indices_A)
                    batch.append(next(iter_A))
            
            full_epoch_indices.extend(batch)

        if not self.drop_last:
            padding_size = self.total_size - len(full_epoch_indices)
            if padding_size <= len(full_epoch_indices):
                full_epoch_indices += full_epoch_indices[:padding_size]
            else:
                full_epoch_indices += (full_epoch_indices * math.ceil(padding_size / len(full_epoch_indices)))[:padding_size]
        else:
            full_epoch_indices = full_epoch_indices[:self.total_size]
        
        assert len(full_epoch_indices) == self.total_size

        indices_for_this_rank = full_epoch_indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices_for_this_rank) == self.num_samples

        return iter(indices_for_this_rank)