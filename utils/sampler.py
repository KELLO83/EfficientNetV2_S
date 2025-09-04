import torch
from torch.utils.data import Sampler
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import math

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