"""
Regression batch samplers for MIL training.
"""
import torch
import numpy as np


class RegressionBatchSampler(torch.utils.data.Sampler):
    """
    Regression-oriented batch sampler that ensures balanced representation
    across different target value ranges.
    """
    def __init__(self, dataset, batch_size, shuffle=True, seed=None, num_bins=10):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.num_bins = num_bins

        # Collect target values
        # Format: (features, coords, bag_size, label, ...)
        # Label is always at index 3 regardless of tuple length
        targets = []
        for i in range(len(dataset)):
            sample = dataset[i]
            # Label is always at index 3
            target = sample[3]
            targets.append(float(target.item() if torch.is_tensor(target) else target))
        
        self.targets = np.array(targets)
        
        # Create bins based on target value ranges
        self.bins = np.linspace(self.targets.min(), self.targets.max(), self.num_bins + 1)
        self.bin_indices = np.digitize(self.targets, self.bins) - 1
        self.bin_indices = np.clip(self.bin_indices, 0, self.num_bins - 1)
        
        # Group samples by bins
        self.bin_pools = {}
        self.bin_ptrs = {}
        for bin_id in range(self.num_bins):
            pool = np.where(self.bin_indices == bin_id)[0]
            if self.shuffle:
                np.random.seed(self.seed)
                np.random.shuffle(pool)
            self.bin_pools[bin_id] = pool
            self.bin_ptrs[bin_id] = 0
        
        # Calculate total batches
        self.N = len(self.targets)
        self.num_batches = int(np.ceil(self.N / float(self.batch_size)))
        
        print(f"RegressionBatchSampler: {self.num_bins} bins, {self.num_batches} batches")
        print(f"Target range: [{self.targets.min():.2f}, {self.targets.max():.2f}]")

    def _draw_from_bin(self, bin_id, k):
        """Draw k indices from bin using cycling without replacement."""
        pool = self.bin_pools[bin_id]
        ptr = self.bin_ptrs[bin_id]
        out = []

        remaining = k
        while remaining > 0:
            available = len(pool) - ptr
            if available == 0:
                # Wrap and reshuffle
                ptr = 0
                if self.shuffle:
                    np.random.shuffle(pool)
                self.bin_pools[bin_id] = pool
                available = len(pool)
            take = min(available, remaining)
            out.extend(pool[ptr:ptr + take].tolist())
            ptr += take
            remaining -= take

        self.bin_ptrs[bin_id] = ptr
        return out

    def __iter__(self):
        batches = []
        for b in range(self.num_batches):
            batch = []
            
            # Distribute samples across bins
            samples_per_bin = self.batch_size // self.num_bins
            remainder = self.batch_size % self.num_bins
            
            # Add samples from each bin
            for bin_id in range(self.num_bins):
                count = samples_per_bin
                if bin_id < remainder:
                    count += 1
                if count > 0:
                    batch.extend(self._draw_from_bin(bin_id, count))
            
            if self.shuffle:
                np.random.shuffle(batch)
            batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches

