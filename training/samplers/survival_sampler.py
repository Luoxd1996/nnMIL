import torch
import numpy as np


class BalancedSurvivalSampler(torch.utils.data.Sampler):
    """
    Sampler that ensures each batch has balanced distribution based on survival status
    """
    def __init__(self, dataset, batch_size, shuffle=True, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        # Get survival status for all samples
        self.status = []
        for i in range(len(dataset)):
            _, _, _, status, _, _ = dataset[i]
            self.status.append(status.item())
        
        self.status = np.array(self.status)
        self.num_classes = len(np.unique(self.status))  # Should be 2 (0=censored, 1=event)
        
        # Calculate samples per class per batch
        self.samples_per_class = max(1, batch_size // self.num_classes)
        self.remainder = batch_size % self.num_classes
        
        # Group samples by status
        self.status_indices = {}
        for status_id in range(self.num_classes):
            self.status_indices[status_id] = np.where(self.status == status_id)[0]
        
        # Calculate total batches based on the minority class
        min_samples_per_class = min(len(indices) for indices in self.status_indices.values())
        if self.samples_per_class > 0:
            self.num_batches = min_samples_per_class // self.samples_per_class
        else:
            self.num_batches = min_samples_per_class
        
        print(f"BalancedSurvivalSampler: {self.num_classes} status classes, {self.samples_per_class} samples per status per batch")
        print(f"Status distribution: {dict(zip(range(self.num_classes), [len(indices) for indices in self.status_indices.values()]))}")
        print(f"Total batches: {self.num_batches}")
        
    def __iter__(self):
        if self.shuffle:
            # Set random seed if provided
            if self.seed is not None:
                np.random.seed(self.seed)
            # Shuffle indices within each status class
            for status_id in self.status_indices:
                np.random.shuffle(self.status_indices[status_id])
        
        batch_indices = []
        for batch_idx in range(self.num_batches):
            batch = []
            
            # Add samples_per_class from each status class
            for status_id in range(self.num_classes):
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                batch.extend(self.status_indices[status_id][start_idx:end_idx])
            
            # Add remainder samples from first few classes
            if self.remainder > 0:
                for i in range(self.remainder):
                    status_id = i % self.num_classes
                    start_idx = batch_idx * self.samples_per_class + i
                    if start_idx < len(self.status_indices[status_id]):
                        batch.append(self.status_indices[status_id][start_idx])
            
            batch_indices.append(batch)
        
        if self.shuffle:
            np.random.shuffle(batch_indices)
        
        return iter(batch_indices)
    
    def __len__(self):
        return self.num_batches


class StratifiedSurvivalSampler(torch.utils.data.Sampler):
    """
    Stratified sampler that maintains survival status distribution across batches
    """
    def __init__(self, dataset, batch_size, shuffle=True, seed=None, stratify_ratio=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.stratify_ratio = stratify_ratio
        
        # Get survival status for all samples
        self.status = []
        self.time = []
        for i in range(len(dataset)):
            _, _, _, status, time, _ = dataset[i]
            self.status.append(status.item())
            self.time.append(time.item())
        
        self.status = np.array(self.status)
        self.time = np.array(self.time)
        
        # Calculate status distribution
        status_counts = np.bincount(self.status)
        status_probs = status_counts / len(self.status)
        
        print(f"StratifiedSurvivalSampler: Status distribution: {dict(zip(range(len(status_counts)), status_counts))}")
        print(f"Status probabilities: {dict(zip(range(len(status_probs)), status_probs))}")
        
        # If stratify_ratio is provided, use it; otherwise use natural distribution
        if self.stratify_ratio is not None:
            self.target_probs = self.stratify_ratio
        else:
            self.target_probs = status_probs
        
        # Calculate total batches
        self.num_batches = len(dataset) // batch_size
        if len(dataset) % batch_size != 0:
            self.num_batches += 1
        
        print(f"Total batches: {self.num_batches}")
        
    def __iter__(self):
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            
            # Create stratified indices
            all_indices = np.arange(len(self.dataset))
            
            # Separate indices by status
            status_indices = {}
            for status_id in range(len(self.target_probs)):
                status_indices[status_id] = all_indices[self.status == status_id]
                np.random.shuffle(status_indices[status_id])
            
            # Create batches with stratified sampling
            batch_indices = []
            status_pointers = {status_id: 0 for status_id in status_indices.keys()}
            
            for batch_idx in range(self.num_batches):
                batch = []
                
                # Calculate how many samples of each status to include
                for status_id, prob in self.target_probs.items():
                    n_samples = int(self.batch_size * prob)
                    
                    # Get samples from this status
                    start_ptr = status_pointers[status_id]
                    end_ptr = min(start_ptr + n_samples, len(status_indices[status_id]))
                    
                    batch.extend(status_indices[status_id][start_ptr:end_ptr])
                    status_pointers[status_id] = end_ptr
                
                # If batch is not full, fill with remaining samples
                if len(batch) < self.batch_size:
                    remaining_needed = self.batch_size - len(batch)
                    all_remaining = []
                    
                    for status_id in status_indices:
                        remaining = status_indices[status_id][status_pointers[status_id]:]
                        all_remaining.extend(remaining)
                    
                    np.random.shuffle(all_remaining)
                    batch.extend(all_remaining[:remaining_needed])
                
                # Ensure batch doesn't exceed batch_size
                batch = batch[:self.batch_size]
                batch_indices.append(batch)
            
            return iter(batch_indices)
        else:
            # No shuffling - return sequential indices
            all_indices = list(range(len(self.dataset)))
            batch_indices = []
            
            for i in range(0, len(all_indices), self.batch_size):
                batch = all_indices[i:i + self.batch_size]
                batch_indices.append(batch)
            
            return iter(batch_indices)
    
    def __len__(self):
        return self.num_batches


class RiskSetBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, min_events=2, overlap=0, shuffle_within=True, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_events = min_events
        self.overlap = overlap
        self.shuffle_within = shuffle_within
        self.seed = seed

        # Extract time and status
        times, status = [], []
        for i in range(len(dataset)):
            # Assume __getitem__ returns (..., status, time, ...)
            _, _, _, s, t, _ = dataset[i]
            times.append(float(t))
            status.append(int(s))
        times = np.asarray(times); status = np.asarray(status)

        # Sort by time in descending order
        self.order = np.argsort(-times, kind='mergesort')
        self.times = times[self.order]
        self.status = status[self.order]

        # Pre-generate batch start positions
        step = self.batch_size - self.overlap
        self.starts = list(range(0, len(self.order), step))

    def __iter__(self):
        rng = np.random.default_rng(self.seed) if self.seed is not None else np.random.default_rng()
        batches = []
        n = len(self.order)

        for s in self.starts:
            e = min(s + self.batch_size, n)
            idx = self.order[s:e]
            # Ensure at least min_events events; extend backward (earlier time) if needed
            extra_ptr = e
            while (self.status[idx].sum() < self.min_events) and (extra_ptr < n):
                take = min(self.min_events - self.status[idx].sum(), n - extra_ptr)
                extra = self.order[extra_ptr: extra_ptr + take]
                idx = np.concatenate([idx, extra], axis=0)
                extra_ptr += take
                if len(idx) >= self.batch_size:
                    break
            # Truncate to batch_size if exceeded
            if len(idx) > self.batch_size:
                idx = idx[:self.batch_size]

            # Light shuffle within window (no cross-window shuffle, preserves time structure)
            if self.shuffle_within:
                rng.shuffle(idx)

            batches.append(idx.tolist())

        # Light shuffle of batch order (optional)
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.starts)