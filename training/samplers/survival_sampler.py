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
            item = dataset[i]
            # Handle both 6-item (old) and 7-item (new with slide_id) formats
            if len(item) == 7:
                _, _, _, status, _, _, _ = item
            else:
                _, _, _, status, _, _ = item
            self.status.append(int(status.item()))
        
        self.status = np.array(self.status, dtype=np.int64)
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
            item = dataset[i]
            # Handle both 6-item (old) and 7-item (new with slide_id) formats
            if len(item) == 7:
                _, _, _, status, time, _, _ = item
            else:
                _, _, _, status, time, _ = item
            self.status.append(int(status.item()))
            self.time.append(float(time.item()))
        
        self.status = np.array(self.status, dtype=np.int64)
        self.time = np.array(self.time, dtype=np.float64)
        
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
            num_status_classes = len(self.target_probs) if isinstance(self.target_probs, np.ndarray) else len(self.target_probs)
            for status_id in range(num_status_classes):
                status_indices[status_id] = all_indices[self.status == status_id]
                np.random.shuffle(status_indices[status_id])
            
            # Create batches with stratified sampling
            batch_indices = []
            status_pointers = {status_id: 0 for status_id in status_indices.keys()}
            
            for batch_idx in range(self.num_batches):
                batch = []
                
                # Calculate how many samples of each status to include
                # Handle both numpy array and dict formats for target_probs
                if isinstance(self.target_probs, np.ndarray):
                    for status_id in range(len(self.target_probs)):
                        prob = self.target_probs[status_id]
                        n_samples = int(self.batch_size * prob)
                        
                        # Get samples from this status
                        if status_id in status_indices:
                            start_ptr = status_pointers[status_id]
                            end_ptr = min(start_ptr + n_samples, len(status_indices[status_id]))
                            
                            batch.extend(status_indices[status_id][start_ptr:end_ptr])
                            status_pointers[status_id] = end_ptr
                else:
                    # Dict format
                    for status_id, prob in self.target_probs.items():
                        n_samples = int(self.batch_size * prob)
                        
                        # Get samples from this status
                        if status_id in status_indices:
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
    """
    A batch sampler that builds batches enriched with comparable (event, at-risk) pairs
    but fixes the number of batches per epoch to len(dataset) // batch_size.
    """

    def __init__(self, dataset, batch_size, shuffle_within=True, seed=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle_within = shuffle_within
        self.seed = seed

        # -------- 1) Extract status and time --------
        self.status = []
        self.time = []

        for i in range(len(dataset)):
            item = dataset[i]
            if len(item) == 7:
                _, _, _, status, t, _, _ = item
            else:
                _, _, _, status, t, _ = item

            status = int(status.item() if hasattr(status, "item") else status)
            t = float(t.item() if hasattr(t, "item") else t)

            self.status.append(status)
            self.time.append(t)

        self.status = np.asarray(self.status, dtype=np.int64)
        self.time = np.asarray(self.time, dtype=np.float64)
        self.n = len(self.status)

        # -------- 2) Build pairs --------
        self.max_pairs_per_event = 20
        self.pairs = self._build_pairs()

        # -------- 3) FIXED number of batches per epoch --------
        self.num_batches = max(1, self.n // self.batch_size)

        print(f"RiskSetBatchSampler: built {len(self.pairs)} comparable pairs from {self.n} samples")
        print(f"Fixed epoch batches = {self.num_batches}, batch size={self.batch_size}")

    def _build_pairs(self):
        pairs = []
        event_indices = np.where(self.status == 1)[0]
        rng = np.random.RandomState(self.seed)

        for i in event_indices:
            t_i = self.time[i]
            mask = self.time >= t_i
            js = np.where(mask)[0]
            js = js[js != i]

            if len(js) == 0:
                continue

            if len(js) > self.max_pairs_per_event:
                js = rng.choice(js, size=self.max_pairs_per_event, replace=False)

            for j in js:
                pairs.append((int(i), int(j)))

        return pairs

    def __iter__(self):
        # Shuffle pairs each epoch
        pairs = np.array(self.pairs)
        if self.shuffle_within:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(pairs)

        batch = []
        used_idx = set()

        ptr = 0
        n_pairs = len(pairs)
        batches_yielded = 0

        # Fixed-number-of-batches loop
        while batches_yielded < self.num_batches:
            # Pair pointer wrap-around
            if ptr >= n_pairs:
                ptr = 0
                if self.shuffle_within:
                    np.random.shuffle(pairs)

            i, j = pairs[ptr]
            ptr += 1

            for idx in (i, j):
                if idx not in used_idx:
                    batch.append(idx)
                    used_idx.add(idx)

                    if len(batch) >= self.batch_size:
                        yield batch[:self.batch_size]
                        batch = []
                        used_idx.clear()
                        batches_yielded += 1
                        break  # end current batch

        # (No leftover batch yielded, because epoch length is fixed)

    def __len__(self):
        return self.num_batches



# class RiskSetBatchSampler(torch.utils.data.Sampler):
#     def __init__(self, dataset, batch_size, min_events=2, overlap=0, shuffle_within=True, seed=None):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.min_events = min_events
#         self.overlap = overlap
#         self.shuffle_within = shuffle_within
#         self.seed = seed

#         # Extract time and status
#         times, status = [], []
#         for i in range(len(dataset)):
#             # Survival dataset returns: features, coords, bag_size, status, time, patient_id, slide_id (7 items)
#             item = dataset[i]
#             if len(item) == 7:
#                 _, _, _, s, t, _, _ = item
#             elif len(item) == 6:
#                 # Old format without slide_id
#                 _, _, _, s, t, _ = item
#             else:
#                 raise ValueError(f"Unexpected dataset item length: {len(item)}")
#             times.append(float(t))
#             status.append(int(s))
#         times = np.asarray(times); status = np.asarray(status)

#         # Sort by time in descending order
#         self.order = np.argsort(-times, kind='mergesort')
#         self.times = times[self.order]
#         self.status = status[self.order]

#         # Pre-generate batch start positions
#         step = self.batch_size - self.overlap
#         self.starts = list(range(0, len(self.order), step))

#     def __iter__(self):
#         rng = np.random.default_rng(self.seed) if self.seed is not None else np.random.default_rng()
#         batches = []
#         n = len(self.order)

#         for s in self.starts:
#             e = min(s + self.batch_size, n)
#             idx = self.order[s:e]
#             # Ensure at least min_events events; extend backward (earlier time) if needed
#             extra_ptr = e
#             while (self.status[idx].sum() < self.min_events) and (extra_ptr < n):
#                 take = min(self.min_events - self.status[idx].sum(), n - extra_ptr)
#                 extra = self.order[extra_ptr: extra_ptr + take]
#                 idx = np.concatenate([idx, extra], axis=0)
#                 extra_ptr += take
#                 if len(idx) >= self.batch_size:
#                     break
#             # Truncate to batch_size if exceeded
#             if len(idx) > self.batch_size:
#                 idx = idx[:self.batch_size]

#             # Light shuffle within window (no cross-window shuffle, preserves time structure)
#             if self.shuffle_within:
#                 rng.shuffle(idx)

#             batches.append(idx.tolist())

#         # Light shuffle of batch order (optional)
#         rng.shuffle(batches)
#         return iter(batches)

#     def __len__(self):
#         return len(self.starts)