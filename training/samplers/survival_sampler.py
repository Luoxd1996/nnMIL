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


class TimeContrastSampler(torch.utils.data.Sampler):
    """
    Time-Contrast Sampler for Survival Analysis.

    This sampler divides the dataset into N time buckets (e.g., early, mid, late events)
    and ensures that each batch contains a balanced mix of samples from all time ranges.

    This maximizes the gradient signal for Cox Loss by ensuring robust comparison candidates
    (Risk Set) exist across the entire time spectrum in every batch, without the downsides
    of sorting data (which hurts Batch Normalization and generalization).
    """
    def __init__(self, dataset, batch_size, buckets=4, shuffle=True, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.buckets = buckets
        self.shuffle = shuffle
        self.seed = seed

        # --- 1. Extract Time Information ---
        # Handle both 6-item (old) and 7-item (new with slide_id) formats: time at index 4
        self.times = []
        for i in range(len(dataset)):
            item = dataset[i]
            time_val = item[4]
            self.times.append(float(time_val.item() if hasattr(time_val, "item") else time_val))

        self.times = np.array(self.times)

        # --- 2. Binning Logic ---
        sorted_indices = np.argsort(self.times)
        self.indices_in_bins = np.array_split(sorted_indices, buckets)

        self.num_batches = len(dataset) // batch_size
        self.samples_per_bin = batch_size // buckets

        print(f"TimeContrastSampler: {self.buckets} time buckets, {self.samples_per_bin} samples per bin per batch, {self.num_batches} batches")

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        shuffled_bins = [
            np.random.permutation(b) if self.shuffle else np.array(b)
            for b in self.indices_in_bins
        ]

        bin_pointers = [0] * self.buckets
        final_indices = []

        for _ in range(self.num_batches):
            batch = []

            for bin_i in range(self.buckets):
                start = bin_pointers[bin_i]
                end = start + self.samples_per_bin

                if end > len(shuffled_bins[bin_i]):
                    if self.shuffle:
                        shuffled_bins[bin_i] = np.random.permutation(self.indices_in_bins[bin_i])
                    start = 0
                    end = self.samples_per_bin
                    bin_pointers[bin_i] = 0

                batch.extend(shuffled_bins[bin_i][start:end].tolist())
                bin_pointers[bin_i] = end

            np.random.shuffle(batch)
            final_indices.append(batch)

        if self.shuffle:
            np.random.shuffle(final_indices)

        return iter(final_indices)

    def __len__(self):
        return self.num_batches

