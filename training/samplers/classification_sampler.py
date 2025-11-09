"""
Classification batch samplers for MIL training.
"""
import torch
import numpy as np


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Sampler that ensures each batch has balanced class distribution.
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=None, labels=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        assert self.batch_size > 0
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if labels is None:
            lbls = []
            for i in range(len(dataset)):
                sample = dataset[i]
                # Label is always at index 3 regardless of tuple length
                # Format: (features, coords, bag_size, label, ...)
                label = sample[3]
                lbls.append(int(label.item() if hasattr(label, "item") else label))
            self.labels = np.asarray(lbls, dtype=int)
        else:
            self.labels = np.asarray(labels, dtype=int)

        self.unique_labels = np.unique(self.labels)
        self.num_classes = len(self.unique_labels)
        if self.num_classes == 0:
            raise ValueError("No classes found.")

        self.class_to_indices = {
            lab: np.where(self.labels == lab)[0].copy()
            for lab in self.unique_labels
        }
        
        # Calculate samples per class in each batch
        # Ensure each class gets at least 1 sample if batch_size >= num_classes
        self.per_class_base = max(1, self.batch_size // self.num_classes)
        self.remainder = self.batch_size - self.per_class_base * self.num_classes
        
        # Calculate number of batches based on smallest class
        min_class_size = min(len(v) for v in self.class_to_indices.values())
        self.num_batches = max(1, min_class_size // self.per_class_base)

    def __iter__(self):
        # Reset pointer for each epoch
        ptr = {lab: 0 for lab in self.unique_labels}
        
        # Shuffle indices for each class at the start of each epoch
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch if self.seed is not None else None)
            shuffled_indices = {}
            for lab in self.unique_labels:
                shuffled_indices[lab] = self.class_to_indices[lab].copy()
                rng.shuffle(shuffled_indices[lab])
        else:
            shuffled_indices = {lab: self.class_to_indices[lab].copy() for lab in self.unique_labels}
        
        batches = []
        labs = list(self.unique_labels)
        round_robin_idx = 0

        # Generate batches
        while True:
            # Determine which classes get extra samples in this batch
            extra_set = set()
            if self.remainder > 0:
                start = round_robin_idx % self.num_classes
                order = labs[start:] + labs[:start]
                extra_set = set(order[:self.remainder])

            # Check if we can form a complete batch
            feasible = True
            needed_per_lab = {}
            for lab in labs:
                need = self.per_class_base + (1 if lab in extra_set else 0)
                needed_per_lab[lab] = need
                if ptr[lab] + need > len(shuffled_indices[lab]):
                    feasible = False
                    break

            if not feasible:
                break

            # Form the batch
            batch = []
            for lab in labs:
                idxs = shuffled_indices[lab]
                p = ptr[lab]
                need = needed_per_lab[lab]
                batch.extend(idxs[p : p + need].tolist())
                ptr[lab] += need

            # Shuffle samples within the batch
            if self.shuffle:
                rng.shuffle(batch)  # Use the same rng with seed
            
            batches.append(batch)
            round_robin_idx += 1

        # Shuffle batch order
        if self.shuffle:
            rng.shuffle(batches)  # Use the same rng with seed

        # Yield batches
        for b in batches:
            yield b
            
        # Increment epoch counter
        self.epoch += 1

    def __len__(self):
        return self.num_batches


class AUCBatchSampler(torch.utils.data.Sampler):
    """
    AUC-oriented stratified BatchSampler with low-variance allocation.

    Key ideas vs. naive random-by-prior:
    - Per-batch class counts are determined by a proportional allocator
      (cumulative target minus cumulative assigned + largest-remainder),
      which greatly reduces variance across batches while preserving
      the dataset's natural class prior (good for AUC / ranking).
    - Within each class we sample WITHOUT replacement until the pool
      is exhausted, then reshuffle and wrap. This improves coverage
      and stability compared to pure random with replacement.
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = seed

        # 1) Collect labels; handle different return formats
        # Format: (features, coords, bag_size, label, ...)
        # Label is always at index 3 regardless of tuple length
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            # Label is always at index 3
            y = sample[3]
            labels.append(int(y.item() if torch.is_tensor(y) else y))
        self.labels = np.asarray(labels, dtype=np.int64)

        # 2) Class stats and priors
        self.classes, counts = np.unique(self.labels, return_counts=True)
        self.num_classes = len(self.classes)
        self.priors = counts / counts.sum()

        # 3) Per-class index pools (use no-replacement cycling)
        self._rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
        self.class_pools = {}
        self.class_ptrs = {}
        for cid in self.classes:
            pool = np.where(self.labels == cid)[0]
            if self.shuffle:
                self._rng.shuffle(pool)
            self.class_pools[int(cid)] = pool
            self.class_ptrs[int(cid)] = 0  # cursor

        # 4) Epoch size ~ cover the dataset once
        self.N = len(self.labels)
        self.num_batches = int(np.ceil(self.N / float(self.batch_size)))

        # 5) Precompute a LOW-VARIANCE class-count plan for every batch
        self.plan = self._build_allocation_plan()

        print(f"AUCBatchSampler: {self.num_classes} classes, priors={np.round(self.priors, 4).tolist()}")
        print(f"Total batches: {self.num_batches} (low-variance proportional allocation)")

    def _build_allocation_plan(self):
        """
        Build a matrix of shape [num_batches, num_classes] with integer counts
        per batch, so that cumulative counts track the global prior as closely
        as possible. This reduces variance vs. naive multinomial draws.
        """
        B = self.num_batches
        C = self.num_classes
        bs = self.batch_size
        pri = self.priors.astype(np.float64)

        # cumulative targets after each batch b: (b+1)*bs*pri
        plan = np.zeros((B, C), dtype=np.int64)
        assigned_cum = np.zeros(C, dtype=np.int64)

        for b in range(B):
            target_cum = np.floor(((b + 1) * bs) * pri).astype(np.int64)
            need = target_cum - assigned_cum  # base integer need
            need = np.maximum(need, 0)

            # If rounding down lost some slots, distribute by largest fractional remainders
            base_sum = int(need.sum())
            short = bs - base_sum
            if short > 0:
                # fractional parts at this batch boundary
                frac = (((b + 1) * bs) * pri) - ((b + 1) * bs * pri).astype(np.int64)
                # Break ties with small random jitter for stability
                jitter = self._rng.uniform(0.0, 1e-6, size=frac.shape)
                order = np.argsort(-(frac + jitter))
                for k in range(short):
                    need[order[k % C]] += 1
            elif short < 0:
                # very rare; trim from smallest fractional remainders
                over = -short
                frac = (((b + 1) * bs) * pri) - ((b + 1) * bs * pri).astype(np.int64)
                jitter = self._rng.uniform(0.0, 1e-6, size=frac.shape)
                order = np.argsort(frac + jitter)  # remove from smallest frac first
                for k in range(over):
                    # only remove where need > 0
                    for j in range(C):
                        c = order[j]
                        if need[c] > 0:
                            need[c] -= 1
                            break

            plan[b] = need
            assigned_cum += need

        # Final tiny correction (if any drift remains due to floors)
        # Ensure each row sums exactly to batch_size (it should already).
        assert np.all(plan.sum(axis=1) == self.batch_size), "Plan rows must sum to batch_size"
        return plan

    def _draw_from_class(self, cid, k):
        """
        Draw k indices from class cid using 'no-replacement until exhausted' cycling.
        """
        pool = self.class_pools[cid]
        ptr = self.class_ptrs[cid]
        out = []

        remaining = k
        while remaining > 0:
            available = len(pool) - ptr
            if available == 0:
                # wrap & reshuffle pool for next cycle
                ptr = 0
                if self.shuffle:
                    self._rng.shuffle(pool)
                self.class_pools[cid] = pool  # not strictly needed but explicit
                available = len(pool)
            take = min(available, remaining)
            out.extend(pool[ptr:ptr + take].tolist())
            ptr += take
            remaining -= take

        self.class_ptrs[cid] = ptr
        return out

    def __iter__(self):
        # Reset pointers at the start of each epoch
        for cid in self.classes:
            self.class_ptrs[int(cid)] = 0
            # Reshuffle each class pool for the new epoch
            if self.shuffle:
                self._rng.shuffle(self.class_pools[int(cid)])

        # Generate batches according to plan
        for b in range(self.num_batches):
            batch = []
            for c_idx, cid in enumerate(self.classes):
                k = int(self.plan[b, c_idx])
                if k > 0:
                    batch.extend(self._draw_from_class(int(cid), k))
            
            # Shuffle batch order (optional)
            if self.shuffle:
                self._rng.shuffle(batch)
            
            yield batch

    def __len__(self):
        return self.num_batches

