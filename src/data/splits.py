from collections import defaultdict
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset


def stratified_split(
    dataset: Dataset, val_per_class: int = 2
) -> Tuple[List[int], List[int]]:
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        np.random.shuffle(indices)
        val_indices.extend(indices[:val_per_class])
        train_indices.extend(indices[val_per_class:])

    return train_indices, val_indices
