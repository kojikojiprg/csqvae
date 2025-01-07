import os

import numpy as np
import torchvision
from torchvision import transforms


class MNIST(torchvision.datasets.MNIST):
    def __init__(
        self,
        train: bool,
        n_labeled_samples: float = None,
        seed: int = 42,
        root: str = "data/",
        download: bool = True,
        summary_path: str = None,
    ):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
            ]
        )
        super().__init__(root, train, transform, download=download)

        if train:
            self.replace_labels_to_nan(n_labeled_samples, seed, summary_path)

    def replace_labels_to_nan(
        self, n_labeled_samples: float, seed: int, summary_path: str = None
    ):
        np.random.seed(seed)
        random_indices = np.random.choice(len(self), len(self), replace=False)
        unlabeled_indices = random_indices[n_labeled_samples:]
        self.targets[unlabeled_indices] = -1

        # summarize
        if summary_path is not None and "WORLD_SIZE" not in os.environ:
            labeled_targets = self.targets[self.targets != -1]
            unique, counts = np.unique(labeled_targets, return_counts=True)
            summary = [("label", "count")]
            summary.extend([(i, c) for i, c in enumerate(counts)])
            summary.append(("unlabeled", len(unlabeled_indices)))
            summary.append(("total", len(random_indices)))
            np.savetxt(summary_path, summary, fmt="%s", delimiter="\t")
