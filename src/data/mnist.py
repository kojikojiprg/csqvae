import numpy as np
import torchvision
from torchvision.transforms import ToTensor


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
        super().__init__(root, train, ToTensor(), download=download)

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
        if summary_path is not None:
            labeled_targets = self.targets[self.targets != -1]
            unique, counts = np.unique(labeled_targets, return_counts=True)
            summary = [("label", "count")]
            summary.extend([(i, c) for i, c in enumerate(counts)])
            summary.append(("total", np.sum(counts).item()))
            np.savetxt(summary_path, summary, fmt="%s", delimiter="\t")
