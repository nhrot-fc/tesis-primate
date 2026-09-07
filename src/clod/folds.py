import numpy as np
import numpy.typing as npt


def by_group(group_of_item: npt.NDArray[np.int64], n_folds: int) -> npt.NDArray[np.int64]:
    size_of_group = np.bincount(group_of_item)
    size_of_fold = np.zeros(n_folds, dtype=np.int64)
    fold_of_group = np.empty(len(size_of_group), dtype=np.int64)

    for group in np.argsort(-size_of_group):
        emptiest = int(np.argmin(size_of_fold))
        fold_of_group[group] = emptiest
        size_of_fold[emptiest] += size_of_group[group]
    return fold_of_group[group_of_item]
