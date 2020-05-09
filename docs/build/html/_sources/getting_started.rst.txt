Getting Started
===============

To get started please clone the repository at https://github.com/ivanpmartell/masters-thesis

For our analysis script to work, we needed sci-kit learn's safe indexing to work with pytorch datasets. Therefore, the following snippet to `sklearn.utils.safe_indexing` should be added:

    elif hasattr(X, "__getitem__"):
      indices = indices if indices.flags.writeable else indices.copy()

      return np.array([X.__getitem__(idx)[0] for idx in indices], dtype=np.float32)
