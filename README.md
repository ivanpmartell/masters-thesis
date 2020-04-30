# masters-thesis
Source code for my Master's thesis.

Add the following snippet to `sklearn.utils.safe_indexing`:

```python
elif hasattr(X, "__getitem__"):
    # Work-around for indexing with read-only indices in pandas
    indices = indices if indices.flags.writeable else indices.copy()
    return np.array([X.__getitem__(idx)[0] for idx in indices], dtype=np.float32)
```
