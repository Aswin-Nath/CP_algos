class FenwickTree:
  def __init__(self, n: int):
    self.vals = [0] * (n + 1)

  def maximize(self, i: int, val: int) -> None:
    """Updates the maximum sum of subsequence ending in (i - 1) with `val`."""
    while i < len(self.vals):
      self.vals[i] = max(self.vals[i], val)
      i += FenwickTree.lowbit(i)

  def get(self, i: int) -> int:
    """Returns the maximum sum of subsequence ending in (i - 1)."""
    res = 0
    while i > 0:
      res = max(res, self.vals[i])
      i -= FenwickTree.lowbit(i)
    return res

  @staticmethod
  def lowbit(i: int) -> int:
    return i & -i
