class DSU:
	def __init__(self, sizes: int) -> None:
		self.parents = [i for i in range(sizes+1)]
		self.sizes = [1 for _ in range(sizes+1)]

	def find(self, x: int) -> int:return x if x==self.parents[x] else self.find(self.parents[x])

	def merge(self, x: int, y: int) -> bool:
		x_root = self.find(x)
		y_root = self.find(y)
		if x_root == y_root:
			return False

		if self.sizes[x_root] < self.sizes[y_root]:
			x_root, y_root = y_root, x_root
		self.parents[y_root] = x_root
		self.sizes[x_root] += self.sizes[y_root]
		return True
	def same(self, x: int, y: int):return self.find(x) == self.find(y)
