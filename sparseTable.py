class SparseTable:
    def __init__(self, arr):
        self.n = len(arr)
        self.k = int(math.log2(self.n)) + 1
        self.st = [[float("inf")]*self.k for _ in range(self.n)]
        
        for i in range(self.n):
            self.st[i][0] = arr[i]
        
        for j in range(1, self.k):
            for i in range(self.n - (1 << j) + 1):
                self.st[i][j] = min(self.st[i][j-1], self.st[i + (1 << (j-1))][j-1])
    
    def query(self, l, r):
        j = int(math.log2(r - l + 1))
        return min(self.st[l][j], self.st[r - (1 << j) + 1][j])
