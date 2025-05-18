class LCA:
    def __init__(self,n,G):
        self.n=n
        self.G=G
        self.root_weight=[0]*self.n
        self.depth=[0]*n
        self.LOG=1
        current=1
        while current<n:
            current*=2
            self.LOG+=1
        self.parent=[[-1]*n for i in range(self.LOG)]
        def dfs(node):
            for nei,W in G[node]:
                if nei!=self.parent[0][node]:
                    self.depth[nei]=self.depth[node]+1
                    self.parent[0][nei]=node
                    self.root_weight[nei]+=W+self.root_weight[node]
                    dfs(nei)
        dfs(0)
        for jump in range(1,self.LOG):
            for node in range(n):
                if self.parent[0][node]!=-1:
                    self.parent[jump][node]=self.parent[jump-1][self.parent[jump-1][node]]

    def lca(self,a,b):
        def find_lca(u, v):
            lower_node=u
            upper_node=v
            if self.depth[u]<self.depth[v]:
                lower_node=v
                upper_node=u
            depth_difference=self.depth[lower_node]-self.depth[upper_node]
            for jump in range(self.LOG):
                if depth_difference&(1<<jump):
                    lower_node=self.parent[jump][lower_node]
            if lower_node==upper_node:
                return lower_node
            for i in reversed(range(self.LOG)):
                if self.parent[i][lower_node]!=self.parent[i][upper_node]:
                    lower_node=self.parent[i][lower_node]
                    upper_node=self.parent[i][upper_node]
            return self.parent[0][lower_node]
        return find_lca(a,b)
    def D(self,a,b):
        return self.root_weight[a]+self.root_weight[b]-2*self.root_weight[self.lca(a,b)]
