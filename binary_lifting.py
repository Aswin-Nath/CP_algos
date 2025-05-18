class Solution:
    def minimumWeight(self, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
        ans=[]
        n=len(edges)+1
        root_weight=[0 for i in range(n)]
        G=defaultdict(list)
        for i,j,w in edges:
            G[i].append((j,w))
            G[j].append((i,w))
        root_weight[0]=0
        depth=[0 for i in range(n)]
        LOG=0
        current=1
        while current<n:
            LOG+=1
            current*=2
        parent=[[-1]*n for _ in range(LOG)]
        def dfs(node):
            for nei,W in G[node]:
                if nei!=parent[0][node]:
                    depth[nei]=depth[node]+1
                    parent[0][nei]=node
                    root_weight[nei]+=W+root_weight[node]
                    dfs(nei)
        dfs(0)
        
        for jump in range(1,LOG):
            for node in range(n):
                if parent[0][node]!=-1:
                    parent[jump][node]=parent[jump-1][parent[jump-1][node]]

        # LCA function using binary lifting
        def lca(u, v):
            if depth[u] < depth[v]:
                u, v = v, u
            # Lift u up to depth v
            diff = depth[u] - depth[v]
            for i in range(LOG):
                if diff & (1 << i):
                    u = parent[i][u]
            if u == v:
                return u
            # Lift both u and v up until their parents match
            for i in reversed(range(LOG)):
                if parent[i][u] != parent[i][v]:
                    u = parent[i][u]
                    v = parent[i][v]
            return parent[0][u]
        print(root_weight)
        def D(a,b):
            return root_weight[a]+root_weight[b]-2*root_weight[lca(a,b)]
        def find(a,b,c):
            return (D(a,b)+D(b,c)+D(a,c))//2
        for a,b,c in queries:
            ans.append(find(a,b,c))
        return ans
