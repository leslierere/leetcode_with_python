### 261. Graph Valid Tree

https://leetcode.com/problems/graph-valid-tree/

#### Solution-dfs-worth

就是以一个为起点，每一个都只会恰好visit一遍

当然visited列表也可以用set替换

Ref: https://www.cnblogs.com/grandyang/p/5257919.html

```python

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        neighbors = {i:[] for i in range(n)}
        visited = [False]*n
        
        for edge in edges:
            neighbors[edge[0]].append(edge[1])
            neighbors[edge[1]].append(edge[0])
        if not self.dfs(neighbors, visited, 0, -1):
            return False
            
        return all(visited)
        
    def dfs(self, neighbors, visited, cur, pre):
        if visited[cur]:
            return False
        visited[cur] = True
        for node in neighbors[cur]:
            if node!= pre:
                if not self.dfs(neighbors, visited, node, cur):
                    return False
        return True
```



#### Solution-bfs-worth

Ref: https://www.cnblogs.com/grandyang/p/5257919.html

```python
from collections import deque
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        neighbors = {i:set() for i in range(n)}
        visited = [True]+[False]*(n-1)
        queue = deque([0])
        
        for edge in edges:
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
            
        while queue:
            parent = queue.popleft()
            
            for node in neighbors[parent]:
                if visited[node]:
                    return False
                visited[node] = True
                queue.append(node)
                neighbors[node].remove(parent) # major diff from dfs
                
        return all(visited)
```



#### Solution-union find-worth

Ref: https://leetcode.com/problems/graph-valid-tree/discuss/69019/Simple-and-clean-c%2B%2B-solution-with-detailed-explanation.





### 323. Number of Connected Components in an Undirected Graph

https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/

#### Solution-bfs, did@1.20

```python
from collections import deque
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        if not n:
            return 0
        neighbors = {i:set() for i in range(n)}
        for edge in edges:
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
        
        notV = set(range(1, n)) # node not visited 
        
        queue = deque([0])
        res = 1
        
        while queue or len(notV)!=0:
            if not queue:
                queue = deque([notV.pop()])
                res+=1
            parent = queue.popleft()
            for node in neighbors[parent]:
                if node in notV:
                    queue.append(node)
                    notV.remove(node)
                neighbors[node].remove(parent)#general
                
        return res
```



#### Solution-dfs, worth

#### Solution-dfs, union find





### 305. Number of Islands II

https://leetcode.com/problems/number-of-islands-ii/description/

#### Solution-union find/disjoint sets

Ref: https://www.cs.princeton.edu/~rs/AlgsDS07/01UnionFind.pdf

https://leetcode.com/problems/number-of-islands-ii/discuss/75468/Compact-Python.

CS61B-lec33

```python
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        parent = {}
        size = {}
        count = [0] # 这样在union()函数中才可以直接使用count，如果是primitive type就会不行
        res = []
        
        def union(x, y):
            x = find(x)
            y = find(y)
            if x!=y: 
              # if they are in the same set, we don't need union
              # as in the set, all items in the same set will have the same parent
                if size[x] < size[y]:
                    parent[x] = y
                    size[y]+=size[x]
                else:
                    parent[y] = x
                    size[x]+=size[y]
                count[0]-=1
                
        def find(x):
            if x != parent[x]:
                parent[x]=find(parent[x])
            return parent[x]
            
        for i,j in positions:
            if (i,j) not in parent: # to prevent situation where there are duplicates of positions
                count[0]+=1
                parent[(i,j)] = (i,j)
                size[(i,j)] = 1
                for x, y in [(0, 1), (0, -1), (1, 0), (-1,0)]:# get the neighbor position
                    neighbor = i+x, j+y
                    if neighbor in parent:
                        union((i, j), neighbor)
            res.append(count[0])
                    
        return res
```



### 133. Clone Graph

https://leetcode.com/problems/clone-graph/description/

#### Solution-dfs-recursive, by myself

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = []):
        self.val = val
        self.neighbors = neighbors
"""
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return node
        valDic = {}
        return self.helper(node, valDic)
        
    def helper(self, node, valDic):
        if node.val in valDic:
            return valDic[node.val]
        
        value = node.val
        newN = Node(value)
        valDic[value] = newN
        for neighbor in node.neighbors:
            newN.neighbors.append(self.helper(neighbor, valDic))
        return newN
```



#### Solution-dfs-iterative-worth



#### Solution-bfs-iterative-worth



### 399. Evaluate Division

https://leetcode.com/problems/evaluate-division/description/

#### Solution-dfs-recursive-by me

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        dic = {}
        
        for i in range(len(values)):
            num,deno = equations[i]
            if num not in dic:
                dic[num] = {deno:values[i]}
            else:
                dic[num][deno] = values[i]
            if deno not in dic:
                dic[deno] = {num: 1/values[i]}
            else:
                dic[deno][num] = 1/values[i]
        
        res = []
        for x,y in queries:
            res.append(self.bfs(x, y, dic, 1.0, set([x])))
        return res
        
        
    def bfs(self, start, end, dic, cur, visited):
        if start not in dic:
            return -1.0
        elif start==end:
            return cur
        for i in dic[start]:
            if i not in visited:
                visited.add(i)
                ans = self.bfs(i, end, dic, cur*dic[start][i], visited)
                if ans!=-1.0:
                    return ans
        return -1.0
```



#### Solution-bfs-iterative

Ref: https://leetcode.com/problems/evaluate-division/discuss/88275/Python-fast-BFS-solution-with-detailed-explantion

```python
class Solution(object):
    def calcEquation(self, equations, values, queries):

        graph = {}
        
        def build_graph(equations, values):
            def add_edge(f, t, value):
                if f in graph:
                    graph[f].append((t, value))
                else:
                    graph[f] = [(t, value)]
            
            for vertices, value in zip(equations, values):
                f, t = vertices
                add_edge(f, t, value)
                add_edge(t, f, 1/value)
        
        def find_path(query):
            b, e = query
            
            if b not in graph or e not in graph:
                return -1.0
                
            q = collections.deque([(b, 1.0)])
            visited = set()
            
            while q:
                front, cur_product = q.popleft()
                if front == e:
                    return cur_product
                visited.add(front)
                for neighbor, value in graph[front]:
                    if neighbor not in visited:
                        q.append((neighbor, cur_product*value))
            
            return -1.0
        
        build_graph(equations, values)
        return [find_path(q) for q in queries]
```





### 310. Minimum Height Trees

https://leetcode.com/problems/minimum-height-trees/description/

#### Solution-bfs-worth

Ref: https://leetcode.com/problems/minimum-height-trees/discuss/76055/Share-some-thoughts

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n==1:
            return [0]
        nodes = [set() for _ in range(n)]# 这里不需要用字典，这里的list就是一个天然字典
        
        for i,j in edges:
            nodes[i].add(j)
            nodes[j].add(i)
        
        leaves = [i for i in range(n) if len(nodes[i])==1]
        while n>2:
            n-=len(leaves)
            newLeaves = []
            for leaf in leaves:
                v = nodes[leaf].pop()
                nodes[v].remove(leaf)
                if len(nodes[v])==1:
                    newLeaves.append(v)
               
            leaves = newLeaves
                
        return leaves
```



### 149. Max Points on a Line

https://leetcode.com/problems/max-points-on-a-line/

Ref: https://www.youtube.com/watch?v=7FPL7nAi9aM

![image-20200125175557438](https://tva1.sinaimg.cn/large/006tNbRwgy1gb9mtptw7dj31c00u0nfy.jpg)

```python
# based on huahua's
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        ans = 0
        
        
        for i in range(len(points)):
            sameP = 1
            otherP = 0
            dic = {}
            point1 = points[i]
            for j in range(i+1, len(points)):
                point2 = points[j]
                if point1[0]==point2[0] and point1[1]==point2[1]:
                    sameP+=1
                else:
                    slope = self.getSlope(point1, point2)
                    dic[slope] = dic.get(slope, 0)+1
                    otherP = max(otherP, dic[slope])
            ans = max(ans, sameP+otherP)
            
        return ans
    
    def getSlope(self, i, j):
        dx = i[0]-j[0]
        dy = i[1]-j[1]
        
        if dx==0:
            return (i[0], 0)
        if dy==0:
            return (0, i[1])
        
        d = self.gcd(dx, dy)
        return (dx//d, dy//d)
    
    def gcd(self, x, y):
        if y==0:
            return x
        else:
            return self.gcd(y, x%y)
```









### Knowledge

Ref: https://leetcode.com/problems/minimum-height-trees/discuss/76055/Share-some-thoughts

> (1) A tree is an undirected graph in which any two vertices are
> connected by exactly one path.
>
> (2) Any connected graph who has `n` nodes with `n-1` edges is a tree.
>
> (3) The degree of a vertex of a graph is the number of
> edges incident to the vertex.
>
> (4) A leaf is a vertex of degree 1. An internal vertex is a vertex of
> degree at least 2.
>
> (5) A path graph is a tree with two or more vertices that is not
> branched at all.
>
> (6) A tree is called a rooted tree if one vertex has been designated
> the root.
>
> (7) The height of a rooted tree is the number of edges on the longest
> downward path between root and a leaf.





#### gcd经典写法

```python
def gcd(self, x, y):
  if y==0:
    return x
  else:
    return self.gcd(y, x%y)
```





#### Trie

