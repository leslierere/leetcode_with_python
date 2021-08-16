### 261. Graph Valid Tree-$bfs和union find做做

https://leetcode.com/problems/graph-valid-tree/

@6.21使用dfs可以用topological sort的思路，只要edges数满足n-1且没有圈，就不会有问题；而这里dfs和bfs的做法是，通过一个node一定可以把所有其他node visit到，感觉是更自然的思路

#### Solution-dfs-$

就是以一个为起点，每一个都只会恰好visit一遍

当然visited列表也可以用set替换

Ref: https://www.cnblogs.com/grandyang/p/5257919.html

```python

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges)!=n-1: # examine the edge number first
            return False
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



#### Solution-bfs

Ref: https://www.cnblogs.com/grandyang/p/5257919.html

```python
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges)!=n-1: # examine the edge number first
            return False
        nodes = collections.defaultdict(set)
        for node1, node2 in edges:
            nodes[node1].add(node2)
            nodes[node2].add(node1)
        
        visited = [0 for _ in range(n)]
        queue = collections.deque([0])
        
        while queue:
            node = queue.popleft()
            if visited[node]:
                return False
            visited[node] = 1
            while nodes[node]:
                neighbor = nodes[node].pop()
                nodes[neighbor].remove(node)
                queue.append(neighbor)
                
        return all(visited)
```



#### Solution-disjoint sets-union find-worth-$$

Ref: https://leetcode.com/problems/graph-valid-tree/discuss/69019/Simple-and-clean-c%2B%2B-solution-with-detailed-explanation.

```python
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges)!=n-1: # examine the edge number first
            return False
        
        nodes = [i for i in range(n)]
        
        for node1, node2 in edges:
            root1 = self.find(node1, nodes) 
            root2 = self.find(node2, nodes)
            
            if root1== root2: # cool
                return False
            
            nodes[root1] = root2
            
        return True
            
            
    def find(self, node, nodes):# do path compression
        if node==nodes[node]:
            return node
        
        nodes[node] = self.find(nodes[node], nodes)
        return nodes[node]
```





### 323. Number of Connected Components in an Undirected Graph-$

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



#### Solution-dfs, did@2.14, 但我的方法不够好，还是用set或者array存一下visited的部分比较好, 然后其实不需要remove neighbor

did@21.7.17

```python
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        count = 0
        
        neighbors = collections.defaultdict(set)
        for node1, node2 in edges:
            neighbors[node1].add(node2)
            neighbors[node2].add(node1)
            
        visited = [False for i in range(n)]
        
        for node in range(n):
            if not visited[node]:
                self.dfs(node, neighbors, visited)
                count+=1
                
        return count
                
                
    def dfs(self, node, neighbors, visited):
        if visited[node]:
            return
        
        visited[node] = True
        # while neighbors[node]:
        for neighbor in neighbors[node]:
            # neighbor = neighbors[node].pop()
            # neighbors[neighbor].remove(node)
            self.dfs(neighbor, neighbors, visited)
```



#### Solution-dfs, union find-$,主要想一下怎么算出岛的数量

Ref: https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/discuss/77574/Easiest-2ms-Java-Solution

```java
public int countComponents(int n, int[][] edges) {
    int[] roots = new int[n];
    for(int i = 0; i < n; i++) roots[i] = i; 

    for(int[] e : edges) {
        int root1 = find(roots, e[0]);
        int root2 = find(roots, e[1]);
        if(root1 != root2) {      
            roots[root1] = root2;  // union
            n--; //太聪明了！
        }
    }
    return n;
}

public int find(int[] roots, int id) {
    while(roots[id] != id) {
        roots[id] = roots[roots[id]];  // optional: path compression
        id = roots[id];
    }
    return id;
}
```





### 305. Number of Islands II-$$

https://leetcode.com/problems/number-of-islands-ii/description/

#### Solution-union find/disjoint sets

@20.6.22 我觉得我做的这个比较好, @21,3,12咋觉得这个也不是很清楚， 看下面的

```python
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        islands = [[0 for i in range(n)] for j in range(m)]
        res = []
        number = 0
        
        for i, j in positions:
            if islands[i][j]==0:
                neighbors = set()
                for x,y in [(i+1, j), (i-1, j), (i,j+1), (i,j-1)]:
                    if 0<=x and x<m and 0<=y and y<n and islands[x][y]!=0:
                        neighbors.add(self.find(islands, x, y))
                        
                if len(neighbors)==0:
                    islands[i][j] = (i,j)
                    number+=1
                    res.append(number)
                else:
                    islands[i][j]=neighbors.pop()
                    while neighbors:
                        neighbor = neighbors.pop()
                        if islands[i][j]!=neighbor:
                            rootI, rootJ = islands[i][j]
                            islands[rootI][rootJ] = neighbor
                            islands[i][j] = neighbor
                            number-=1
                    res.append(number)
            else:
                res.append(number)
                    
        return res

                    
            
            
    def find(self, islands, i, j): #return the root
        if islands[i][j]==(i,j):
            return (i,j)
        rootI,rootJ = islands[i][j]
        islands[i][j] = self.find(islands,rootI,rootJ)
        return islands[i][j]
```

@21.3.12这个清楚点

```python
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        graph = [[0 for j in range(n)] for i in range(m)]
        number = 0
        result = []
        
        for i, j in positions:
            
            if graph[i][j]==0: # may add node that has been added before
                number+=1
                graph[i][j] = (i, j)
                for neighbor_i, neighbor_j in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                    if neighbor_i>=0 and neighbor_i<m and neighbor_j>=0 and neighbor_j<n and graph[neighbor_i][neighbor_j]!=0:
                        root_i, root_j = self.find(neighbor_i, neighbor_j, graph)
                        my_root = self.find(i, j, graph)
                        if my_root != (root_i, root_j):
                            graph[my_root[0]][my_root[1]] = (root_i, root_j)
                            number-=1

            result.append(number)
            
        return result
            
    def find(self, i, j, graph):
        if graph[i][j] == (i, j):
            return graph[i][j]
        neigjbor_i, neighbor_j = graph[i][j]
        graph[i][j] = self.find(neigjbor_i, neighbor_j, graph)
        return graph[i][j]
```



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



### 133. Clone Graph-$

https://leetcode.com/problems/clone-graph/description/

#### Solution-dfs-recursive, by myself

did@21.7.23

写的最好的一次

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return node
        nodes = dict()
        return self.dfs(node, nodes)
        
        
    def dfs(self, node, nodes):
        if node.val in nodes:
            return nodes[node.val]
        
        cpy_node = Node(val = node.val)
        nodes[node.val] = cpy_node
        for neighbor in node.neighbors:
            nbr_cpy = self.dfs(neighbor, nodes)
            cpy_node.neighbors.append(nbr_cpy)
            
        return cpy_node
```



#### Solution-dfs-iterative-worth

did@2021.3.13

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if node is None:
            return None
        
        stack = [node]
        visited = dict()
        visited[node.val] = Node(val = node.val)
        
        while stack:
            node = stack.pop()
            for neighbor in node.neighbors:
                if neighbor.val not in visited:
                    neighbor_cpy = Node(neighbor.val)
                    visited[neighbor.val] = neighbor_cpy
                    stack.append(neighbor)
                visited[node.val].neighbors.append(visited[neighbor.val])
                    
        return visited[1]
                    
```



#### Solution-bfs-iterative

Ref: https://leetcode.com/problems/clone-graph/discuss/42314/Python-solutions-(BFS-DFS-iteratively-DFS-recursively).e

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return node
        root = Node(val=node.val)
        visited = {node.val:root}
        queue = collections.deque([(node, visited[node.val])])
        
        while queue:
            node, cpy_node = queue.popleft()
            for nbr in node.neighbors:
                if nbr.val not in visited:
                    visited[nbr.val] = Node(val=nbr.val)
                    queue.append((nbr, visited[nbr.val]))
                visited[node.val].neighbors.append(visited[nbr.val])
                
        return root
```





### 399. Evaluate Division-$

https://leetcode.com/problems/evaluate-division/description/

#### Solution-dfs-@2021.3.14

Pre in dfs is not necessary

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = collections.defaultdict(dict)
        for i in range(len(values)):
            value = values[i]
            node1 = equations[i][0]
            node2 = equations[i][1]
            
            graph[node1][node2] = value
            graph[node2][node1] = 1/value
        
        result = []
        for node1, node2 in queries:
            result.append(self.dfs(node1, node2, None, graph, set()))
        return result
            
        
    def dfs(self, node, dest, pre, graph, visited):
        if node not in graph or dest not in graph:
            return -1
        if node==dest:
            return 1
        if dest in graph[node]:
            return graph[node][dest]
        if node in visited:
            return -1
        
        next_val = -1
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor != pre:
                next_val = self.dfs(neighbor, dest, node, graph, visited)
                if next_val!=-1:
                    return next_val * graph[node][neighbor]
                
        return next_val
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





### 310. Minimum Height Trees-$$

https://leetcode.com/problems/minimum-height-trees/description/

#### Solution-bfs-worth

Ref: https://leetcode.com/problems/minimum-height-trees/discuss/76055/Share-some-thoughts

这里的做法比我5.22做的简单

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
                if len(nodes[v])==1:# 因为有这个也不用担心一个node被多次加入，因为只有剩一个连接点的时候会被加入
                    newLeaves.append(v)
               
            leaves = newLeaves
                
        return leaves
```



### 208. Implement Trie (Prefix Tree)-easy

#### Solution-trie

https://leetcode.com/problems/implement-trie-prefix-tree/

Ref: coursera, https://leetcode.com/problems/implement-trie-prefix-tree/discuss/58989/My-python-solution

```python
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.isWord = False
        self.children = dict()

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        
        
    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        node.isWord = True
            

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for letter in word:
            if letter not in node.children:
                return False
            node = node.children[letter]
            
        return node.isWord
        

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for letter in prefix:
            if letter not in node.children:
                return False
            node = node.children[letter]
        return True
        

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```



### 211. Add and Search Word - Data structure design

https://leetcode.com/problems/add-and-search-word-data-structure-design/description/

#### Solution-dfs

Ref: https://leetcode.com/problems/add-and-search-word-data-structure-design/discuss/59725/Python-easy-to-follow-solution-using-Trie.

用defaultdict可以变快

```python
class TrieNode:
    def __init__(self):
        self.isWord = False
        self.children = {}

class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        node = self.root
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]
        node.isWord = True
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        
        def dfs(node, word):
            
            for i in range(len(word)):
                letter = word[i]
                if letter!=".":
                    if letter not in node.children:
                        return False
                    node = node.children[letter]
                else:
                    for child in node.children.values():
                        if dfs(child, word[i+1:]):
                            return True
                    return False
            return node.isWord
        
        
        return dfs(self.root, word)

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### Solution-bfs@5.30

```python
    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        # node = self.root
        queue = collections.deque()
        queue.append(self.root)
        for letter in word:
            length = len(queue)
            for _ in range(length):
                node = queue.popleft()
                if letter == '.':
                    for child in node.children:
                        queue.append(node.children[child])
                elif letter in node.children:
                    queue.append(node.children[letter])
            if len(queue)==0:
                return False
        for node in queue:
            if node.isWord:
                return True
        return False
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```





### 212. Word Search II-$

https://leetcode.com/problems/word-search-ii/description/

#### Solution-trie, backtrack

@2021.3.18看下面那个就好，但不知道为啥突然变巨慢。。。

@2.16按照自己的思路, 但是就按照下面那个解法，其实并不需要一个visited表，直接在原表上面改后面复原即可

```python
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.word = None
        self.children = defaultdict(TrieNode)
        
class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children[char]
        node.word = word

class Solution:
        
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        trie = Trie()
        for word in words:
            trie.insert(word)
        
        rows = len(board)
        cols = len(board[0])
        res = set()
        visited = [[False for i in range(cols)] for j in range(rows)]
        
        
        def helper(node, x, y):
            # if not node:
            #     return
            if board[x][y] not in node.children:
                return
            else:
                child = node.children[board[x][y]]
                if child.word:
                    res.add(child.word)
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    x1 = x+dx
                    y1 = y+dy
                    if x1>=0 and x1<rows and y1>=0 and y1<cols and not visited[x1][y1]:
                        visited[x1][y1] = True
                        helper(child, x1, y1)
                        visited[x1][y1] = False
        
        
        for i in range(rows):
            for j in range(cols):
                visited[i][j] = True
                helper(trie.root, i, j)
                visited[i][j] = False
                    
        return list(res)
```







Ref: https://leetcode.com/problems/word-search-ii/discuss/59780/Java-15ms-Easiest-Solution-(100.00)

> Intuitively, start from every cell and try to build a word in the dictionary. `Backtracking (dfs)` is the powerful way to exhaust every possible ways. Apparently, we need to do `pruning` when current character is not in any word.
>
> 
>
> 1. How do we instantly know the current character is invalid? `HashMap`?
> 2. How do we instantly know what's the next valid character? `LinkedList`?
> 3. But the next character can be chosen from a list of characters. `"Mutil-LinkedList"`?
>
> 
>
> Combing them, `Trie` is the natural choice. Notice that:

```python
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = self.buildTrie(words)
        res = []
        
        self.rows = len(board)
        self.cols = len(board[0])
        for i in range(self.rows):
            for j in range(self.cols):
                self.dfs(i, j, root, board, res)
        return res
                
    def dfs(self, i, j, node, board, res):
        if i<0 or i>=self.rows or j<0 or j>=self.cols:
            return
        
        cur = board[i][j]
        
        if cur =='#' or cur not in node.children:
            return
        
        node = node.children[cur]
        
        if node.word: # 必须要当前就检查，不然跳到下一个可能超过边界了
            res.append(node.word)
            node.word = None # de-duplicate
        
        
        board[i][j] = "#"
        for x, y in [(i+1, j), (i-1,j), (i, j+1), (i, j-1)]:
            self.dfs(x, y, node, board, res)
        board[i][j] = cur
        
        
    
    def buildTrie(self, words):
        root = TrieNode()
        for w in words:
            node = root
            for letter in w:
                if letter not in node.children:
                    node.children[letter] = TrieNode()
                node = node.children[letter]
            node.word = w
        return root
        
        
class TrieNode:
    def __init__(self):
        self.word = None
        self.children = dict()
```





### 1761. Minimum Degree of a Connected Trio in a Graph

https://leetcode.com/problems/minimum-degree-of-a-connected-trio-in-a-graph/

#### Solution

Ref: https://leetcode.com/problems/minimum-degree-of-a-connected-trio-in-a-graph/discuss/1064616/Python-3-simple-brute-force-solution

