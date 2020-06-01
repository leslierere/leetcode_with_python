### 261. Graph Valid Tree-$

https://leetcode.com/problems/graph-valid-tree/

其实和topological sort差不多，只是多了对所有nodes是否visit过的判断

#### Solution-dfs-$

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



#### Solution-bfs

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



#### Solution-union find-worth-$

Ref: https://leetcode.com/problems/graph-valid-tree/discuss/69019/Simple-and-clean-c%2B%2B-solution-with-detailed-explanation.

```c++
class Solution {
public:
    bool validTree(int n, vector<pair<int, int>>& edges) {
        vector<int> nodes(n,0);
        for(int i=0; i<n; i++) nodes[i] = i;
        for(int i=0; i<edges.size(); i++){
            int f = edges[i].first;
            int s = edges[i].second;
            while(nodes[f]!=f) f = nodes[f];
            while(nodes[s]!=s) s = nodes[s];
            if(nodes[f] == nodes[s]) return false;
            nodes[s] = f;
        }
        return edges.size() == n-1;
    }
};
```

did@2020.5.29, 我觉得上面这个比较好，一发现一个圈就退出了, 但我的做了path compression比较好

```python
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges)!=n-1:
            return False
        
        nodes = [i for i in range(n)]
        
        for node1, node2 in edges:
            root1 = self.find(node1, nodes)
            root2 = self.find(node2, nodes)
            nodes[root1] = root2

        for i in range(n):
            self.find(i, nodes)
            
        return len(set(nodes))==1
            
    
    def find(self, position, nodes):
        if position==nodes[position]:
            return position
        nodes[position] = self.find(nodes[position], nodes)
        return nodes[position]
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



#### Solution-dfs, did@2.14, 但我的方法不够好，还是用set或者array存一下visited的部分比较好

#### Solution-dfs, union find-$

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

@20.5.29 我觉得我做的这个比较好

```python
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        grid = [[0 for i in range(n)] for j in range(m)]
        res = []
        number = 0
        visited = set()
        
        for i, j in positions:
            if (i, j) not in visited:
                
                neighbors = set()
                for x, y in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                    if 0<=x<len(grid) and 0<=y<len(grid[0]) and grid[x][y]!=0:
                        root = self.find(x, y, grid)
                        neighbors.add(root)
                if neighbors:
                    root1 = neighbors.pop()
                    grid[i][j] = root1
                    for neighbor in neighbors:
                        x, y = neighbor
                        grid[x][y] = root1
                        number-=1
                else:
                    number+=1
                    grid[i][j] = (i, j)
            visited.add((i, j))
            res.append(number)
                
        return res
                    
        
        
    def find(self, i, j, grid):
        if grid[i][j] == (i,j):
            return (i,j)
        x, y = grid[i][j]
        grid[i][j] = self.find(x, y, grid)
        return grid[i][j]
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

@2.15我觉得我第二次做的好理解一些

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
        
        newN = Node(node.val)
        valDic[value] = newN
        for neighbor in node.neighbors:
            newN.neighbors.append(self.helper(neighbor, valDic))
        return newN
```

@2.15

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = []):
        self.val = val
        self.neighbors = neighbors
"""
# for every node that we will then add all its neighbors, we marked as visited
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return node
        copy = Node(val = node.val)
        vertices = {}
        self.dfs(copy, node, vertices)
        return copy
        
        
    def dfs(self, copy, node, vertices):
        if node in vertices:
            return
        vertices[node] = copy
        for neighbor in node.neighbors:
            if neighbor not in vertices:
                neighborCopy = Node(val = neighbor.val)
                copy.neighbors.append(neighborCopy)
                self.dfs(neighborCopy, neighbor, vertices)
            else:
                copy.neighbors.append(vertices[neighbor])
```



#### Solution-dfs-iterative-worth



#### Solution-bfs-iterative

by myself@2.15

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = []):
        self.val = val
        self.neighbors = neighbors
"""
from collections import deque
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return node
        
        origins = deque()
        copies = deque()
        origins.append(node)
        root = Node(val = node.val)
        copies.append(root)
        edges = set()# don't need this
        vertices = {}
        vertices[root.val] = root
        
        while origins:
            origin = origins.popleft()
            copy = copies.popleft()
            
            for neighbor in origin.neighbors:
                if (origin.val, neighbor.val) not in edges:# don't need this
                    edges.add((origin.val, neighbor.val))# don't need this

                    if neighbor.val not in vertices:
                        copyNeighbor = Node(val=neighbor.val)
                        vertices[neighbor.val] = copyNeighbor
                        copy.neighbors.append(copyNeighbor)

                        origins.append(neighbor)
                        copies.append(copyNeighbor)
                    else:
                        copy.neighbors.append(vertices[neighbor.val])
                        
        return root                
```

但其实没必要维护两个queue，用原始的node作为key就好, 而且并不需要对vertix是否visite过做判断，因为queue里面总是存的新建的node

Ref: https://leetcode.com/problems/clone-graph/discuss/42314/Python-solutions-(BFS-DFS-iteratively-DFS-recursively).e

```python
def cloneGraph1(self, node):
    if not node:
        return 
    nodeCopy = UndirectedGraphNode(node.label)
    dic = {node: nodeCopy}
    queue = collections.deque([node])
    while queue:
        node = queue.popleft()
        for neighbor in node.neighbors:
            if neighbor not in dic: # neighbor is not visited
                neighborCopy = UndirectedGraphNode(neighbor.label)
                dic[neighbor] = neighborCopy
                dic[node].neighbors.append(neighborCopy)
                queue.append(neighbor)
            else:
                dic[node].neighbors.append(dic[neighbor])
    return nodeCopy
```





### 399. Evaluate Division-$

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





### 310. Minimum Height Trees-$

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



### 149. Max Points on a Line-$$

https://leetcode.com/problems/max-points-on-a-line/

Ref: https://www.youtube.com/watch?v=7FPL7nAi9aM

![image-20200125175557438](https://tva1.sinaimg.cn/large/006tNbRwgy1gb9mtptw7dj31c00u0nfy.jpg)

```python
# based on huahua's
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        ans = 0
        
        
        for i in range(len(points)):
            sameP = 1 # first add the point itself
            otherP = 0 # 局部最优解
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



