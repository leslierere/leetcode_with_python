### 207. Course Schedule

https://leetcode.com/problems/course-schedule/

#### Solution-dfs-need refine

Ref: https://www.coursera.org/learn/algorithms-graphs-data-structures/home/welcome

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        visited = [False for _ in range(numCourses)]
        
        for start, end in prerequisites:
            graph[start].append(end)
            
        return self.dfs(list(range(numCourses)), visited, graph)
                
                
    def dfs(self, nums, visited, graph):
        while nums:
            i = nums.pop()
            if not visited[i]:# we only need do dfs on those not visited
                if graph[i]:
                    if self.dfs(graph[i], visited, graph):
                        if visited[i]:
                            return False
                        else:
                            visited[i] = True
                    else:
                        return False
                else: # the sink vertix
                    visited[i] = True
                
        return True
    
    # every time we meet a vertix, we delete it, but it will finally be marked as visited, if it doesn't have next vertix(it may be passed so deleted or it is a sink vertix), then we just mark this visited. If it has next vertix, we do dfs, if in the dfs, it is marked as True, it means there is a circle or the dfs find circle itself, we can immediately return True, else we just mark this visited
```





### 210. Course Schedule II

https://leetcode.com/problems/course-schedule-ii/description/

#### Solution-based on the above one-need refine

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(numCourses)]
        visited = [-1 for _ in range(numCourses)]
        path = [0]*numCourses
        for start, end in prerequisites:
            graph[start].append(end)
        self.order = 0
            
        if self.dfs(list(range(numCourses)), visited, graph):
            for index, value in enumerate(visited):
                path[value] = index
            return path
        else:
            return []
    

    def dfs(self, nums, visited, graph):
        while nums:
            i = nums.pop()
            if visited[i] == -1:# we only need do dfs on those not visited
                if graph[i]:
                    if self.dfs(graph[i], visited, graph):
                        if visited[i]!=-1:
                            return False
                        else:
                            visited[i] = self.order
                            self.order+=1
                    else:
                        return False
                else: # the sink vertix
                    visited[i] = self.order
                    self.order+=1
                
        return True
```





### 269. Alien Dictionary

https://leetcode.com/problems/alien-dictionary/description/

Ref: https://leetcode.com/problems/alien-dictionary/discuss/70115/3ms-Clean-Java-Solution-(DFS)

Borrow the idea on how to build graph

```python
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        if not words:
            return ""
        if len(words)==1:
            return words[0]
        
        letters = [-2]*26
        edges = [[] for _ in range(26)]
        res = [""]*26
        
        self.buildGraph(words, letters, edges)
        
        self.order = letters.count(-1)-1
        
        
        
        for i in range(26):
            if letters[i]!=-2: # the char exist
                char = chr(i+97)
                if not self.dfs(char, letters, edges):
                    return ""
                res[letters[i]] = char
                
        return "".join(res)
                
       
        
    def dfs(self, char, letters, edges):
        pos = ord(char)-97
        if letters[pos]==-1: # it exists but not visited
            while edges[pos]:
                cur = edges[pos].pop()
                if self.dfs(cur, letters, edges):
                    if letters[pos]>=0:# it is visited during the dfs
                        return False
                    
                else:
                    return False
            else:
                letters[pos] = self.order
                self.order-=1
                
        return True
            
            
        
        
    def buildGraph(self, words, letters, edges):
        for i in range(1, len(words)):
            word1 = words[i-1]
            word2 = words[i]
            length = min(len(word1), len(word2))
            for j in range(length):
                if word1[j]!=word2[j]:
                    edges[ord(word1[j])-97].append(word2[j])
                    break
                letters[ord(word1[j])-97] = -1
            for remain in range(j, len(word1)):
                letters[ord(word1[remain])-97] = -1
            for remain in range(j, len(word2)):
                letters[ord(word2[remain])-97] = -1
                
```

