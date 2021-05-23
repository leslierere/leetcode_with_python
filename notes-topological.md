### 207. Course Schedule

https://leetcode.com/problems/course-schedule/

#### Solution-dfs

我感觉第一次写得更好，考虑在dfs完了以后当前course有没有被visit过

Ref: https://www.coursera.org/learn/algorithms-graphs-data-structures/home/welcome

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        edges = [list() for i in range(numCourses)]
        visited = [False]*numCourses
        
        for i, j in prerequisites:
            edges[i].append(j)
        
        for i in range(numCourses):
          # we only need do dfs on those not visited， to speed it up
            if not visited[i] and not self.dfs(i, visited, edges):
                return False
            
        return True
                
    def dfs(self, vertix, visited, edges):
        if not visited[vertix]: # we only need do dfs on those not visited
            
            while edges[vertix]:
                neighbor = edges[vertix].pop()
                if not self.dfs(neighbor, visited, edges):
                    return False

                if visited[vertix]: 
                    return False

            visited[vertix] = True
            
        return True
                
# Every time we meet a vertix it will finally be marked as visited. If it doesn't have next vertix(it may be that all its neighbors are visited or it is a sink vertix), then it comes to the time to mark this visited.
# However, if it has next vertix, we do dfs. If in the dfs, this vertix is marked as visited, it means there is a circle or the dfs find circle itself, we can immediately return False, else we just mark this visited
```

@3.29

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        courses = collections.defaultdict(list)
        visited = [0]*numCourses
    
        
        for pre, post in prerequisites:
            courses[pre].append(post)
            
         
        return all([self.helper(i, visited, courses) for i in range(numCourses) if not visited[i]])# we only need do dfs on those not visited，ans this helpes to speed it up
    
    def helper(self, course, visited, courses):
        if course not in courses:
            return True
        if visited[course]:
            return False
        
        visited[course] = 1
        
        while courses[course]:
            if not self.helper(courses[course].pop(), visited, courses):
                return False
        
        courses.pop(course)
            
        
        return True
```



#### Solution-bfs-worth

```c++
bool canFinish(int n, vector<pair<int, int>>& pre) {
    vector<vector<int>> adj(n, vector<int>());
    vector<int> degree(n, 0);
    for (auto &p: pre) {
        adj[p.second].push_back(p.first);
        degree[p.first]++;
    }
    queue<int> q;
    for (int i = 0; i < n; i++)
        if (degree[i] == 0) q.push(i);
    while (!q.empty()) {
        int curr = q.front(); q.pop(); n--;
        for (auto next: adj[curr])
            if (--degree[next] == 0) q.push(next);
    }
    return n == 0;
}
```



### 210. Course Schedule II-$

https://leetcode.com/problems/course-schedule-ii/description/

#### Solution@2.18, based on the above idea

```python
from collections import defaultdict
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        res = []
        edges = defaultdict(list)
        visited = [False for i in range(numCourses)]
        
        for i, j in prerequisites:
            edges[i].append(j)
            
        for i in range(numCourses):
            if not self.dfs(i, edges, res, visited):
                return []
            
        return res
            
            
    def dfs(self, vertix, edges, res, visited):
        if not visited[vertix]:
            while edges[vertix]:
                neighbor = edges[vertix].pop()
                if not self.dfs(neighbor, edges, res, visited):
                    return False
                if visited[vertix]:
                    return False
            visited[vertix] = True # @3.29考虑在dfs时当前course有没有被visit过
            res.append(vertix)
            
        return True
```



#### Solution-@2.8-太烂了

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





### 269. Alien Dictionary-c comments

https://leetcode.com/problems/alien-dictionary/description/

Ref: https://leetcode.com/problems/alien-dictionary/discuss/70115/3ms-Clean-Java-Solution-(DFS)

需要注意的几个点：

* if only one word is given
* take care of letters that don't exist, this can be combined with a visited array, use different number to mark visited and existed, modify on top of solution@2.18
* some letters may not have order

* first word is "abc", next is "ab"

```python
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        if not words:
            return ""
        if len(words)<2:
            return words[0]
        chars = collections.defaultdict(list)
        charSet = set()
        for i in range(len(words)-1):
            word1 = words[i]
            word2 = words[i+1]
            
            j = 0
            while j<len(word1) and j<len(word2) and word1[j]==word2[j]:
                charSet.add(word1[j])
                j+=1
            if j==len(word2) and j<len(word1):
                return ""
            if j<len(word1) and j<len(word2):
                chars[word2[j]].append(word1[j])
            for k in range(j, len(word1)):
                charSet.add(word1[k])
                
            for k in range(j, len(word2)):
                charSet.add(word2[k])
                
        order=['']
        visited = set()
        for char in charSet:
            if char not in visited and not self.dfs(char, chars, visited, order):
                return ""
        return order[0]
            
            
    def dfs(self, char, chars, visited, order):
        if char not in visited:
            while chars[char]:
                neighbor = chars[char].pop()
                if not self.dfs(neighbor, chars, visited, order):
                    return False
                if char in visited:
                    return False
            visited.add(char)
            order[0]+=char
        return True
        
         
# @2.18, 这里buildgraph比较好
    def buildGraph(self, words):
        edges = defaultdict(list)
        letters = set()
        for i in range(len(words)-1):
            word1 = words[i]
            word2 = words[i+1]
            
            length = min(len(word1), len(word2))
            j = 0
            while j<length:
                if word1[j]==word2[j]:
                    letters.add(word1[j])
                    j+=1
                else:
                    edges[word2[j]].append(word1[j])
                    break
            for letter in word1[j:]:
                letters.add(letter)
            for letter in word2[j:]:
                letters.add(letter)
                
                
        return edges, letters
```





Borrow the idea on how to build graph@2.8

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

