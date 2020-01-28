### 200. Number of Islands

https://leetcode.com/problems/number-of-islands/

#### Solution-dfs

Ref: https://leetcode.com/problems/number-of-islands/discuss/56340/Python-Simple-DFS-Solution

其实就可以想成tree traversal的dfs，只是print变成了mark

```python
def numIslands(self, grid):
    if not grid:
        return 0
        
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                self.dfs(grid, i, j)
                count += 1
    return count

def dfs(self, grid, i, j):
    if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != '1':
        return
    grid[i][j] = '#'
    self.dfs(grid, i+1, j)
    self.dfs(grid, i-1, j)
    self.dfs(grid, i, j+1)
    self.dfs(grid, i, j-1)
```





### 286. Walls and Gates-$

https://leetcode.com/problems/walls-and-gates/description/

#### Solution-bfs-recursive-worth

他这个是从gate出发

Ref: https://leetcode.com/problems/walls-and-gates/discuss/72746/My-short-java-solution-very-easy-to-understand



#### Solution-bfs-queue-worth

Ref: https://leetcode.com/problems/walls-and-gates/discuss/72753/6-lines-O(mn)-Python-BFS

```python
def wallsAndGates(self, rooms):
    q = [(i, j) for i, row in enumerate(rooms) for j, r in enumerate(row) if not r]
    for i, j in q:
        for I, J in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
            if 0 <= I < len(rooms) and 0 <= J < len(rooms[0]) and rooms[I][J] > 2**30:
                rooms[I][J] = rooms[i][j] + 1
                q += (I, J),
```



### 130. Surrounded Regions-想想思路

https://leetcode.com/problems/surrounded-regions/

#### Solution-bfs-recursive-by myself

没必要建一个marked表

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return
        # 先找到边上的mark好，再把所有0翻过来
        rows = len(board)
        cols = len(board[0])
        
        if rows<3 or cols<3:
        		return
                
        marked = [[0 for i in range(cols)] for j in range(rows)]
        
        # start from top and bottom boarder
        for i in [0, rows-1]:
            for j in range(cols):
                if board[i][j]=='O':
                    self.dfs(i, j, board, marked)
                    
        # start from left and right boarder            
        for i in range(1, rows-1):
            for j in [0, cols-1]:
                if board[i][j]=='O':
                    self.dfs(i, j, board, marked)
                    
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if board[i][j]=='O' and not marked[i][j]:
                    board[i][j]="X"
        
        
    def dfs(self, i, j, board, marked):
        if i<0 or j<0 or i>=len(board) or j>= len(board[0]):
            return
        if board[i][j]=='O' and not marked[i][j]:
            marked[i][j]=1
            self.dfs(i-1, j, board, marked)
            self.dfs(i+1, j, board, marked)
            self.dfs(i, j-1, board, marked)
            self.dfs(i, j+1, board, marked)
```



#### Solution-bfs-queue-worth

Ref: https://leetcode.com/problems/surrounded-regions/discuss/41652/Python-short-BFS-solution.

https://leetcode.com/problems/surrounded-regions/discuss/41630/9-lines-Python-148-ms

```python
def solve(self, board):
    queue = collections.deque([])
    for r in xrange(len(board)):
        for c in xrange(len(board[0])):
            if (r in [0, len(board)-1] or c in [0, len(board[0])-1]) and board[r][c] == "O":
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        if 0<=r<len(board) and 0<=c<len(board[0]) and board[r][c] == "O":
            board[r][c] = "D"
            queue.append((r-1, c)); queue.append((r+1, c))
            queue.append((r, c-1)); queue.append((r, c+1))
        
    for r in xrange(len(board)):
        for c in xrange(len(board[0])):
            if board[r][c] == "O":
                board[r][c] = "X"
            elif board[r][c] == "D":
                board[r][c] = "O"
            #这里按照stephan的可以改成
            # board[r][c] ='OX'[c == 'O']
```





### 339. Nested List Weight Sum-$

https://leetcode.com/problems/nested-list-weight-sum/description/

#### Solution-dfs

https://leetcode.com/articles/nested-list-weight-sum/

```java
public int depthSum(List<NestedInteger> nestedList) {
    return depthSum(nestedList, 1);
}

public int depthSum(List<NestedInteger> list, int depth) {
    int sum = 0;
    for (NestedInteger n : list) {
        if (n.isInteger()) {
            sum += n.getInteger() * depth;
        } else {
            sum += depthSum(n.getList(), depth + 1);
        }
    }
    return sum;
}
```





### 364. Nested List Weight Sum II-$

https://leetcode.com/problems/nested-list-weight-sum-ii/

#### Solution-bfs

Ref: https://leetcode.com/problems/nested-list-weight-sum-ii/discuss/83641/No-depth-variable-no-multiplication

```java
public int depthSumInverse(List<NestedInteger> nestedList) {
    int unweighted = 0, weighted = 0;
    while (!nestedList.isEmpty()) {
        List<NestedInteger> nextLevel = new ArrayList<>();
        for (NestedInteger ni : nestedList) {
            if (ni.isInteger())
                unweighted += ni.getInteger();
            else
                nextLevel.addAll(ni.getList());
        }
        weighted += unweighted;
        nestedList = nextLevel;
    }
    return weighted;
}
```



#### Solution-dfs

Ref: https://leetcode.com/problems/nested-list-weight-sum-ii/discuss/114195/Java-one-pass-DFS-solution-mathematically

> Actually got this from the tagged company phone interview after hinted by the interviewer.
> The idea is to deduct number depth - level times.
> For example, 1x + 2y + 3z = (3 + 1) * (x + y + z) - (3x + 2y + z);
> So we can convert this problem to Nested List Weight Sum I and just record max depth and flat sum at the same time.



### 127. Word Ladder-$

https://leetcode.com/problems/word-ladder/description/

#### Solution-bfs

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        d = collections.defaultdict(list)
        for w in wordList:
            for i in range(len(w)):
                s = w[:i]+'_'+w[i+1:]#不要换成a-z
                d[s].append(w)
        q = collections.deque([(beginWord,1)])
        visited = set()
        while q:
            w, l = q.popleft()
            for i in range(len(w)):
                w1 = w[:i]+'_'+w[i+1:]
                for w2 in d[w1]:
                    if w2==endWord: return l+1
                    if w2 not in visited: 
                        visited.add(w2)
                        q.append((w2, l+1))
        return 0
```



### 126. Word Ladder II-$

https://leetcode.com/problems/word-ladder-ii/description/

#### Solution-bfs

https://leetcode.com/problems/word-ladder-ii/discuss/40434/C%2B%2B-solution-using-standard-BFS-method-no-DFS-or-backtracking

> The line `wordList.insert(endWord);` should be deleted in latest problem, otherwise it will get wrong answer for test case

```c++
class Solution {
public:
    vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
        //very interesting problem
        //It can be solved with standard BFS. The tricky idea is doing BFS of paths instead of words!
        //Then the queue becomes a queue of paths.
        vector<vector<string>> ans;
        queue<vector<string>> paths;
        // wordList.insert(endWord);
        paths.push({beginWord});
        int level = 1;
        int minLevel = INT_MAX;
        
        //"visited" records all the visited nodes on this level
        //these words will never be visited again after this level 
        //and should be removed from wordList. This is guaranteed
        // by the shortest path.
        unordered_set<string> visited; 
        
        while (!paths.empty()) {
            vector<string> path = paths.front();
            paths.pop();
            if (path.size() > level) {
                //reach a new level
                for (string w : visited) wordList.erase(w);
                visited.clear();
                if (path.size() > minLevel)
                    break;
                else
                    level = path.size();
            }
            string last = path.back();
            //find next words in wordList by changing
            //each element from 'a' to 'z'
            for (int i = 0; i < last.size(); ++i) {
                string news = last;
                for (char c = 'a'; c <= 'z'; ++c) {
                    news[i] = c;
                    if (wordList.find(news) != wordList.end()) {
                        //判断在不在里面， True就在
                    //next word is in wordList
                    //append this word to path
                    //path will be reused in the loop
                    //so copy a new path
                        vector<string> newpath = path;
                        newpath.push_back(news);
                        visited.insert(news);
                        if (news == endWord) {
                            minLevel = level;
                            ans.push_back(newpath);
                        }
                        else
                            paths.push(newpath);
                    }
                }
            }
        }
        return ans;
        
    }
};
```



By myself@1.27

延续127的思路

```python
from collections import defaultdict, deque
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        dic = defaultdict(set)
        for word in wordList:
            for i in range(len(word)):
                key = word[:i]+"_"+word[i+1:]
                dic[key].add(word)
                
        queue = deque()
        queue.append([beginWord])
        res=[]
        wordSet = set(wordList)
        
        while queue:
            length = len(queue)
            toDelete = set()
            
            for _ in range(length):
                first = queue.popleft()
                if first[-1]==endWord:
                    res.append(first)
                if not res:
                    for i in range(len(first[-1])):
                        var = first[-1][:i]+"_"+first[-1][i+1:]

                        if var in dic:
                            for word in dic[var]:
                                if word in wordSet:
                                    queue.append(first+[word])
                                    toDelete.add(word)
            if res:
                return res                        
            for w in toDelete:
                wordSet.remove(w)
            
        return res
```





### 51. N-Queens-$

https://leetcode.com/problems/n-queens/

queens can attack other queen in the same row, same column and the diagonal

#### Solution-dfs-worth

Ref: https://leetcode.com/problems/n-queens/discuss/19971/Python-recursive-dfs-solution-with-comments.

```python
def solveNQueens(self, n):
    res = []
    self.dfs([-1]*n, 0, [], res)
    return res
 
# nums is a one-dimension array, like [1, 3, 0, 2] means
# first queen is placed in column 1, second queen is placed
# in column 3, etc.
def dfs(self, nums, index, path, res):
    if index == len(nums):
        res.append(path)
        return  # backtracking
    for i in xrange(len(nums)):
        nums[index] = i
        if self.valid(nums, index):  # pruning
            tmp = "."*len(nums)
            self.dfs(nums, index+1, path+[tmp[:i]+"Q"+tmp[i+1:]], res)

# check whether nth queen can be placed in that column
def valid(self, nums, n):# n 是刚加进去的row no
    for i in xrange(n):
        if abs(nums[i]-nums[n]) == n -i or nums[i] == nums[n]:
            return False
    return True
```



By myself@1.27

```python
from collections import deque
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        nums = [0]*n
        lefts = deque(range(n))
        res = []
        self.dfs(nums, res, 0, lefts)
        return res
        
    def dfs(self, nums, res, row, lefts):
        if not lefts:
            res.append(self.helper(nums))
            return
                      
        length = len(lefts)
        for _ in range(length):
            num = lefts.popleft()
            if self.isValid(nums, num, row):
                nums[row] = num
                self.dfs(nums, res, row+1, lefts)
            nums[row] = 0
            lefts.append(num)
                     
                      
    def isValid(self, nums, num, row):
        for i in range(row):
            if abs(i-row)==abs(nums[i]-num):
                return False
        return True
                      
    def helper(self, nums):
        res = []
        string = "."*len(nums)
        for i in nums:
            res.append(string[:i]+"Q"+string[i+1:])
        return res                      
```





### 52. N-Queens II

https://leetcode.com/problems/n-queens-ii/

上一题会这一题也知道了





