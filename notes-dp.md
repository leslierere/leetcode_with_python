### 70. Climbing Stairs

https://leetcode.com/problems/climbing-stairs/

#### Solution-dp

这道题本质是 fibonacci sequence，只是F1, F2从1，1开始

**Fibonacci sequence:**

> 1，1，2，3，5，8，13，21，……。即数列满足递推公式![[公式]](https://www.zhihu.com/equation?tex=+F_%7Bn%2B2%7D+%3D+F_%7Bn%2B1%7D+%2B+F_%7Bn%7D)，（![[公式]](https://www.zhihu.com/equation?tex=F_1+%3D+F_2+%3D+1)）





### 62. Unique Paths

https://leetcode.com/problems/unique-paths/

#### Solution-dp

update row by row

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1]*m
        for i in range(n-1):
            for j in range(1, m):
                dp[j] = dp[j]+dp[j-1]
                
        return dp[-1]
```



### 63. Unique Paths II

https://leetcode.com/problems/unique-paths-ii/

#### Solution-dp

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # turn 0 to 1, turn 1 to 0, thus 0 is the obstacle now
        if obstacleGrid[0][0]==1 or obstacleGrid[-1][-1]==1:
            return 0
        noRow = len(obstacleGrid)
        noCol = len(obstacleGrid[0])
        for i in range(noRow):
            for j in range(noCol):
                obstacleGrid[i][j] = ~obstacleGrid[i][j]&1
        
        # need update the first row and first column first
        for i in range(1, noCol):
            if obstacleGrid[0][i]==0:
                continue
            obstacleGrid[0][i] = obstacleGrid[0][i-1]
            
        for j in range(1, noRow):
            if obstacleGrid[j][0]==0:
                continue
            obstacleGrid[j][0] = obstacleGrid[j-1][0]
        
        
        for i in range(1, noRow):
            for j in range(1, noCol):
                cur = obstacleGrid[i][j]
                if cur == 0:
                    continue
                obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]
                
        return obstacleGrid[-1][-1]
```





### 279. Perfect Squares

https://leetcode.com/problems/perfect-squares/description/

#### Solution-dp-worth

Ref: https://leetcode.com/articles/perfect-squares/

#### Solution-Greedy Enumeration-worth

Ref: https://leetcode.com/articles/perfect-squares/

#### Solution- Greedy + BFS-worth

Ref: https://leetcode.com/articles/perfect-squares/





### 139. Word Break

https://leetcode.com/problems/word-break/

#### Solution-memorization-slow

```python
class Solution:
    def __init__(self):
        self.notwork = set()
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        if s in wordDict:
            return True
        if s in self.notwork:
            return False
        
        if not s:
            return True
        for i in range(1, len(s)):
            if s[:i] in wordDict:
                if self.wordBreak(s[i:], wordDict):
                    return True
                else:
                    self.notwork.add(s[i:])
            else:
                self.notwork.add(s[:i])
            
                
        return False
```



#### Solution-dp-worth

Ref: https://leetcode.com/problems/word-break/discuss/43788/4-lines-in-Python

在list中查找换成set是一种常见的提速方法

```python
class Solution:
    
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        dp=[True]
        words = set(wordDict) # can speed up from 40 ms to 36ms
        for i in range(1, len(s)+1):
            dp.append(any(s[j:i] in words and dp[j] for j in range(i)))
        return dp[-1]
```



### 375. Guess Number Higher or Lower II

https://leetcode.com/problems/guess-number-higher-or-lower-ii/description/

#### Solution-dp-worth

[https://leetcode.com/problems/guess-number-higher-or-lower-ii/discuss/84766/Clarification-on-the-problem-description.-Problem-description-need-to-be-updated-!!!](https://leetcode.com/problems/guess-number-higher-or-lower-ii/discuss/84766/Clarification-on-the-problem-description.-Problem-description-need-to-be-updated-!!!)



### 312. Burst Balloons

https://leetcode.com/problems/burst-balloons/

#### Solution-dp-worth

https://leetcode.com/articles/burst-balloons/



### 322. Coin Change

https://leetcode.com/problems/coin-change/

#### Solution-dp

既然都已经想到了递推关系，就从小到大一直算下去就好了呀，也就是bottom up！也就是link里面的approach3。 或者使用递归(link里面的approach2)，top down,但要使用记忆。

Ref: https://leetcode.com/articles/coin-change/

> First, let's define:
>
> > F(S)*F*(*S*) - minimum number of coins needed to make change for amount S*S* using coin denominations \[c0…cn−1\]
>
> We note that this problem has an optimal substructure property, which is the key piece in solving any Dynamic Programming problems. In other words, the optimal solution can be constructed from optimal solutions of its subproblems. 



### Knowledge

#### any()&all()

`any()` function returns True if any item in an iterable are true, otherwise it returns False.

`any(*iterable*)`

`all()` returns true if all of the items are True (or if the iterable is empty). All can be thought of as a sequence of AND operations on the provided iterables. It also short circuit the execution i.e. stop the execution as soon as the result is known.