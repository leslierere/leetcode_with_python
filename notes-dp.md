1.14

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

#### Solution-dp-worth

既然都已经想到了递推关系，就从小到大一直算下去就好了呀，也就是bottom up！也就是link里面的approach3。 或者使用递归(link里面的approach2)，top down,但要使用记忆。

Ref: https://leetcode.com/articles/coin-change/

> First, let's define:
>
> > F(S)*F*(*S*) - minimum number of coins needed to make change for amount S*S* using coin denominations \[c0…cn−1\]
>
> We note that this problem has an optimal substructure property, which is the key piece in solving any Dynamic Programming problems. In other words, the optimal solution can be constructed from optimal solutions of its subproblems. 



1.15

### 256. Paint House

https://leetcode.com/problems/paint-house/discuss/?currentPage=1&orderBy=most_votes&query=

#### Solution-dp-worth-O(n)

Ref: https://leetcode.com/problems/paint-house/discuss/68203/Share-my-very-simple-Java-solution-with-explanation.

#### Solution-dp-worth-improvement-O(1)

Ref: [https://leetcode.com/problems/paint-house/discuss/68256/Python-solutions-with-different-space-usages.](https://leetcode.com/problems/paint-house/discuss/68256/Python-solutions-with-different-space-usages.)

```python
# O(1) space, shorter version, can be applied 
# for more than 3 colors
def minCost(self, costs):
    if not costs:
        return 0
    dp = costs[0]
    for i in xrange(1, len(costs)):
        pre = dp[:] # here should take care
        for j in xrange(len(costs[0])):
            dp[j] = costs[i][j] + min(pre[:j]+pre[j+1:])
    return min(dp)
```



### 265. Paint House II

https://leetcode.com/problems/paint-house-ii/description/

#### Solution-dp

上一个的一般化

#### Solution-dp-厉害

secondmin的使用, Ref: https://leetcode.com/problems/paint-house-ii/discuss/69495/Fast-DP-Java-solution-Runtime-O(nk)-space-O(1)

> Explanation: dp[i][j] represents the min paint cost from house 0 to house i when house i use color j; The formula will be dp[i][j] = Math.min(any k!= j| dp[i-1][k]) + costs[i][j].
>
> 
>
> Take a closer look at the formula, we don't need an array to represent dp[i][j], we only need to know the min cost to the previous house of any color and if the color j is used on previous house to get prev min cost, use the second min cost that are not using color j on the previous house. So I have three variable to record: prevMin, prevMinColor, prevSecondMin. and the above formula will be translated into: dp\[currentHouse\] = (currentColor == prevMinColor? prevSecondMin: prevMin) + costs\[currentHouse]



### 64. Minimum Path Sum

https://leetcode.com/problems/minimum-path-sum/

跟前面unique path差不多，下次想想就行了@1.15

#### Solution-dp



### 72. Edit Distance

https://leetcode.com/problems/edit-distance/description/

#### Solution-dp

最基本的操作是，对一个字母进行insert/remove/replace的操作，从中间选取最小的

[worth reading](https://leetcode.com/problems/edit-distance/discuss/159295/Python-solutions-and-intuition)，understand how to use recursive method to solve problem(memorize subresult to reduce stack), and how to convert recursive method to dp to prevent overflows of stack 

https://leetcode.wang/leetCode-72-Edit-Distance.html





### 97. Interleaving String-important

https://leetcode.com/problems/interleaving-string/description/

#### Solution-dp+memorization(backtrack)

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1)+len(s2)!=len(s3):
            return False
        notwork = set()
        return self.helper(s1,s2,s3, notwork)
        
    def helper(self, s1, s2, s3, notwork):
        if (s1,s2,s3) in notwork:
            return False
        if len(s3)==0:
            return True
        if s1 and s1[0]==s3[0] and self.helper(s1[1:], s2, s3[1:], notwork):
            return True
        elif s2 and s2[0]==s3[0] and self.helper(s1, s2[1:], s3[1:], notwork):
            return True
        else:
            notwork.add((s1, s2,s3))
            return False
```

#### Solution-dp-bottom up

Ref: https://leetcode.wang/leetCode-97-Interleaving-String.html

>  我们定义一个 boolean 二维数组 dp [ i ] [ j ] 来表示 s1[ 0, i ) 和 s2 [ 0, j ） 组合后能否构成 s3 [ 0, i + j )，注意不包括右边界，主要是为了考虑开始的时候如果只取 s1，那么 s2 就是空串，这样的话 dp [ i ] [ 0 ] 就能表示 s2 取空串。

#### Solution-dp-bfs-worth thinking

Ref: https://leetcode.com/problems/interleaving-string/discuss/31948/8ms-C%2B%2B-solution-using-BFS-with-explanation

> s1 = 'a' s2 = 'b' s3 = 'ab'
>
> 0 --- a --- 0
>
> |              |
>
> b              b
>
> |              |
>
> 0 --- a --- 0



1.16

### 174. Dungeon Game

https://leetcode.com/problems/dungeon-game/description/

#### Solution-dp-worth, right bottom to left top

Ref: https://leetcode.com/problems/dungeon-game/discuss/52774/C%2B%2B-DP-solution





### 221. Maximal Square

https://leetcode.com/problems/maximal-square/description/

#### Solution-worth

太厉害了

Ref：https://leetcode.com/articles/maximal-square/



### 84. Largest Rectangle in Histogram-!

https://leetcode.com/problems/largest-rectangle-in-histogram/

#### Solution1-stack

Ref: https://leetcode.com/problems/largest-rectangle-in-histogram/discuss/28917/AC-Python-clean-solution-using-stack-76ms

This can ensure every bar(in other words, at different heights) would be calculated given the two boundaries that are just smaller than it.



### 85. Maximal Rectangle!!

https://leetcode.com/problems/maximal-rectangle/description/

#### Solution1-the solution1 in 84.

#### Solution2-dp

https://leetcode.com/problems/maximal-rectangle/discuss/29054/Share-my-DP-solution









### Knowledge

#### any()&all()

`any()` function returns True if any item in an iterable are true, otherwise it returns False.

`any(*iterable*)`

`all()` returns true if all of the items are True (or if the iterable is empty). All can be thought of as a sequence of AND operations on the provided iterables. It also short circuit the execution i.e. stop the execution as soon as the result is known.