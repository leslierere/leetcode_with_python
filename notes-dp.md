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

update row by row, no need to update the first col separately

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

这个写得也太烦了吧，因为其实不用单独update first row or first col

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





### 279. Perfect Squares-$$

https://leetcode.com/problems/perfect-squares/description/

#### Solution-dp-worth-slow

Ref: https://leetcode.com/articles/perfect-squares/

```python
class Solution:
    def numSquares(self, n: int) -> int:
        num_squares = [i**2 for i in range(1, int(n**0.5)+1)]
        
        dp = [0] + [float('inf')]*(n)
        
        for i in range(1, n+1):
            for square in num_squares:
                if square > i:
                    break
                dp[i] = min(dp[i], dp[i-square]+1)
                
        return dp[n]
```



#### Solution-Greedy Enumeration-worth-$

Ref: https://leetcode.com/articles/perfect-squares/

#### Solution- Greedy + BFS-worth

Ref: https://leetcode.com/articles/perfect-squares/

```python
from collections import deque
class Solution:
    def numSquares(self, n: int) -> int:
        if n**0.5==int(n**0.5):
            return 1
        
        num_squares = [i**2 for i in range(1, int(n**0.5)+1)]
        queue = deque()
        queue.append(n)
        layer = 1
        
        while queue:
            length = len(queue)
            for _ in range(length):
                cur = queue.popleft()
                for i in num_squares:
                    if cur<i:
                        break
                    remain = cur-i
                    
                    if remain**0.5==int(remain**0.5):
                        return layer+1
                    else:
                        queue.append(remain)
            layer+=1
```





### 139. Word Break-$dp

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



### 375. Guess Number Higher or Lower II-$$

https://leetcode.com/problems/guess-number-higher-or-lower-ii/description/

#### Solution-dp-worth

[https://leetcode.com/problems/guess-number-higher-or-lower-ii/discuss/84766/Clarification-on-the-problem-description.-Problem-description-need-to-be-updated-!!!](https://leetcode.com/problems/guess-number-higher-or-lower-ii/discuss/84766/Clarification-on-the-problem-description.-Problem-description-need-to-be-updated-!!!)

worst case is that you will take as many steps as it can to know the correct answer. But you should wisely(which means to minimize) pick a number thus make your spending as less as possible.

Bottom-up

```python
def getMoneyAmount(self, n):
    need = [[0] * (n+1) for _ in range(n+1)]
    for lo in range(n, 0, -1):#从后往前，这很关键！想想！
        for hi in range(lo+1, n+1):
            need[lo][hi] = min(x + max(need[lo][x-1], need[x+1][hi])
                               for x in range(lo, hi))
    return need[1][n]
```







### 312. Burst Balloons-$$

https://leetcode.com/problems/burst-balloons/

#### Solution-recursive-worth

https://leetcode.com/articles/burst-balloons/

https://www.youtube.com/watch?v=z3hu2Be92UA

关键没有想通边界的处理，亦即code这里`nums[left] * nums[i] * nums[right]`中`nums[left]`和`nums[left]`的部分，因为我一直想的是先爆哪个，其实应该考虑的是先保留哪个

 要用这个functools才会变快

```python
from functools import lru_cache
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1] + nums + [1]
        @lru_cache(None)
        def dp(left, right):
            if left + 1 == right: return 0
            return max(nums[left]*nums[i]*nums[right] + dp(left,i) + dp(i,right) for i in range(left+1, right))
        
        return dp(0, len(nums)-1)
```



#### Solution-dp-fast





### 322. Coin Change-$

https://leetcode.com/problems/coin-change/

#### Solution-recursive

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort(reverse = True)
        lenc, self.res = len(coins), 2**31-1

        def dfs(pt, rem, count):
            if not rem:
                self.res = min(self.res, count)
            for i in range(pt, lenc):
                if coins[i] <= rem < coins[i] * (self.res-count): # if hope still exists
                    dfs(i, rem-coins[i], count+1)

        for i in range(lenc):
            dfs(i, amount, 0)
        return self.res if self.res < 2**31-1 else -1
```



#### Solution-dp-$

既然都已经想到了递推关系，就从小到大一直算下去就好了呀，也就是bottom up！也就是link里面的approach3。 或者使用递归(link里面的approach2)，top down,但要使用记忆。

@2.8，我开始陷入了一个误区，那就是先尽量用大的数填进去，但实际上这可能会造成后面的余数要用更多的coin来相加。

Ref: https://leetcode.com/articles/coin-change/

> First, let's define:
>
> > F(S) - minimum number of coins needed to make change for amount S using coin denominations \[c0…cn−1\]
>
> We note that this problem has an optimal substructure property, which is the key piece in solving any Dynamic Programming problems. In other words, the optimal solution can be constructed from optimal solutions of its subproblems. 

https://leetcode.com/articles/coin-change/

#### Approach #3 (Dynamic programming - Bottom up) [Accepted]

还没想太明白由coin循环的

![image-20200509202750965](https://tva1.sinaimg.cn/large/007S8ZIlgy1gen3jnynxnj31gg08smys.jpg)

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1 
```



1.15

### 256. Paint House-$$

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

> Explanation: dp\[i][j] represents the min paint cost from house 0 to house i when house i use color j; The formula will be dp\[i][j] = Math.min(any k!= j| dp\[i-1][k]) + costs\[i][j].
>
> 
>
> Take a closer look at the formula, we don't need an array to represent dp\[i][j], we only need to know the min cost to the previous house of any color and if the color j is used on previous house to get prev min cost, use the second min cost that are not using color j on the previous house. So I have three variable to record: prevMin, prevMinColor, prevSecondMin. and the above formula will be translated into: dp\[currentHouse\] = (currentColor == prevMinColor? prevSecondMin: prevMin) + costs\[currentHouse]



### 64. Minimum Path Sum

https://leetcode.com/problems/minimum-path-sum/

跟前面unique path差不多，下次想想就行了@1.15

#### Solution-dp



### 72. Edit Distance-$

https://leetcode.com/problems/edit-distance/description/

#### Solution-dp

@5.10思路：首先假设已经有一个recursion的function可以求出edit(i, j), 写出recursion的做法，然后再bottom up写出来, 参照dp的章节

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        if not word1:
            return len(word2)
        elif not word2:
            return len(word1)
        
        
        dp = [i for i in range(len(word1)+1)]
        
            
        for i in range(len(word2)):
            newRow = [i+1]+[0]*len(word1)
            for j in range(1, len(word1)+1):
                if word1[j-1]==word2[i]:
                    newRow[j] = dp[j-1]
                else:
                    newRow[j] = min(dp[j-1], dp[j], newRow[j-1])+1
            dp = newRow
        
        return dp[-1]
```



最基本的操作是，对一个字母进行insert/remove/replace的操作，从中间选取最小的

[worth reading](https://leetcode.com/problems/edit-distance/discuss/159295/Python-solutions-and-intuition)，understand how to use recursive method to solve problem(memorize subresult to reduce stack), and how to convert recursive method to dp to prevent overflows of stack 

https://leetcode.wang/leetCode-72-Edit-Distance.html





### 97. Interleaving String-$

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

did@5.10

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1)+len(s2)!=len(s3):
            return False
        
        dp = [[False]*(len(s1)+1) for _ in range(len(s2)+1)]
        # dp[i][j] represents isInterleave(s1[:i], s[:j], s3[:i+j])
        dp[0][0] = True
        
        for j in range(1, len(dp[0])):
            dp[0][j] = s1[j-1]== s3[j-1]
            if not dp[0][j]:
                break
            
        for i in range(1, len(dp)):
            dp[i][0] = s2[:i]==s3[:i]
            for j in range(1, len(dp[0])):
                
                dp[i][j] = (dp[i-1][j] and s2[i-1]==s3[i+j-1]) or (dp[i][j-1] and s1[j-1]==s3[i+j-1])
                
        return dp[-1][-1]
```



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

### 174. Dungeon Game-$$

https://leetcode.com/problems/dungeon-game/description/

#### Solution-dp-worth, right bottom to left top

Ref: https://leetcode.com/problems/dungeon-game/discuss/52774/C%2B%2B-DP-solution

A natural way to understand why we could only solve the problem from bottom right to top left is just remember dp is a bottom-up solution, and in this case, top left is where our final state(the maximum subproblem) is located because our result should be the minimum hp at top left. Rather than other dp problems our final state would always end at the bottom right.

```python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        
        rows = len(dungeon)
        cols = len(dungeon[0])
        
        dp = [[float('inf') for i in range(cols+1)] for j in range(rows+1)] # set maximum float here
        # if we don't set these 2, we would need deal with the bottom right seperately
        dp[rows][cols-1]=1
        dp[rows-1][cols]=1
        
        for i in range(rows-1, -1, -1):
            for j in range(cols-1, -1, -1):
                need = min(dp[i+1][j], dp[i][j+1]) -dungeon[i][j]
                if need>0:
                    dp[i][j] = need
                else:
                    dp[i][j] = 1
                
        return dp[0][0]
```





### 221. Maximal Square-下次想一想就好了

https://leetcode.com/problems/maximal-square/description/

#### Solution-worth

Ref：https://leetcode.com/articles/maximal-square/

注意one-dimension array就够，所以可以优化

Optimal solution是某个位置作为右下角的最大square area， 因为每一个正方形一定会被它的右下角唯一确定且不会遗漏。







### 85. Maximal Rectangle-$$

https://leetcode.com/problems/maximal-rectangle/description/

#### Solution1-the solution1 in 84.

#### Solution2-dp

https://leetcode.com/problems/maximal-rectangle/discuss/29054/Share-my-DP-solution

It would be easier to understand how to calculate the right and left if you rotate the matrix 90 degrees, clockwise and counter clockwise.

Why this is an exhausted solution is that it actually calculates the maximum area with a given bar, i.e. the given height at each row.

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        res = 0
        height = [0]*len(matrix[0])
        right = [len(matrix[0])-1]*len(matrix[0])
        left = [-1]*len(matrix[0])
        
        
        for i in range(len(matrix)):
            curLeft = -1
            curRight = len(matrix[0])-1
            
            for j in range(len(matrix[0])-1, -1, -1):
                if matrix[i][j] =='1':
                    right[j] = min(curRight, right[j])
                else:
                    curRight = j-1
                    right[j] = len(matrix[0])-1 # This should be changed to this value, rather than (j-1), think of why
                    
            for j in range(len(matrix[0])):
                if matrix[i][j] =='1':
                    height[j] = height[j]+1
                    left[j] = max(curLeft, left[j])
                else:
                    height[j] = 0
                    curLeft = j
                    left[j] = -1
            
                res = max((right[j]-left[j])*height[j], res)
            
        return res
```





### 363. Max Sum of Rectangle No Larger Than K-$$

https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/description/

@5.12

*Max Sum of subarray* which can be solved using [Kadane’s Algorithm](https://hackernoon.com/kadanes-algorithm-explained-50316f4fd8a6)

recursion mindset :

For an array, for each index at i, we need to find out the maximum subarray that ends at i, so the formula is:

maxSumSubarray(i) = max(maxSumSubarray(i-1), 0) + array[i]

and the base case is maxSumSubmatrix(0)=array[0]

So for the 2-dimensional one, we should imagine to compress the rows into just one row which are the sum of different subrow, and in this way, we can consider it like a one-dimensional array.

#### Solution-dp-worth

First consider the problem *Max Sum of Rectangle* which is actually the 2D of the problem of *Max Sum of subarray* which can be solved using [Kadane’s Algorithm](https://hackernoon.com/kadanes-algorithm-explained-50316f4fd8a6)

```java
public int maxSumSubmatrix(int[][] matrix, int k) {
    //2D Kadane's algorithm + 1D maxSum problem with sum limit k
    //2D subarray sum solution
    
    //boundary check
    if(matrix.length == 0) return 0;
    
    int m = matrix.length, n = matrix[0].length;
    int result = Integer.MIN_VALUE;
    
    //outer loop should use smaller axis
    //now we assume we have more rows than cols, therefore outer loop will be based on cols 
    for(int left = 0; left < n; left++){
        //array that accumulate sums for each row from left to right 
        int[] sums = new int[m];
        for(int right = left; right < n; right++){
            //update sums[] to include values in curr right col
            for(int i = 0; i < m; i++){
                sums[i] += matrix[i][right];
            }
            
            //we use TreeSet to help us find the rectangle with maxSum <= k with O(logN) time
            TreeSet<Integer> set = new TreeSet<Integer>();
            //add 0 to cover the single row case
            set.add(0);//a set given for every different left and right boundary
            int currSum = 0;
            
            for(int sum : sums){
                currSum += sum; 
                //we use sum subtraction (curSum - sum) to get the subarray with sum <= k
                //therefore we need to look for the smallest sum >= currSum - k
                // we need (curSum - sum) to be as closed as k, but it cannot be larger than k
                Integer num = set.ceiling(currSum - k);//这个想法很赞
              // ceiling()返回在这个集合中大于或者等于给定元素的最小元素，如果不存在这样的元素,返回null.
                if(num != null) result = Math.max( result, currSum - num );
                set.add(currSum);
            }
        }
    }
    
    return result;
}
```

但在python里面没有set.ceiling()这个方法，所以直接把每一列的可能结果搞出来后, 求一个max_sum_no_larger_than_k

```python
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        if not matrix:
            return 0
        
        rows, cols = len(matrix), len(matrix[0])
        
        def max_sum_no_larger_than_k(arr): #[2, 1, 5, 2, 1, 3] k = 4
            sub_s_max = float('-inf')
            s_curr = 0
            prefix_sums = [float('inf')]
            for x in arr:
                bisect.insort(prefix_sums, s_curr)
                s_curr += x
                i = bisect.bisect_left(prefix_sums, s_curr - k)
                sub_s_max = max(sub_s_max, s_curr - prefix_sums[i])
            return sub_s_max
        
        for row in range(rows):
            for col in range(cols):
                matrix[row][col] += matrix[row][col - 1] if col > 0 else 0
                
        max_sum = - float('inf')
        for col in range(cols):
            for col_right in range(col, cols):
                one_dim_list_for_tow_cols = [matrix[row][col_right] - (matrix[row][col - 1] if col > 0 else 0) for row in range(rows)]
                
                guess = max_sum_no_larger_than_k(one_dim_list_for_tow_cols)
                max_sum = max(max_sum, guess)
                
        return max_sum
```



### 198. House Robber

https://leetcode.com/problems/house-robber/

#### Solution-dp-easy

会做tree里的337这个就没问题





### 213. House Robber II-$实现很容易，但要想想

https://leetcode.com/problems/house-robber-ii/

#### Solution-dp

A little change on the solution of 212, **1. not rob the 1st house; 2. not rob the last house**

A natural way is to split into 3 cases: 1. rub the first without robbing the last, 2. rub the last without rubbing the first, 3. neither rub the first nor the last.

In **not rob the 1st house**, two cases are included, 2. rub the last without rubbing the first, 3. neither rub the first nor the last.

In  **not rob the last house**, two cases are included, 1. rub the first without robbing the last, 3. neither rub the first nor the last.

Since we want the maximum value, we can have overlaps



### 276. Paint Fence-$

https://leetcode.com/problems/paint-fence/description/

#### Solution-recursive-TLE

```python
class Solution:
    def numWays(self, n: int, k: int) -> int:
        if n==0:
            return 0
        if n==1:
            return k
        if n==2:
            return k**2
        
        return self.helper(3, n, k, k) + self.helper(2, n, k, k)
            
    def helper(self, begin, n, k, curSum): 
        # every time we starts from the one with 2 choices
        # which means last time we choose a color that's different from its previous one
        if begin == n:
            return curSum*(k-1)
        elif begin>n:
            return curSum
        
        elif begin+1<=n: # normal case
            return self.helper(begin+2, n, k, curSum*(k-1)) + self.helper(begin+1, n, k, curSum*(k-1))
```





#### Solution-dp

did@5.13-need improvement

```python
class Solution:
    def numWays(self, n: int, k: int) -> int:
        if n==0 or k==0:
            return 0
        
        dp = [k, k*k]
        if n<3:
            return dp[n-1]
        
        for i in range(2, n):
            dp.append(dp[i-2]*(k-1)+dp[i-1]*(k-1))
            # different from the last one
            
        return dp[-1]
```



Ref: https://leetcode.com/problems/paint-fence/discuss/178010/The-only-solution-you-need-to-read

其实还是可以根据recursive的推导出来，只是确实没想到

要想我们只有两个case，一个是和前面画一样的颜色，另一个是不画

```java
class Solution {
    public int numWays(int n, int k) {
        // if there are no posts, there are no ways to paint them
        if (n == 0) return 0;
        
        // if there is only one post, there are k ways to paint it
        if (n == 1) return k;
        
        // if there are only two posts, you can't make a triplet, so you 
        // are free to paint however you want.
        // first post, k options. second post, k options
        if (n == 2) return k*k;
        
        int table[] = new int[n+1];
        table[0] = 0;
        table[1] = k;
        table[2] = k*k;
        for (int i = 3; i <= n; i++) {
            // the recursive formula that we derived
            table[i] = (table[i-1] + table[i-2]) * (k-1);
        }
        return table[n];
    }
}
```





### 91. Decode Ways

https://leetcode.com/problems/decode-ways/description/

@by others, 感觉还是初始化为0的比较能一般化

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        # return reduce(lambda t,d : (t[1], (d>'0')*t[1]+(9<int(t[2]+d)<27)*t[0], d), s, (0, s>'', ''))[1]*1
    
        if not s:
            return 0
        dp = [0 for _ in range(len(s) + 1)]
        dp[0] = 1
        for i in range(1, len(s)+1):
            if s[i-1:i] != '0': # s[i-1]
                dp[i] += dp[i-1]
            if i != 1 and s[i-2] != '0' and s[i-2:i] < '27':
                dp[i] += dp[i-2]
        return dp[len(s)]
```



#### Solution-dp-by myself

Think of it in the way as the one in 276

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        s1 = s.lstrip("0")
        if s1!=s: # if there is 0 on the left side, unable to decode
            return 0
        
        if len(s)<=1:
            return len(s)
        
        dp = [1]
        if int(s[0]+s[1])<=26:
            if s[1] == '0':
                dp.append(1)
            else:
                dp.append(2)
        else:
            if s[1]== '0':
                return 0
            else:
                dp.append(1)
            
        for i in range(2, len(s)):
            digits = int(s[i-1]+s[i])
            if digits == 0:
                return 0
            elif digits<10: # 1-9
                dp.append(dp[i-1])
            elif digits==10 or digits==20:
                dp.append(dp[i-2])
            elif digits<=26:
                dp.append(dp[i-2] + dp[i-1])
                # dp[i-2] is the one the current will combine with the previous one
            else:
                if s[i]=='0':
                    return 0
                dp.append(dp[i-1])
                
        return dp[-1]
```



1.18

### 10. Regular Expression Matching-$$

https://leetcode.com/problems/regular-expression-matching/description/

Here we don't consider the circumstances where * would be the first char in p.

#### Solution-dp-worth!!!

Ref: https://leetcode.com/problems/regular-expression-matching/discuss/5684/C%2B%2B-O(n)-space-DP



@2021.5.19

上面当然也是对的思路，但是我觉得不够自然

First let's forget about the fact that we will use dp for the solution.
Naturally, we would think try to match chars in pattern and string one by one, for string, there is not a lot to think about, as they are all letters, but for the pattern, there are 3 different situations when checking one specific char in pattern and string.

* Char in pattern is plain letter

  The only matching situations lies in, the previous chars in string and the ones in pattern is the same, and the current letter we are checking in pattern should be the same as the one in string.

* Char in pattern is "."

  As long as previous chars in string and pattern match, then the string and pattern so far is matching

* Char in pattern is "*"

  * If the char in pattern, before "\*", say "a",  with "\*" doesn't match the current letter, say "s" in string, the only matching situation is the "\*" will only repeat zero times of the char before it which is "a", and the substring before "a" in pattern should match the substring till "s"(included) in the string
  * Otherwise, the char in pattern, before "\*", say "a" can match the current letter "a" in string, then for 2 substrings to match, we just need satisfy that the pattern till the "a"(included), will match the substring to "a"(excluded) in string.

Condsider the above 3 situations, we can easily find out every time we rely on previous substring matching situation to duduct the current one, and that's where dp comes to be natural.

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if len(p)==0 and len(s)==0: 
            return True
        if len(p)==0:
            return False
        
        dp = [[False for _ in range(len(s)+1)] for _ in range(len(p)+1)]
        dp[0][0] = True
            
        for i in range(1, len(dp)):
            for j in range(0, len(dp[0])):
                char_s = s[j-1] if j-1>=0 else ""
                char_p = p[i-1]
                
                if char_p == "*": 
                    dp[i][j] = dp[i-2][j] or (p[i-2] in [".", char_s] and dp[i][j-1])
                elif char_p == ".":
                    if j-1>=0:
                        dp[i][j] = dp[i-1][j-1]
                else:
                    if j-1>=0:
                        dp[i][j] = dp[i-1][j-1] and char_s==char_p
                    
        return dp[-1][-1]
```



#### Solution-recursive-worth

Ref: https://leetcode.com/problems/regular-expression-matching/discuss/5665/My-concise-recursive-and-DP-solutions-with-full-explanation-in-C%2B%2B

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        return self.helper(s, p, 0, 0, {})
        
    def helper(self, s, p, i, j, dp):
        if (i, j) in dp:
            return dp[(i, j)]
        if i>=len(s) and j>=len(p):
            return True
        if j>=len(p):
            return False
        
        
        if j<len(p)-1 and p[j+1]=="*":
            dp[(i, j)] = self.helper(s, p, i, j+2, dp) or (i!=len(s) and p[j] in [".", s[i]] and self.helper(s, p, i+1, j, dp))
            return dp[(i, j)]
        else:
            dp[(i, j)] = i!=len(s) and (s[i]==p[j] or p[j]==".") and self.helper(s, p, i+1, j+1, dp)
            return dp[(i, j)]
```





### 44. Wildcard Matching-$$

https://leetcode.com/problems/wildcard-matching/description/

#### Solution-dp

did@21.5.26, 可以对比https://leetcode.com/problems/regular-expression-matching/

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        dp = [[False for _ in range(len(s)+1)] for _ in range(len(p)+1)]
        dp[0][0] = True
        
        for i in range(1, len(dp)):
            for j in range(0, len(dp[0])):
                if p[i-1]=="*":
                    dp[i][j] = dp[i-1][j] or (j>0 and (dp[i][j-1] or dp[i-1][j-1]))
                elif p[i-1]=="?":
                    dp[i][j] = j>0 and dp[i-1][j-1]
                else:
                    dp[i][j] = j>0 and p[i-1]==s[j-1] and dp[i-1][j-1]
                    
        return dp[-1][-1]
```



Ref:https://leetcode.com/problems/wildcard-matching/discuss/17812/My-java-DP-solution-using-2D-table

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        dp = [[False for i in range(len(s)+1)] for j in range(len(p)+1)]
        
        dp[0][0] = True
        
        for i in range(1, len(dp)):
            for j in range(len(dp[0])):
                if j==0:
                    dp[i][j] = p[i-1]=='*' and dp[i-1][j]
                    continue
                if p[i-1]=='*':
                    dp[i][j]=dp[i-1][j] or dp[i][j-1]
                    # 我总是想不清要匹配的情况
                    # dp[i-1][j] suggests that * represents ""
                    # dp[i][j-1], the difference between dp[i][j-1] and dp[i][j] is that we consider one more, that is s[j-1], so here we suggests that we use * to match s[j-1]
                elif p[i-1]==s[j-1] or p[i-1]=='?':
                    dp[i][j] = dp[i-1][j-1]
                    
        return dp[len(p)][len(s)]
```



#### Solution-two pointers, backtrack



### 1048. Longest String Chain-$

https://leetcode.com/problems/longest-string-chain/

像这道题我就还是用数组做的，但其实用hash就会快很多



### 956. Tallest Billboard

https://leetcode.com/problems/tallest-billboard/

how to think of this problem: 

> I think dp[d] mean the maximum pair of sum we can get with pair difference = **d**

其实下面两个思路差不多

#### Solution-dp

[https://leetcode.com/problems/tallest-billboard/discuss/203181/JavaC%2B%2BPython-DP-min(O(SN2)-O(3N2-*-N)](https://leetcode.com/problems/tallest-billboard/discuss/203181/JavaC%2B%2BPython-DP-min(O(SN2)-O(3N2-*-N))

#### Solution-permutation /w memorization

https://leetcode.com/problems/tallest-billboard/discuss/219700/Python-DP-clean-solution(1D)



### 1320. Minimum Distance to Type a Word Using Two Fingers

https://leetcode.com/problems/minimum-distance-to-type-a-word-using-two-fingers/

#### Solution-dp

https://leetcode.com/problems/minimum-distance-to-type-a-word-using-two-fingers/discuss/477652/JavaC%2B%2BPython-1D-DP-O(1)-Space

2d的思路跟956很像，1d还没看 