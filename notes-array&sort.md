7.23

### 27. Remove Element-two pointers

https://leetcode.com/problems/remove-element/



### 26. Remove Duplicates from Sorted Array-two pointers

https://leetcode.com/problems/remove-duplicates-from-sorted-array/

Ref: https://leetcode.com/problems/remove-duplicates-from-sorted-array/discuss/11782/Share-my-clean-C%2B%2B-code

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0 # last index till which no duplicates exist
        
        for j in range(1, len(nums)):
            if nums[i]!=nums[j]:
                nums[i+1]=nums[j]
                i+=1
                
        return i+1
```



### 80. Remove Duplicates from Sorted Array II

https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/



### 277. Find the Celebrity

https://leetcode.com/problems/find-the-celebrity/



7.24

### 189. Rotate Array

https://leetcode.com/problems/rotate-array/

#### solution

Approach #4 in Solution Using Reverse 



### 41. First Missing Positive

https://leetcode.com/problems/first-missing-positive/

* solution

  https://www.cnblogs.com/grandyang/p/4395963.html

   



### 299. Bulls and Cows

https://leetcode.com/problems/bulls-and-cows/

* solution-我做的稍微复杂了点，思想类似，学会这种写法

  ```python
  def getHint(self, secret, guess):
          A = sum(a==b for a,b in zip(secret, guess))
          B = collections.Counter(secret) & collections.Counter(guess)
          return "%dA%dB" % (A, sum(B.values()) - A)
  ```

  

### 274. H-Index

https://leetcode.com/problems/h-index/

* Solution-besides mine

  用extra space来做，faster

  [https://leetcode.com/problems/h-index/discuss/70818/Java-O(n)-time-with-easy-explanation.](https://leetcode.com/problems/h-index/discuss/70818/Java-O(n)-time-with-easy-explanation.)



7.25

### 275. H-Index II

https://leetcode.com/problems/h-index-ii/



### 134. Gas Station

https://leetcode.com/problems/gas-station/

#### Solution



We can absolutely know that if total cost is larger than total cost, we got no solution. And for total cost==total gas, how to prove we must have a solution?

Suppose there is a new array which will records the increment amount at a gas station, increments[i] = gas[i]-cost[i].

Suppose when total cost==total gas, there is no solution, in this case, *starting from any gas station, at least one of the cumulative amount of increments will be less than zero*, 

```
gas = [1,2,3,4,5], cost = [3,4,5,1,2], increments = [-2, -2, -2, 3, 3]
```

Use this example, starting from i = 2, the cumulative value is -2, 1, 4, 2, 0 consecutively. So starting from i=2 is impossible to finish the route. See the cumulative value on the other side, we can know it is actually the sum of the sub array. To iterate *starting from any gas station, at least one of the cumulative amount of increments will be less than zero*, it is actually for any starting index, at least one sum of subarray is less than zero. Since we know the sum of the max length array is 0, if we find one sum of subarray<0, name it **array1**, a sum equals **sum1**, then the sum of subarray composed of the remaining numbers>0, name it **array2**, a sum equals **sum2**, we should finish going through **array2** before **array1**, in this case, we can make sure in **array1** we won't starve as sum1+sum2 = 0. However, you may ask, in the **array2**, there can be subarray whose sum<0, then for the remaining part, the sum also>0, and we should go through that first, and in the worst case, we always find one subarray whose sum<0, and we do this recursively, we will finnally land at an array whose sum>0 and there is just one element there. So we can definitly has one solution.

Ref: https://leetcode.com/problems/gas-station/discuss/42572/Proof-of-%22if-total-gas-is-greater-than-total-cost-there-is-a-solution%22.-C%2B%2B

好好体会。。。



### 243. Shortest Word Distance

https://leetcode.com/problems/shortest-word-distance/



### 244. Shortest Word Distance II

https://leetcode.com/problems/shortest-word-distance-ii/





7.26

### 245. Shortest Word Distance III

https://leetcode.com/problems/shortest-word-distance-iii/



### 55. Jump Game

https://leetcode.com/problems/jump-game/

* Solution-greedy method



### 45. Jump Game II

https://leetcode.com/problems/jump-game-ii/

* Solution-greedy method, didn't submit

* Solution-dp

Did@21.5.26, https://leetcode.com/problems/jump-game-ii/discuss/1233094/Easy-to-understand-python-solution

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        steps = [1001 for _ in range(len(nums))]
        steps[0] = 0
        
        for i in range(len(nums)):
            for number in range(1, nums[i]+1):
                next_pos = i+number
                if next_pos<len(nums):
                    steps[next_pos] = min(steps[i]+1, steps[next_pos])
                else:
                    break
                
        return steps[-1]
```





### 121. Best Time to Buy and Sell Stock

https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

#### Solution

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        
        min_price = prices[0]
        
        for price in prices:
            profit = max(price-min_price, profit)
            min_price = min(min_price, price)
            
        return profit
```

#### Solution-maximum subarray

wow

Ref: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/discuss/39038/Kadane's-Algorithm-Since-no-one-has-mentioned-about-this-so-far-%3A)-(In-case-if-interviewer-twists-the-input)



### 122. Best Time to Buy and Sell Stock II

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

#### Solution-stack

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        stack = [prices[0]]
        profit = 0
        
        for i in range(1, len(prices)):
            if prices[i]<=stack[-1]:
                if len(stack)>1:
                    profit += stack[-1]-stack[0]
                stack = [prices[i]]
            else:
                stack.append(prices[i])
        
        if stack:
            profit+=stack[-1]-stack[0]
        return profit
```



#### Solution

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res = 0
        
        for i in range(1, len(prices)):
            diff = prices[i]-prices[i-1]
            if diff>0:
                res+=diff
            
        return res
```





7.27

### 123. Best Time to Buy and Sell Stock III

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/

#### Solution-dynamic programming-$$$

Ref: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/135704/Detail-explanation-of-DP-solution

TLE

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0 for i in range(len(prices))] for j in range(3)]
        
        for k in range(1,3): # number of transactions
            for i in range(1, len(dp[0])): # selling day
                dp[k][i] = max(dp[k][i-1], prices[i]-prices[0]) # today sell or not sell
                for j in range(0,i): # buying day, TODO
                    dp[k][i] = max(dp[k][i], prices[i]-prices[j]+dp[k-1][j])
                
        return dp[2][-1]
```



做一个min_val的替换

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0 for i in range(len(prices))] for j in range(3)]
        
        for k in range(1,3): # number of transactions
            for i in range(1, len(dp[0])): # selling day
                # at first, min_val = prices[j]-dp[k-1][j-1]
                # since we iterate j starting at 1, so when j==0
                # min_val = prices[0] - dp[k-1][0-1], dp[k-1][j-1] has to be 0
                min_val = prices[0]
                for j in range(1,i): # buying day
                    min_val = min(min_val, prices[j]-dp[k-1][j-1]) 
                dp[k][i] = max(dp[k][i-1], prices[i]-min_val)
                    
        return dp[2][-1]
```



```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0 for i in range(len(prices))] for j in range(3)]
        
        for k in range(1,3): # number of transactions
            min_val = prices[0]
            dp[k][1] = max(dp[k][0], prices[1]-min_val)
            for i in range(2, len(dp[0])): # selling day
                # min_val = prices[0]
                #for j in range(1,i): # buying day, everytime, what we actually update is just i-1
                min_val = min(min_val, prices[i-1]-dp[k-1][i-2]) 
                   	# min_val = min(min_val, prices[j]-dp[k-1][j-1]) 
                dp[k][i] = max(dp[k][i-1], prices[i]-min_val)
                    
        return dp[2][-1]
      
```







### 188. Best Time to Buy and Sell Stock IV

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/

* Solution-dp

  ***worth thinking and doing***



Refer to 123

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if len(prices)<2:
            return 0
        dp = [[0 for i in range(len(prices))] for j in range(k+1)]
        
        for k in range(1,k+1): # number of transactions
            min_val = prices[0]
            dp[k][1] = max(dp[k][0], prices[1]-min_val)
            for i in range(2, len(dp[0])): # selling day
                min_val = min(min_val, prices[i-1]-dp[k-1][i-2]) 
                dp[k][i] = max(dp[k][i-1], prices[i]-min_val)
                    
        return dp[k][-1]
```





### 11. Container With Most Water

https://leetcode.com/problems/container-with-most-water/

##### solution- 2 pointer

https://leetcode.wang/leetCode-11-Container-With-Most-Water.html

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height)-1
        min_height = min(height[left], height[right])
        res = (right-left)*min_height
        
        while left<right:
            while height[left]<=min_height and left<right:
                left+=1
            while height[right]<=min_height and left<right:
                right-=1
            min_height = min(height[left], height[right])
            res = max((right-left)*min_height, res)
            
        return res
```



### 334. Increasing Triplet Subsequence

https://leetcode.com/problems/increasing-triplet-subsequence/

* solution

  分类的顺序要么小到大要么大到小，不要一下这这区间一下一下又跳到那个，而且小大，大小也是有讲究的，另外初始值的设置也很关键

  https://leetcode.com/problems/increasing-triplet-subsequence/discuss/78995/Python-Easy-O(n)-Solution

  ```python
  def increasingTriplet(nums):
      first = second = float('inf')
      for n in nums:
          if n <= first:
              first = n
          elif n <= second:
              second = n
          else:
              return True
      return False
  ```





7.29

### 42. Trapping Rain Water-not submitted

* Solution-stack
* Solution-two pointers

***worth thinking and doing***



### 128. Longest Consecutive Sequence

https://leetcode.com/problems/longest-consecutive-sequence/

* Solution-hashset





### 164. Maximum Gap

https://leetcode.com/problems/maximum-gap/submissions/

#### Solution-radix sort-$

@21.6.18

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        if len(nums)<2:
            return 0
        maxNum = max(nums)
        if maxNum==0:
            return 0
        
        radix = 0 # the number of digits we have
        num = maxNum
        while num:
            num=num//10
            radix+=1
         
        for i in range(radix):
            divisor=10**i
            temp = [[] for i in range(10)] 
            for num in nums:
                index = (num//divisor)%10
                temp[index].append(num)
            nums = []
            for subarray in temp:
                for num in subarray:
                    nums.append(num)
        
        maxDiff = 0
        for i in range(1, len(nums)):
            maxDiff = max(maxDiff, nums[i]-nums[i-1])
            
        return maxDiff
```



#### Solution-bucket sort

Ref: https://leetcode.com/problems/maximum-gap/discuss/1240543/Python-Bucket-sort-explained

可以看一下cs61b notes里关于bucket sort的理解

Ref: https://leetcode.com/problems/maximum-gap/discuss/50643/bucket-sort-JAVA-solution-with-explanation-O(N)-time-and-space

> Let gap = ceiling[(***max*** - ***min*** ) / (N - 1)]. We divide all numbers in the array into n-1 buckets, where k-th bucket contains all numbers in [***min*** + (k-1)gap, ***min*** + k*gap). Since there are n-2 numbers that are not equal ***min*** or ***max*** and there are n-1 buckets, **at least one of the buckets are empty**. We only need to store the largest number and the smallest number in each bucket.

The key here is that **at least one of the buckets are empty**. Also, I feel like it is needed that we get the ceiling rather than the floor.



7.30

### 287. Find the Duplicate Number-similar to 136 Single number

https://leetcode.com/problems/find-the-duplicate-number/

***worth thinking and doing***

* solution-Floyd's Tortoise and Hare (Cycle Detection)

  > First assume when **fast** and **slow** meet, slow has moved **a** steps, and fast has moved **2a** steps. They meet in the circle, so the difference **a** must be a multiple of the length of the circle.
  > Next assume the distance between beginning to the entry point is **x**, then we know that the **slow** has traveled in the circle for **a-x** steps.
  > How do we find the entry point? Just let **slow** move for another **x** steps, then **slow** will have moved **a** steps in the circle, which is a multiple of length of the circle.
  > So we start another pointer at the beginning and let **slow** move with it. Remember **x** is the distance between beginning to the entry point, after **x**steps, both pointer will meet at the entry of circle.
  >
  > https://leetcode.com/problems/find-the-duplicate-number/discuss/72846/My-easy-understood-solution-with-O(n)-time-and-O(1)-space-without-modifying-the-array.-With-clear-explanation./268392



### 4. Median of Two Sorted Arrays

https://leetcode.com/problems/median-of-two-sorted-arrays/

* Solution-divide&conquer

https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2652/Share-one-divide-and-conquer-O(log(m%2Bn))-method-with-clear-description

***worth thinking and doing***

* Solution-binary search

***worth thinking and doing***



### 289. Game of Life

https://leetcode.com/problems/game-of-life/

***follow up-worth thinking and doing***



### 16. 3Sum Closest

https://leetcode.com/problems/3sum-closest/

* Solution-two pointers

how to think of 2 pointers: https://leetcode.com/problems/3sum-closest/discuss/7871/Python-O(N2)-solution/210643

> I think the insight is something like this - Given an array and a brute force algorithm that seems waaay too slow (n^3), try to think of ways that we could get it to n^2, nlogn, n. If the given array problem is the type of a problem where order/index doesn't matter, always consider sorting the array. Once you've got it sorted, you have a great heuristic to use to iterate over the array.
>
> 
>
> If you've gotten to that point, and are wondering how to traverse the array, 1, 2, 3+ pointers is always something that should be at the top of your list of things to consider when tackling an unfamiliar problem.

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        result = sum(nums[:3])
        
        for i in range(len(nums)-2):
            k, j = i+1, len(nums)-1
            
            while k<j:
                candidate = nums[i]+nums[j]+nums[k]
                if candidate==target:
                    return target
                elif candidate<target:
                    k+=1
                else:
                    j-=1
                if abs(candidate-target)<abs(result-target):
                    result = candidate
                    
        return result
```



### 56. Merge Intervals

https://leetcode.com/problems/merge-intervals/

***worth thinking the use of sort***





### 57. Insert Interval

https://leetcode.com/problems/insert-interval/

搞清边界很tricky

这个就很简洁：https://leetcode.com/problems/insert-interval/discuss/21602/Short-and-straight-forward-Java-solution



7.31

### 253. Meeting Rooms II

https://leetcode.com/problems/meeting-rooms-ii/



### 352. Data Stream as Disjoint Intervals

https://leetcode.com/problems/data-stream-as-disjoint-intervals/

* Similar questions

  [57.Insert Interval](https://leetcode.com/problems/insert-interval/)

* solution

  Worth try binary search



### 53. Maximum subarray

https://leetcode.com/problems/maximum-subarray/submissions/

* Solution-dynamic programming

  ***worth thinking***

  > Apparently, this is a optimization problem, which can be usually solved by DP. So when it comes to DP, the first thing for us to figure out is the format of the sub problem(or the state of each sub problem). The format of the sub problem can be helpful when we are trying to come up with the recursive relation.
  >
  > https://leetcode.com/problems/maximum-subarray/discuss/20193/DP-solution-and-some-thoughts

  > algorithm that operates on arrays: it starts at the left end (element A[1]) and scans through to the right end (element A[n]), keeping track of the maximum sum subvector seen so far. The maximum is initially A[0]. Suppose we've solved the problem for A[1 .. i - 1]; how can we extend that to A[1 .. i]?
  >
  > https://leetcode.com/problems/maximum-subarray/discuss/20211/Accepted-O(n
  >
  > ***超级像演绎推理，把问题一般化***

* Solution-divide and conquer

  ***worth thinking and doing***
  
  Ref: https://leetcode.com/problems/maximum-subarray/discuss/199163/Python-O(N)-Divide-and-Conquer-solution-with-explanations



### 239. Sliding Window Maximum

https://leetcode.com/problems/sliding-window-maximum/

#### Solution-pq

did@21.6.16

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        pq = []
        for i in range(k):
            pq.append((-nums[i],i))
        heapq.heapify(pq)
        res = [-pq[0][0]]
        
        for i in range(1, len(nums)-k+1):
            num = nums[i+k-1]
            heapq.heappush(pq, (-num, i+k-1))
            value, index = pq[0]
            while index<i:
                heapq.heappop(pq)
                value, index = pq[0]
            res.append(-pq[0][0])
            
        return res
```



#### Solution-deque-not done

***worth thinking and doing***

Ref: https://leetcode.com/problems/sliding-window-maximum/discuss/65884/Java-O(n)-solution-using-deque-with-explanation

Our current window always keeps the things in the range and keeps in a descending order.

#### Solution- dynamic programming-not done

dont understand



### 295. Find Median from Data Stream

https://leetcode.com/problems/find-median-from-data-stream/

* Solution-two priority queue

  ***worth thinking and doing***

  虽然heapq是最小堆，但换为负数就好了。

  善用两个heap



8.2

### 325. Maximum Size Subarray Sum Equals k

https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/

* solution

  > Hint2: Given *S[i]* a partial sum that starts at position *0* and ends at *i*, what can *S[i - k]* tell you ?
  >
  > Hint3: Use HashMap + prefix sum array.

  应该想到既然要使S[i - k]有意义的话，要如何设计dic，这里则是将常规的value用作key



### 209. Minimum Size Subarray Sum

https://leetcode.com/problems/minimum-size-subarray-sum/

* Solution-two pointer

  思考怎么精简我的做法

* solution-binary search

  not done



### 238. Product of Array Except Self

https://leetcode.com/problems/product-of-array-except-self/

* solution

  最开始的想法还是维持两个array，这样就可以从O(n^2)降到O(n)



### 228. Summary Ranges-easy

https://leetcode.com/problems/summary-ranges/





### 152. Maximum Product Subarray

https://leetcode.com/problems/maximum-product-subarray/

* solution

  ***worth thinking and doing***

  Ref: https://leetcode.com/problems/maximum-product-subarray/discuss/48230/Possibly-simplest-solution-with-O(n)-time-complexity

  ```java
  int maxProduct(int A[], int n) {
      // store the result that is the max we have found so far
      int r = A[0];
  
      // imax/imin stores the max/min product of
      // subarray that ends with the current number A[i]
      for (int i = 1, imax = r, imin = r; i < n; i++) {
          // multiplied by a negative makes big number smaller, small number bigger
          // so we redefine the extremums by swapping them
          if (A[i] < 0)
              swap(imax, imin);
  
          // max/min product for the current number is either the current number itself
          // or the max/min by the previous number times the current one
          imax = max(A[i], imax * A[i]);
          imin = min(A[i], imin * A[i]);
  
          // the newly computed max value is a candidate for our global result
          r = max(r, imax);
      }
      return r;
  }
  ```

  Comment:

  > 这道题妙就妙在它不仅仅依赖了一个状态（前一个数所能获得的最大乘积），而是两个状态（最大和最小乘积）。比较简单的dp问题可能就只是会建立一个`dp[]`，然后把最大值放到其中。但是这道题给我们打开了新的思路：我们的dp数组里面可以存更多的信息。而上面的解法之所以没有用dp数组的原因是`dp[i]`只依赖于`dp[i - 1]`因此没有必要把前面所有的信息都存起来，只需要存前一个`dp[i-1]`的最大和最小的乘积就可以了。下面的代码使用了自定义的内部类`Tuple`,从而可以同时存`imax`和`imin`,并将所有的`imax`和imin存到了dp数组中。虽然稍显复杂，但是有助于加深理解。



8.4

### 163. Missing Ranges

https://leetcode.com/problems/missing-ranges/

* solution

  对upper bound一个很常见的处理：

  ```python
  def findMissingRanges(self, A, lower, upper):
          result = []
          A.append(upper+1)
          pre = lower - 1
          for i in A:
             if (i == pre + 2):
                 result.append(str(i-1))
             elif (i > pre + 2):
                 result.append(str(pre + 1) + "->" + str(i -1))
             pre = i
          return result
  ```



### 88. Merge Sorted Array

https://leetcode.com/problems/merge-sorted-array/

#### Solution-2 pointer

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
       
        i = m-1
        j = n-1
        index = m+n-1
        
        while index>=0:
            if j<0 or i>=0 and nums1[i]>=nums2[j]:
                nums1[index] = nums1[i]
                i-=1
            else:
                nums1[index] = nums2[j]
                j-=1
            index-=1
```



但是这个人的判断条件写得更好, 我们最后只需要考虑剩下j的情况，因为nums1中前i个自然就在那里了

Ref: https://leetcode.com/problems/merge-sorted-array/discuss/29522/This-is-my-AC-code-may-help-you

```java
class Solution {
public:
    void merge(int A[], int m, int B[], int n) {
        int i=m-1;
		int j=n-1;
		int k = m+n-1;
		while(i >=0 && j>=0)
		{
			if(A[i] > B[j])
				A[k--] = A[i--];
			else
				A[k--] = B[j--];
		}
		while(j>=0)
			A[k--] = B[j--];
    }
};
```





### 75. Sort Colors

https://leetcode.com/problems/sort-colors/

* solution-two pointers

  \#Array Transformation\#常用思路

  想这个思路时，可以先考虑只有两个数的时候再推导到三个数的

  ```python
  class Solution:
      def sortColors(self, nums: List[int]) -> None:
          """
          Do not return anything, modify nums in-place instead.
          """
          left = p = 0
          right = len(nums)-1
          
          while p<=right: # think of why equal here
              if nums[p]==1:
                  p+=1
              elif nums[p]==0:
                  nums[p], nums[left] = nums[left], nums[p]
                  left+=1
                  p+=1
              else:
                  nums[p],nums[right] = nums[right], nums[p]
                  right-=1
  ```

  

* Solution-worth thinking

  本质是我们常用的递归思想，先假设一个小问题解决了，然后假如再来一个数该怎么操作。

  Ref: https://leetcode.wang/leetCode-75-Sort-Colors.html, https://leetcode.com/problems/sort-colors/discuss/26500/Four-different-solutions

  ```java
  // one pass in place solution
  void sortColors(int A[], int n) {
      int n0 = -1, n1 = -1, n2 = -1;
      for (int i = 0; i < n; ++i) {
          if (A[i] == 0) 
          {
              A[++n2] = 2; A[++n1] = 1; A[++n0] = 0;
          }
          else if (A[i] == 1) 
          {
              A[++n2] = 2; A[++n1] = 1;
          }
          else if (A[i] == 2) 
          {
              A[++n2] = 2;
          }
      }
  }
  ```
  
  

### 283. Move Zeroes

https://leetcode.com/problems/move-zeroes/

* Solution-two pinter

  \#Array Transformation\#

  和75.sort colors一起想



### 376. Wiggle Subsequence

https://leetcode.com/problems/wiggle-subsequence/

* Solution-greedy
* Solution-dynamic programming-not done



8.5

### 280. Wiggle Sort

https://leetcode.com/problems/wiggle-sort/

* Solution-还可以更精简

  思路参照一下，index为偶数和奇数分情况即可，ref: https://leetcode.com/problems/wiggle-sort/discuss/71692/Java-O(N)-solution

  ```java
  public void wiggleSort(int[] nums) {
      for (int i=1; i<nums.length; i++) {
          int a = nums[i-1];
          if ((i%2 == 1) == (a > nums[i])) {
              nums[i-1] = nums[i];
              nums[i] = a;
          }
      }
  }
  ```

  

### 324. Wiggle Sort II

https://leetcode.com/problems/wiggle-sort-ii/

* Solution-not done!!!

