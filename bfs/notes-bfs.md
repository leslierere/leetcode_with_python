## Foundation

### 7.23

#### 27. Remove Element-two pointers

https://leetcode.com/problems/remove-element/



#### 26. Remove Duplicates from Sorted Array-two pointers

https://leetcode.com/problems/remove-duplicates-from-sorted-array/



#### 80. Remove Duplicates from Sorted Array II

https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/



#### 277. Find the Celebrity

https://leetcode.com/problems/find-the-celebrity/



### 7.24

#### 189. Rotate Array

https://leetcode.com/problems/rotate-array/

* solution

  Approach #4 Using Reverse 



#### 41. First Missing Positive

https://leetcode.com/problems/first-missing-positive/

* solution

  https://www.cnblogs.com/grandyang/p/4395963.html

   



#### 299. Bulls and Cows

https://leetcode.com/problems/bulls-and-cows/

* solution-我做的稍微复杂了点，思想类似，学会这种写法

  ```python
  def getHint(self, secret, guess):
          A = sum(a==b for a,b in zip(secret, guess))
          B = collections.Counter(secret) & collections.Counter(guess)
          return "%dA%dB" % (A, sum(B.values()) - A)
  ```

  

#### 274. H-Index

https://leetcode.com/problems/h-index/

* Solution-besides mine

  用extra space来做，faster

  [https://leetcode.com/problems/h-index/discuss/70818/Java-O(n)-time-with-easy-explanation.](https://leetcode.com/problems/h-index/discuss/70818/Java-O(n)-time-with-easy-explanation.)



### 7.25

#### 275. H-Index II

https://leetcode.com/problems/h-index-ii/



#### 134. Gas Station

https://leetcode.com/problems/gas-station/

* solution

  https://leetcode.com/problems/gas-station/discuss/42568/Share-some-of-my-ideas.

  > - If car starts at A and can not reach B. Any station between A and B
  >   can not reach B.(B is the first station that A can not reach.)
  > - If the total number of gas is bigger than the total number of cost. There must be a solution.



#### 243. Shortest Word Distance

https://leetcode.com/problems/shortest-word-distance/



#### 244. Shortest Word Distance II

https://leetcode.com/problems/shortest-word-distance-ii/





### 7.26

#### 245. Shortest Word Distance III

https://leetcode.com/problems/shortest-word-distance-iii/



#### 55. Jump Game

https://leetcode.com/problems/jump-game/

* Solution-greedy method



#### 45. Jump Game II

https://leetcode.com/problems/jump-game-ii/

* Solution-greedy method, didn't submit



#### 121. Best Time to Buy and Sell Stock

https://leetcode.com/problems/best-time-to-buy-and-sell-stock/



#### 122. Best Time to Buy and Sell Stock II

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/





### 7.27

#### 123. Best Time to Buy and Sell Stock III

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/

* Solution-dynamic programming

  worth thinking

  主要是一个累计的思想，dp到底是啥？



#### 188. Best Time to Buy and Sell Stock IV

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/

* solution

  ***worth thinking and doing***



#### 11. Container With Most Water

https://leetcode.com/problems/container-with-most-water/

* solution

  Worth thinking when two columns are equal

  https://leetcode.wang/leetCode-11-Container-With-Most-Water.html



#### 334. Increasing Triplet Subsequence

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





### 7.29

#### 42. Trapping Rain Water-not submitted

* Solution-stack
* Solution-two pointers

***worth thinking and doing***



#### 128. Longest Consecutive Sequence

https://leetcode.com/problems/longest-consecutive-sequence/

* Solution-hashset



#### 164. Maximum Gap

https://leetcode.com/problems/maximum-gap/submissions/

* Solution-bucket sort

  

### 7.30

#### 287. Find the Duplicate Number-similar to 136 Single number

https://leetcode.com/problems/find-the-duplicate-number/

***worth thinking and doing***

* solution-Floyd's Tortoise and Hare (Cycle Detection)

  > First assume when **fast** and **slow** meet, slow has moved **a** steps, and fast has moved **2a** steps. They meet in the circle, so the difference **a** must be a multiple of the length of the circle.
  > Next assume the distance between beginning to the entry point is **x**, then we know that the **slow** has traveled in the circle for **a-x** steps.
  > How do we find the entry point? Just let **slow** move for another **x** steps, then **slow** will have moved **a** steps in the circle, which is a multiple of length of the circle.
  > So we start another pointer at the beginning and let **slow** move with it. Remember **x** is the distance between beginning to the entry point, after **x**steps, both pointer will meet at the entry of circle.
  >
  > https://leetcode.com/problems/find-the-duplicate-number/discuss/72846/My-easy-understood-solution-with-O(n)-time-and-O(1)-space-without-modifying-the-array.-With-clear-explanation./268392



#### 4. Median of Two Sorted Arrays

https://leetcode.com/problems/median-of-two-sorted-arrays/

* Solution-divide&conquer

https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2652/Share-one-divide-and-conquer-O(log(m%2Bn))-method-with-clear-description

***worth thinking and doing***

* Solution-binary search

***worth thinking and doing***



#### 289. Game of Life

https://leetcode.com/problems/game-of-life/

***follow up-worth thinking and doing***



#### 56. Merge Intervals

https://leetcode.com/problems/merge-intervals/

***worth thinking the use of sort***





#### 57. Insert Interval

https://leetcode.com/problems/insert-interval/





## 语法

### list

```python
list.reverse()
```



### 正负无穷

```python
正无穷：float("inf"); 负无穷：float("-inf")
```

利用 inf 做简单加、乘算术运算仍会得到 inf





### Dynamic Programming

> Dynamic Programming is mainly an optimization over plain [recursion](https://www.geeksforgeeks.org/recursion/). Wherever we see a recursive solution that has repeated calls for same inputs, we can optimize it using Dynamic Programming. The idea is to simply store the results of subproblems, so that we do not have to re-compute them when needed later. This simple optimization reduces time complexities from exponential to polynomial. For example, if we write simple recursive solution for [Fibonacci Numbers](https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/), we get exponential time complexity and if we optimize it by storing solutions of subproblems, time complexity reduces to linear.

![](https://www.geeksforgeeks.org/wp-content/uploads/Dynamic-Programming-1.png)







### Hashset

对于 HashSet 而言，它是基于 HashMap 实现的，底层采用 HashMap 来保存元素

如果此 set 中尚未包含指定元素，则添加指定元素。更确切地讲，如果此 set 没有包含满足(e==null ? e2==null : e.equals(e2)) 的元素 e2，则向此 set 添加指定的元素 e。如果此 set 已包含该元素，则该调用不更改 set 并返回 false。但底层实际将将该元素作为 key 放入 HashMap。思考一下为什么？

由于 HashMap 的 put() 方法添加 key-value 对时，当新放入 HashMap 的 Entry 中 key 与集合中原有 Entry 的 key 相同（hashCode()返回值相等，通过 equals 比较也返回 true），新添加的 Entry 的 value 会将覆盖原来 Entry 的 value（HashSet 中的 value 都是`PRESENT`），但 key 不会有任何改变，因此如果向 HashSet 中添加一个已经存在的元素时，新添加的集合元素将不会被放入 HashMap中，原来的元素也不会有任何改变，这也就满足了 Set 中元素不重复的特性。





### 防止0的出现

```python
size = (b-a)//(len(num)-1) or 1
```

