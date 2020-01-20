7.23

#### 27. Remove Element-two pointers

https://leetcode.com/problems/remove-element/



#### 26. Remove Duplicates from Sorted Array-two pointers

https://leetcode.com/problems/remove-duplicates-from-sorted-array/



#### 80. Remove Duplicates from Sorted Array II

https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/



#### 277. Find the Celebrity

https://leetcode.com/problems/find-the-celebrity/



7.24

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



7.25

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





7.26

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





7.27

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





7.29

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

  

7.30

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



7.31

#### 253. Meeting Rooms II

https://leetcode.com/problems/meeting-rooms-ii/



#### 352. Data Stream as Disjoint Intervals

https://leetcode.com/problems/data-stream-as-disjoint-intervals/

* Similar questions

  [57.Insert Interval](https://leetcode.com/problems/insert-interval/)

* solution

  Worth try binary search



#### 53. Maximum subarray

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

* Solution-divide and conquer-not done

  ***worth thinking and doing***



#### 239. Sliding Window Maximum

https://leetcode.com/problems/sliding-window-maximum/

* Solution-deque-not done

  ***worth thinking and doing***

* Solution- dynamic programming-not done

  dont understand



#### 295. Find Median from Data Stream

https://leetcode.com/problems/find-median-from-data-stream/

* Solution-two priority queue

  ***worth thinking and doing***

  虽然heapq是最小堆，但换为负数就好了。

  善用两个heap



8.2

#### 325. Maximum Size Subarray Sum Equals k

https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/

* solution

  > Hint2: Given *S[i]* a partial sum that starts at position *0* and ends at *i*, what can *S[i - k]* tell you ?
  >
  > Hint3: Use HashMap + prefix sum array.

  应该想到既然要使S[i - k]有意义的话，要如何设计dic，这里则是将常规的value用作key



#### 209. Minimum Size Subarray Sum

https://leetcode.com/problems/minimum-size-subarray-sum/

* Solution-two pointer

  思考怎么精简我的做法

* solution-binary search

  not done



#### 238. Product of Array Except Self

https://leetcode.com/problems/product-of-array-except-self/

* solution

  最开始的想法还是维持两个array，这样就可以从O(n^2)降到O(n)



#### 228. Summary Ranges-easy

https://leetcode.com/problems/summary-ranges/





#### 152. Maximum Product Subarray

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

#### 163. Missing Ranges

https://leetcode.com/problems/missing-ranges/

* solution

  对upper bound一个很棒的处理：

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



#### 88. Merge Sorted Array

https://leetcode.com/problems/merge-sorted-array/



#### 75. Sort Colors

https://leetcode.com/problems/sort-colors/

* solution-two pointers

  \#Array Transformation\#常用思路

  想这个思路时，可以先考虑只有两个数的时候再推导到三个数的

* Solution-worth thinking

  本质是我们常用的递归思想，先假设一个小问题解决了，然后假如再来一个数该怎么操作。

  Ref: https://leetcode.wang/leetCode-75-Sort-Colors.html

  ```java
  public void sortColors(int[] nums) {
      int n0 = -1, n1 = -1, n2 = -1;
      int n = nums.length;
      for (int i = 0; i < n; i++) {
          if (nums[i] == 0) {
              n2++;
              nums[n2] = 2;
              n1++;
              nums[n1] = 1;
              n0++;
              nums[n0] = 0;
          } else if (nums[i] == 1) {
              n2++;
              nums[n2] = 2;
              n1++;
              nums[n1] = 1;
          } else if (nums[i] == 2) {
              n2++;
              nums[n2] = 2;
          }
      }
  }
  
  ```

  

#### 283. Move Zeroes

https://leetcode.com/problems/move-zeroes/

* Solution-two pinter

  \#Array Transformation\#

  和75.sort colors一起想



#### 376. Wiggle Subsequence

https://leetcode.com/problems/wiggle-subsequence/

* Solution-greedy
* Solution-dynamic programming-not done



8.5

#### 280. Wiggle Sort

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

  

#### 324. Wiggle Sort II

https://leetcode.com/problems/wiggle-sort-ii/

* Solution-not done!!!



* 





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





### heapq模块

可以参照cs61b-priority queue

#### priority queue基本概念

A priority queue is like a dictionary, it contains entries that each consists of a key and an associated value. However, while a dictionary is used when we want to be able to look up arbitrary key, a priority queue is used to prioritize entries, thus that you can easily access and manipulate the value with the largest/smallest key.

#### 一般的operation

* insert()
* min()
* removeMin()

#### Implementation

Binary heap(i.e. a complete binary tree whose entries satisfy keep-order property. For example, the key of a child always >= the key of its parent, and in this way, 这是一个最小堆)

To store as array, map treenodes to array indices with level-numbering: level-order traversal with root at index 1. In this way, node i's children are 2i and 2i+1, parent is i//2

#### Priority queue in Python

**堆**

> 堆是一种树形数据结构，其中子节点与父节点之间是一种有序关系。最大堆中父节点大于或等于两个子节点，最小堆父节点小于或等于两个子节点。Python的heapq模块实现了一个最小堆。

* 创建堆

  * 用[]初始化
  
* 已有list转化为heap，[heapify()](https://docs.python.org/zh-cn/3/library/heapq.html#heapq.heapify)，heapq.heapify(*list*)
  
* heapq模块可以接受元组对象，默认元组的第一个元素作为`priority`

* heapq.heappush(*heap*, *item*)

  将 *item* 的值加入 *heap* 中，保持堆的不变性。

* heapq.heappop(*heap*)

  弹出并返回 heap 的最小的元素，保持堆的不变性。如果堆为空，抛出 IndexError 。使用 heap[0] ，可以只访问最小的元素而不弹出它。

* heapq.heappushpop(*heap*, *item*)

  将 item 放入堆中，然后弹出并返回 heap 的最小元素。该组合操作比先调用  heappush() 再调用 heappop()运行起来更有效率。

#### Real Priority queue in Python	

[`PriorityQueue`](https://docs.python.org/3/library/queue.html#queue.PriorityQueue)



### Divide and Conquer

Let's follow here a solution template for the divide and conquer problems :

- Define the base case(s).
- Split the problem into subproblems and solve them recursively.
- Merge the solutions for the subproblems to obtain the solution for the original problem.





### The use of comma

Ref: https://leetcode.com/problems/summary-ranges/discuss/63193/6-lines-in-Python

I have these two basic cases:

```python
ranges += [],
r[1:] = n,
```

Why the trailing commas? Because it turns the right hand side into a tuple and I get the same effects as these more common alternatives:

```python
ranges += [[]]
or
ranges.append([])

r[1:] = [n]
```



Without the comma, ...

- `ranges += []` wouldn't add `[]` itself but only its elements, i.e., nothing.
- `r[1:] = n` wouldn't work, because my `n` is not an iterable.

Why do it this way instead of the more common alternatives I showed above? Because it's shorter and faster (according to tests I did a while back).





### map()

map(function, iterable,…)

iterable,…可以传入一个或多个序列

python3返回迭代器