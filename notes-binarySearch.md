### 35. Search Insert Position

Ref: https://leetcode.com/articles/search-insert-position/

Mid faster using bit shift:

```python
pivot = (left + right) >> 1
```

prevent bit flows in java or c++

```java
pivot = left + (right - left) / 2;
```



### 33. Search in Rotated Sorted Array

Ref: https://leetcode.com/problems/search-in-rotated-sorted-array/

先看看我自己本来的code，其实分成两种情况就好



### 153. Find Minimum in Rotated Sorted Array

Ref: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/





### 81. Search in Rotated Sorted Array II

Ref: https://leetcode.com/problems/search-in-rotated-sorted-array-ii/discuss/28195/Python-easy-to-understand-solution-(with-comments).

```python
def search(self, nums, target):
    l, r = 0, len(nums)-1
    while l <= r:
        mid = l + (r-l)//2
        if nums[mid] == target:
            return True
        while l < mid and nums[l] == nums[mid]: # tricky part
            l += 1
        # 只要分两种情况就好了
        # the first half is ordered
        if nums[l] <= nums[mid]:
            # target is in the first half
            if nums[l] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        # the second half is ordered
        else:
            # target is in the second half
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return False
```



### 153. Find Minimum in Rotated Sorted Array

Ref: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/



### 154. Find Minimum in Rotated Sorted Array II

Ref: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/

#### Solution

Ref: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/discuss/167981/Beats-100-Binary-Search-with-Explanations

学会这样分析



### 162. Find Peak Element

https://leetcode.com/problems/find-peak-element/description/

Ref: https://leetcode.com/problems/find-peak-element/discuss/50239/Java-solution-and-explanation-using-invariants





### 34. Find First and Last Position of Element in Sorted Array

https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/

#### Solution

two binary search



### 315. Count of Smaller Numbers After Self

https://leetcode.com/problems/count-of-smaller-numbers-after-self/description/



#### Solution-mergesort

Ref: https://leetcode.com/problems/count-of-smaller-numbers-after-self/discuss/76584/Mergesort-solution

```python
def countSmaller(self, nums):
    def sort(enum):
        half = len(enum) // 2
        if half:
            left, right = sort(enum[:half]), sort(enum[half:])
            for i in range(len(enum))[::-1]:
                if not right or left and left[-1][1] > right[-1][1]:
                  # (not right) or (left and left[-1][1] > right[-1][1])
                    smaller[left[-1][0]] += len(right)
                    enum[i] = left.pop()
                else:
                    enum[i] = right.pop()
        return enum
    smaller = [0] * len(nums)
    sort(list(enumerate(nums)))
    return smaller
```





### 300. Longest Increasing Subsequence

https://leetcode.com/problems/longest-increasing-subsequence/description/

#### Solution-dp-O(N2)

#### Solution-greedy, binary search

https://leetcode.com/problems/longest-increasing-subsequence/discuss/74824/JavaPython-Binary-search-O(nlogn)-time-with-explanation

https://www.cs.princeton.edu/courses/archive/spring13/cos423/lectures/LongestIncreasingSubsequence.pdf





### 354. Russian Doll Envelopes

https://leetcode.com/problems/russian-doll-envelopes/description/

#### Solution

https://leetcode.com/problems/russian-doll-envelopes/discuss/82796/A-Trick-to-solve-this-problem.

but why????



### 4. Median of Two Sorted Arrays

https://leetcode.com/problems/median-of-two-sorted-arrays/

#### Solution-worth

https://leetcode.com/problems/median-of-two-sorted-arrays/discuss/2481/Share-my-O(log(min(mn)))-solution-with-explanation



### 69. Sqrt(x)

https://leetcode.com/problems/sqrtx/description/



### 167. Two Sum II - Input array is sorted

https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/



### 50. Pow(x, n)

https://leetcode.com/problems/powx-n/description/

#### Solution-worth

https://leetcode.com/problems/powx-n/discuss/19560/Shortest-Python-Guaranteed

recursive&iterative are both worh reviewing

***My explanation of why the iterative one work:***

Let's get rid of the the math theories, instead, just think of the conversion between binary number and its according base 10 number.

First of all, the code below can be regarded as converting a base 10 number to binary number with a few extra step to modify *pow*.

```python
while n:
    if n & 1:
        pow *= x
    x *= x
    n >>= 1
```

To illustrate, say, initially, n=11, pow = 1

11/2 = 5 remains 1, X^1, pow = x^1

5/2 = 2 remains 1, X^2, pow = x^3

2/2 = 1 remains 0, X^4

1/2 = 0 remains 1, X^8, pow = x^11

Now, you may wonder, why do we update *pow* (multiplied by current X) when there is 1 remaining, in other words, when should we add the current power(1, 2, 8) to get a sum that is our final power(11), in other other words, how we can get a sum that is made of the aggregation of different, non-duplicate powers of 2, at this time, just think backwards, how we can convert binary 1011 to its base10 value 11, and here comes the answer.



### 367. Valid Perfect Square

https://leetcode.com/problems/valid-perfect-square/description/









## Binary search

* 对left和right的替换

  我的习惯是，right = len(n)-1, while left>right, left = mid + 1, right  = mid

* 先对特殊值处理会更快（374）

* 使用inorder traversal来实现（744）

  to be continued

* 使用two pointer来实现（167

  大概比较适合有两个list的，类似的概念，总之用到两个元素

  

* 避免溢出（278）

  > 那就是如果left和right都特别大的话，那么left+right可能会溢出，我们的处理方法就是变成left + (right - left) / 2，很好的避免的溢出问题

  Ref: https://www.cnblogs.com/grandyang/p/4790469.html

