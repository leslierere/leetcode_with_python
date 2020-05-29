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







### 69. Sqrt(x)

https://leetcode.com/problems/sqrtx/description/



### 167. Two Sum II - Input array is sorted

https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/



### 50. Pow(x, n)

https://leetcode.com/problems/powx-n/description/



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

