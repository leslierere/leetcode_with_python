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

did@21.6.29

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)-1
        
        while left<right:
            mid = (left+right)//2
            if nums[mid]==target:
                return mid
            elif target < nums[mid]:
                right = mid-1
            else:
                left = mid+1
                
        if nums[left]<target:
            return left+1
        else:
            return left
```



### 33. Search in Rotated Sorted Array

Ref: https://leetcode.com/problems/search-in-rotated-sorted-array/

 [this](https://leetcode.com/problems/search-in-rotated-sorted-array/discuss/14437/Python-binary-search-solution-O(logn)-48ms)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)-1
        
        while left<right:
            mid = (left+right)//2
            if nums[mid]==target:
                return mid
            if nums[left]<=nums[mid]: # left part is in order, has to be equal here, as the left part can be only one element
                if target<nums[mid] and target>=nums[left]:
                    right = mid-1
                else:
                    left = mid+1
            else:
                if target>nums[mid] and target<=nums[right]:
                    left = mid+1
                else:
                    right = mid-1
        if target==nums[left]:
            return left
        else:
            return -1
```



A really [cool](https://leetcode.com/problems/search-in-rotated-sorted-array/discuss/14435/Clever-idea-making-it-simple) one that I don't understand yet.





### 81. Search in Rotated Sorted Array II-$

#### Solution

我觉得这样比较清晰

did@21.7.1

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        left = 0
        right = len(nums)-1
        
        while left<right:
            mid = (left+right)//2
            if nums[mid]==target:
                return True
            elif nums[left]<nums[mid]:
                if target<nums[mid] and target>=nums[left]:
                    right = mid-1
                else:
                    left = mid+1
            elif nums[mid]<nums[right]:
                if target<=nums[right] and target>nums[mid]:
                    left = mid+1
                else:
                    right = mid-1
            elif nums[left]==nums[mid]:
                left+=1
            elif nums[right]==nums[mid]:
                right-=1
                
        return nums[left]==target or nums[right]==target
```



Ref: https://leetcode.com/problems/search-in-rotated-sorted-array-ii/discuss/28195/Python-easy-to-understand-solution-(with-comments).

For the subarray where the break locates, the invariant is the array[0]>=array[-1]

If array[0]<array[-1], we can know it is ordered. Here in the solution, the equal will only happen, if left index equals to the mid, and they are still ordered.

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

did@21.6.16

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        
        left = 0
        right = len(nums)-1
        
        while left<right:
            mid = (left+right)//2
            if nums[left]>nums[mid]: # the left is in disorder
                right = mid
            elif nums[mid]>nums[right]: # the right is in disorder
                left = mid+1
            else:
                return nums[left]
                
        return nums[left]
```



### 154. Find Minimum in Rotated Sorted Array II

Ref: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/

#### Solution

Ref: https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/discuss/167981/Beats-100-Binary-Search-with-Explanations

学会这样分析

did@ 21.6.16

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums)-1
        
        while left<right:
            mid = (left+right)//2
            if nums[left]>nums[mid]: # left side is disorder
                right = mid
            elif nums[mid]>nums[right]:
                left = mid+1
            elif nums[mid]==nums[right]:
                right-=1
            elif left!=mid and nums[left]==nums[mid]:
                left+=1
            else:
                return nums[left]
            
        return nums[left]
```





### 162. Find Peak Element

https://leetcode.com/problems/find-peak-element/description/

#### Solution-$$

Ref: https://leetcode.wang/leetcode-162-Find-Peak-Element.html

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left = 0
        right = len(nums)-1
        
        while left<right:
            mid1 = (left+right)//2
            mid2 = (left+right)//2+1
            
            if nums[mid1]<nums[mid2]:
                left = mid2
            else: # nums[mid1]>nums[mid2]
                right = mid1
                
        return left
```





### 34. Find First and Last Position of Element in Sorted Array

https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/

#### Solution

two binary search, also if we find the first, we don't need to start from zero when we try to find the upper bound.

did@21.5.24, feels messy

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0:
            return [-1, -1]
        
        left_low = right_low= 0
        left_high = right_high = len(nums)-1
        
        while left_low<left_high:
            mid = (left_low+left_high)//2
            if target<nums[mid]:
                left_high = mid
            elif target>nums[mid]:
                left_low = mid+1
            else: # nums[mid]==target
                if mid>0 and nums[mid-1]==target:
                    left_high = mid-1
                else:
                    left_low = left_high = mid
                    break
            
        if left_low!=left_high or nums[left_low]!=target:
            return [-1, -1]
        
        while right_low<right_high:
            mid = (right_low+right_high)//2
            if target>nums[mid]:
                right_low = mid+1
            elif target<nums[mid]:
                right_high = mid-1
            else:
                if nums[mid+1]==target:
                    right_low = mid+1
                else:
                    right_low = mid
                    break
                
        
        return [left_low, right_low]
```





感觉这个代码清晰点, ref: https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/discuss/14734/Easy-java-O(logn)-solution

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0:
            return [-1, -1]
        
        left_low = right_low= 0
        left_high = right_high = len(nums)-1
        idx1=idx2 = -1
        
        
        while left_low<=left_high:
            mid = (left_low+left_high)//2
            if target==nums[mid]:
                idx1 = mid
            if target<=nums[mid]:
                left_high = mid-1
            else:
                left_low = mid+1
            
        if idx1==-1:
            return [-1, -1]
        
        while right_low<=right_high:
            mid = (right_low+right_high)//2
            if target==nums[mid]:
                idx2 = mid
            if target>=nums[mid]:
                right_low = mid+1
            else:
                right_high = mid-1
                
        return [idx1, idx2]
```





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

#### Solution-find kth smallest number

Ref: https://leetcode.wang/leetCode-4-Median-of-Two-Sorted-Arrays.html



```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        mid = (len(nums1) + len(nums2)-1)//2
        # for any time, we calculate the mid index
        # if odd, we can get the exact index of the median
        # if even, we get the index of the first half part of the median        
        
        if (len(nums1) + len(nums2))%2 ==1:
            return self.findKth(nums1, nums2, mid+1)
        else:
            return (self.findKth(nums1, nums2, mid+1)+self.findKth(nums1, nums2, mid+2))/2
        
    
    def findKth(self, nums1, nums2, k):
        # the kth element in the array, satrting from 1
        if len(nums1)==0:
            return nums2[k-1]
        if len(nums2)==0:
            return nums1[k-1]
        if k==1:
            return min(nums1[0], nums2[0])
        
        mid_idx = k//2 - 1 # if k is odd, say 7, we wanna compare the lower 3 numbers in each array, so we round down, if we round up,we are comparing the lower 4 in each, which exceeds 7 in total, and we would like to keep an invariant that is, the left 2 halves are always less than our target k
        # index1+1+index2+1<=k+1
        # index1+index2 <= k-1
        # since we always give up the remain
        index1 = min(len(nums1)-1, mid_idx)
        index2 = min(len(nums2)-1, mid_idx)
        
        if nums1[index1]<nums2[index2]:
            return self.findKth(nums1[index1+1:], nums2, k-index1-1)
        else: # if equal, it is the same to give up either of them
            return self.findKth(nums1, nums2[index2+1:], k-index2-1)
```



#### Solution-also binary search

Ref: https://leetcode.wang/leetCode-4-Median-of-Two-Sorted-Arrays.html

解法4



### 69. Sqrt(x)

https://leetcode.com/problems/sqrtx/description/

#### Solution-iterative

did@21.7.1

similar to the normal iterative solution of problem 50.

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        
        cum_num = 0
        while (cum_num+1)**2<=x:
            number = 1
            while (cum_num+number*2)**2<=x:
                number*=2
                
            cum_num+=number
            
        return cum_num
```

#### Solution-binary search

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        number = x
        left = 0
        right = x
        mid = 0
        while left<right:
            mid = (left+right)>>1
            result = mid*mid
            if result<=x and (mid+1)*(mid+1)>x:
                return mid
            elif result<x:
                left = mid+1
            else:
                right = mid
                
        return left
```





### 167. Two Sum II - Input array is sorted

https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/



### 50. Pow(x, n)

https://leetcode.com/problems/powx-n/description/

#### Solution-recursive-👖

https://leetcode.com/problems/powx-n/discuss/19560/Shortest-Python-Guaranteed

```python
class Solution:
    def myPow(self, x, n):
        if not n:
            return 1
        if n < 0:
            return 1 / self.myPow(x, -n)
        if n % 2:
            return x * self.myPow(x, n-1)
        return self.myPow(x*x, n/2)# this is so cool
```

#### Solution-iterative-👖

***My explanation of why the iterative one work:*** https://leetcode.com/problems/powx-n/discuss/19560/Shortest-Python-Guaranteed/647115

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

Now, you may wonder, why do we update *pow* (multiplied by current X) when there is 1 remaining, in other words, when should we add the current power (1=2^0, 2=2^1, 8=2^3) to get a sum that is our final power(11), in other other words, how we can get a target power(11 in this case) that is the sum of different, non-duplicate powers of 2, at this time, just think backwards, how we can convert binary 1011 to its base10 value 11, and here comes the answer.

#### Solution-iterative, 常规思路

did@21.7.1

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n==0:
            return 1
        if x==0:
            return x
        
        negative = False
        if n<0:
            n = -n
            negative = True
            
        res = 1
        while n>0:
            times = 1
            subres = x
            while times*2<=n:
                subres = subres*subres
                times*=2
                
            res = subres*res
            n -= times
        
        if negative:
            return 1/res
        else:
            return res
```



### 1922. Count Good Numbers

https://leetcode.com/problems/count-good-numbers/

#### Solution

My pow, refer to problem 50

```python
class Solution:
    def countGoodNumbers(self, n: int) -> int:
        # count = 1
        even_nums = 5
        prime_nums = 4
        modu = 10**9 + 7
        even_pos = (n+1)//2
        odd_pos = n//2
        
        
        count = self.myPow(even_nums*prime_nums, odd_pos, modu)
        if even_pos>odd_pos:
            count*=even_nums
            
        return count%modu 
    

    def myPow(self, x, n, modu):
        if not n:
            return 1
        if n % 2:
            return (x%modu * self.myPow(x, n-1, modu))%modu
        return (self.myPow(x*x%modu, n//2, modu))%modu
```







### 367. Valid Perfect Square

https://leetcode.com/problems/valid-perfect-square/description/



### 29. Divide Two Integers

https://leetcode.com/problems/divide-two-integers/

#### Solution-binary search

```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        if dividend==0:
            return 0
        elif divisor == 1:
            return dividend
        elif divisor == -1:
            if dividend==-2**31:
                return 2**31-1
            return -dividend
        
        if dividend>0 and divisor>0:
            return self.helper(dividend, divisor)
        elif dividend<0 and divisor<0:
            return self.helper(-dividend, -divisor)
        else:
            return -self.helper(abs(dividend), abs(divisor))
                
    def helper(self, dividend, divisor):
        if dividend<divisor:
            return 0
        elif dividend==divisor:
            return 1
        origin_divisor = divisor
        times = 1
        while dividend>divisor+divisor:
            divisor+=divisor
            times+=times
            
        return times+self.helper(dividend-divisor, origin_divisor)
```







## Binary search

* 对left和right的替换

  我的习惯是，right = len(n)-1, while left>right, left = mid + 1, right  = mid

  有一种情况会出现死循环。。。应该是left = mid, right = mid, 但是到了最后缩到两个元素, 算mid总是会落到第一个元素，而他又不等于target，但left = mid+1就可以避免这点

* 先对特殊值处理会更快（374）

* 使用inorder traversal来实现（744）

  to be continued

* 使用two pointer来实现（167

  大概比较适合有两个list的，类似的概念，总之用到两个元素

  

* 避免溢出（278）

  > 那就是如果left和right都特别大的话，那么left+right可能会溢出，我们的处理方法就是变成left + (right - left) / 2，很好的避免的溢出问题

  Ref: https://www.cnblogs.com/grandyang/p/4790469.html

